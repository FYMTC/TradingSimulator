//! M3: heterogeneous strategy agents plus the macro scenario engine.
//!
//! Two additions over M2's learning investors:
//!
//! - **Strategy diversity** in the spirit of the design report's agent
//!   taxonomy: market makers quoting both sides around the mid with
//!   inventory-skewed prices, trend followers trading fast/slow moving
//!   average crossovers of the trade tape, mean reverters fading
//!   deviations from a trailing anchor, and fundamentalists anchoring on
//!   a private, drifting estimate of fair value.  The zero-intelligence
//!   crowd still supplies the liquidity base.
//! - **A scenario engine** (the "AI Director" of the design report): a
//!   Markov chain over {calm, bull, bear, crisis} that switches on an
//!   exponential clock and only *modulates agent parameters* - wake-up
//!   activity, market-maker spreads, fundamental drift - never prices or
//!   the book.  Regime-driven activity is what generates volatility
//!   clustering and the volume-volatility relation on top of the M1
//!   stylized facts, and an intraday U-shaped activity profile shapes
//!   session volume like a real trading day.
//!
//! Everything runs on the shared deterministic [`crate::engine`] core:
//! one RNG, strictly ordered events, canonical replay.

use crate::bar::{Bar, TapePrint, aggregate_bars};
use crate::engine::{self, KIND_SETTLE, MarketCore, MarketDriver};
use crate::sim::{LOT_SIZE, NoiseAgentParams, QuoteView, noise_order_price, noise_order_quantity};
use crate::{
    AccountId, Event, Exchange, LimitOrderRequest, Money, OrderId, Price, Quantity, Side, SimTime,
};

const KIND_NOISE: u8 = 1;
const KIND_MM: u8 = 2;
const KIND_TREND: u8 = 3;
const KIND_MEAN_REV: u8 = 4;
const KIND_FUND: u8 = 5;
/// The scenario engine's own clock; not an agent.
const KIND_REGIME: u8 = 6;

/// Macro market regimes the scenario engine switches between.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Regime {
    Calm = 0,
    Bull = 1,
    Bear = 2,
    Crisis = 3,
}

impl Regime {
    fn index(self) -> usize {
        self as usize
    }

    /// The three other regimes in canonical order; switch rows index into
    /// this order.
    fn others(self) -> [Regime; 3] {
        match self {
            Regime::Calm => [Regime::Bull, Regime::Bear, Regime::Crisis],
            Regime::Bull => [Regime::Calm, Regime::Bear, Regime::Crisis],
            Regime::Bear => [Regime::Calm, Regime::Bull, Regime::Crisis],
            Regime::Crisis => [Regime::Calm, Regime::Bull, Regime::Bear],
        }
    }
}

/// Parameters of the Markov scenario engine.  Sojourns are exponential
/// with the configured mean; when one ends the chain jumps to a
/// *different* regime drawn from the row of the current one.
///
/// The engine only modulates agent parameters (wake-up activity,
/// market-maker spreads, fundamental drift), never prices - regime-driven
/// activity differences are what generate volatility clustering and the
/// volume-volatility relation in the tape.
#[derive(Clone, Debug)]
pub struct RegimeParams {
    /// Mean sojourn per regime, in simulated milliseconds.
    pub mean_sojourn_ms: SimTime,
    /// Wake-up activity multiplier per regime, indexed by [`Regime`].
    pub activity: [f64; 4],
    /// Market-maker half-spread multiplier per regime.
    pub mm_half_spread: [f64; 4],
    /// Fundamental value drift per fundamental wake, in ticks, per regime.
    pub fund_drift_ticks: [f64; 4],
    /// Fundamental value noise per wake, in ticks.
    pub fund_vol_ticks: f64,
    /// Switch probabilities when a sojourn ends, per source regime,
    /// ordered as [`Regime::others`].
    pub switch_rows: [[f64; 3]; 4],
}

impl Default for RegimeParams {
    fn default() -> Self {
        Self {
            mean_sojourn_ms: 600_000,
            activity: [0.55, 1.5, 1.3, 2.4],
            mm_half_spread: [0.8, 1.0, 1.2, 2.2],
            fund_drift_ticks: [0.0, 0.8, -0.8, -2.0],
            fund_vol_ticks: 2.0,
            switch_rows: [
                [0.45, 0.45, 0.10], // calm -> bull / bear / crisis
                [0.55, 0.25, 0.20], // bull -> calm / bear / crisis
                [0.55, 0.25, 0.20], // bear -> calm / bull / crisis
                [0.60, 0.20, 0.20], // crisis -> calm / bull / bear
            ],
        }
    }
}

/// Intraday activity profile: U-shaped - busy at the open and the close,
/// quiet over midday, like a real trading session.  Ranges over
/// roughly [0.5, 1.5].
pub fn intraday_multiplier(now_ms: SimTime, day_length_ms: SimTime) -> f64 {
    let f = (now_ms % day_length_ms) as f64 / day_length_ms as f64;
    0.5 + (((f - 0.5).abs()) * 2.0).powi(2)
}

/// A two-sided liquidity provider with an inventory skew.
#[derive(Clone, Debug)]
struct MmAgent {
    account_id: AccountId,
    resting: Vec<OrderId>,
    wake_rate_per_second: f64,
    quote_lots: i64,
    base_half_spread_ticks: f64,
    /// Quote centre shifts this many ticks per lot of inventory deviation
    /// (long inventory -> quote lower to shed it).
    inv_skew_ticks_per_lot: f64,
    target_lots: i64,
}

/// Trades fast/slow moving-average crossovers of the trade tape.
#[derive(Clone, Debug)]
struct TrendAgent {
    account_id: AccountId,
    resting: Vec<OrderId>,
    wake_rate_per_second: f64,
    fast_n: usize,
    slow_n: usize,
    band_ticks: i64,
    lots: i64,
    position_cap_lots: i64,
}

/// Fades deviations of the last price from a trailing anchor.
#[derive(Clone, Debug)]
struct MeanRevAgent {
    account_id: AccountId,
    resting: Vec<OrderId>,
    wake_rate_per_second: f64,
    anchor_window: usize,
    band_ticks: i64,
    lots: i64,
    position_cap_lots: i64,
}

/// Anchors on a private, drifting estimate of fair value.
#[derive(Clone, Debug)]
struct FundamentalAgent {
    account_id: AccountId,
    resting: Vec<OrderId>,
    wake_rate_per_second: f64,
    /// Private fair-value estimate in ticks; evolves with the regime.
    value_ticks: f64,
    band_ticks: i64,
    lots: i64,
    position_cap_lots: i64,
}

/// Full configuration of one heterogeneous market run.
#[derive(Clone, Debug)]
pub struct HeteroMarketConfig {
    pub symbol: String,
    /// Reference price in ticks used before the first trade forms a mid.
    pub ref_price: Price,
    /// Simulated milliseconds per trading day; T+1 settles on each
    /// boundary and the intraday activity profile restarts.
    pub day_length_ms: SimTime,
    pub seed: u64,
    pub agent_cash: Money,
    /// Settled, sellable shares seeded into every agent.
    pub agent_seed_shares: Quantity,
    pub n_noise: usize,
    pub n_market_makers: usize,
    pub n_trend: usize,
    pub n_mean_revert: usize,
    pub n_fundamental: usize,
    pub noise: NoiseAgentParams,
    pub regime: RegimeParams,
}

impl Default for HeteroMarketConfig {
    fn default() -> Self {
        Self {
            symbol: "600000.SH".to_owned(),
            ref_price: 1_000,
            day_length_ms: 1_200_000,
            seed: 1,
            agent_cash: 1_000_000_000_000,
            agent_seed_shares: 1_000_000,
            n_noise: 48,
            n_market_makers: 4,
            n_trend: 8,
            n_mean_revert: 8,
            n_fundamental: 4,
            noise: NoiseAgentParams::default(),
            regime: RegimeParams::default(),
        }
    }
}

/// A heterogeneous market: strategy agents of the design-report taxonomy
/// plus the zero-intelligence crowd, all driven by the scenario engine.
#[derive(Clone, Debug)]
pub struct HeteroMarket {
    config: HeteroMarketConfig,
    core: MarketCore,
    regime: Regime,
    /// Every regime transition with its timestamp, for diagnostics.
    regime_history: Vec<(SimTime, Regime)>,
    noise_resting: Vec<Vec<OrderId>>,
    mms: Vec<MmAgent>,
    trends: Vec<TrendAgent>,
    mean_revs: Vec<MeanRevAgent>,
    funds: Vec<FundamentalAgent>,
}

impl HeteroMarket {
    /// Builds the market, funds the accounts, gives every strategy agent
    /// its deterministic personality, and schedules all first wake-ups,
    /// the first regime switch, and the first T+1 settlement boundary.
    pub fn new(config: HeteroMarketConfig) -> Self {
        assert!(config.n_noise >= 2, "need at least two noise agents");
        let seed_lots = config.agent_seed_shares / LOT_SIZE;
        let n_agents = config.n_noise
            + config.n_market_makers
            + config.n_trend
            + config.n_mean_revert
            + config.n_fundamental;

        let mut core = MarketCore::new(
            config.symbol.clone(),
            config.ref_price,
            config.day_length_ms,
            config.seed,
        );
        for agent in 0..n_agents {
            core.add_funded_account(
                agent as AccountId,
                config.agent_cash,
                config.agent_seed_shares,
            );
        }

        // Deterministic personalities, drawn in fixed population order so
        // the whole market is reproducible from the seed.
        let mut next_id = config.n_noise as AccountId;
        let mut mms = Vec::with_capacity(config.n_market_makers);
        for _ in 0..config.n_market_makers {
            mms.push(MmAgent {
                account_id: next_id,
                resting: Vec::new(),
                wake_rate_per_second: 3.0 + 3.0 * core.rng.next_f64(),
                quote_lots: core.rng.uniform_int(2, 8),
                base_half_spread_ticks: 1.0 + 2.0 * core.rng.next_f64(),
                inv_skew_ticks_per_lot: 0.3 + 0.5 * core.rng.next_f64(),
                target_lots: seed_lots,
            });
            next_id += 1;
        }
        let mut trends = Vec::with_capacity(config.n_trend);
        for _ in 0..config.n_trend {
            let fast_n = core.rng.uniform_int(8, 20) as usize;
            let lots = core.rng.uniform_int(2, 6);
            trends.push(TrendAgent {
                account_id: next_id,
                resting: Vec::new(),
                wake_rate_per_second: 0.10 + 0.15 * core.rng.next_f64(),
                fast_n,
                slow_n: fast_n * 6,
                band_ticks: core.rng.uniform_int(1, 3),
                lots,
                position_cap_lots: 6 * lots,
            });
            next_id += 1;
        }
        let mut mean_revs = Vec::with_capacity(config.n_mean_revert);
        for _ in 0..config.n_mean_revert {
            let lots = core.rng.uniform_int(2, 5);
            mean_revs.push(MeanRevAgent {
                account_id: next_id,
                resting: Vec::new(),
                wake_rate_per_second: 0.05 + 0.10 * core.rng.next_f64(),
                anchor_window: 600,
                band_ticks: core.rng.uniform_int(8, 20),
                lots,
                position_cap_lots: 6 * lots,
            });
            next_id += 1;
        }
        let mut funds = Vec::with_capacity(config.n_fundamental);
        for _ in 0..config.n_fundamental {
            let lots = core.rng.uniform_int(2, 5);
            funds.push(FundamentalAgent {
                account_id: next_id,
                resting: Vec::new(),
                wake_rate_per_second: 0.02 + 0.04 * core.rng.next_f64(),
                value_ticks: config.ref_price as f64 + core.rng.standard_normal() * 10.0,
                band_ticks: core.rng.uniform_int(10, 30),
                lots,
                position_cap_lots: 6 * lots,
            });
            next_id += 1;
        }

        let mut market = Self {
            config,
            core,
            regime: Regime::Calm,
            regime_history: vec![(0, Regime::Calm)],
            noise_resting: vec![Vec::new(); n_agents],
            mms,
            trends,
            mean_revs,
            funds,
        };
        for agent in 0..market.config.n_noise {
            let rate = market.config.noise.wake_rate_per_second
                * market.activity_multiplier(market.core.now_ms());
            let gap = market.core.rng.poisson_gap_ms(rate);
            market.core.schedule(gap, KIND_NOISE, agent);
        }
        for (index, agent) in market.mms.iter().enumerate() {
            let gap = market.core.rng.poisson_gap_ms(agent.wake_rate_per_second);
            market.core.schedule(gap, KIND_MM, index);
        }
        for (index, agent) in market.trends.iter().enumerate() {
            let rate =
                agent.wake_rate_per_second * market.activity_multiplier(market.core.now_ms());
            let gap = market.core.rng.poisson_gap_ms(rate);
            market.core.schedule(gap, KIND_TREND, index);
        }
        for (index, agent) in market.mean_revs.iter().enumerate() {
            let rate =
                agent.wake_rate_per_second * market.activity_multiplier(market.core.now_ms());
            let gap = market.core.rng.poisson_gap_ms(rate);
            market.core.schedule(gap, KIND_MEAN_REV, index);
        }
        for (index, agent) in market.funds.iter().enumerate() {
            let rate =
                agent.wake_rate_per_second * market.activity_multiplier(market.core.now_ms());
            let gap = market.core.rng.poisson_gap_ms(rate);
            market.core.schedule(gap, KIND_FUND, index);
        }
        // The scenario engine's own exponential sojourn clock.
        let switch_rate = 1000.0 / market.config.regime.mean_sojourn_ms as f64;
        let gap = market.core.rng.poisson_gap_ms(switch_rate);
        market.core.schedule(gap, KIND_REGIME, 0);
        market
            .core
            .schedule(market.config.day_length_ms, KIND_SETTLE, 0);
        market
    }

    /// The underlying exchange.
    pub fn exchange(&self) -> &Exchange {
        &self.core.exchange
    }

    /// Current simulation time in milliseconds.
    pub fn now_ms(&self) -> SimTime {
        self.core.now_ms()
    }

    /// The executed trade tape in order.
    pub fn tape(&self) -> &[TapePrint] {
        self.core.tape()
    }

    /// Post-trade spread samples (in ticks) collected during the run.
    pub fn spread_samples_ticks(&self) -> &[i64] {
        self.core.spread_samples_ticks()
    }

    /// Aggregates the tape into OHLCV bars of `width_ms`.
    pub fn bars(&self, width_ms: SimTime) -> Vec<Bar> {
        aggregate_bars(self.core.tape().iter().copied(), width_ms)
    }

    /// Submissions rejected by risk checks so far (diagnostics).
    pub fn rejected_submits(&self) -> usize {
        self.core.rejected_submits()
    }

    /// The recorded event log, replayable through [`Exchange::replay`].
    pub fn replay_log(&self) -> &[Event] {
        self.core.replay_log()
    }

    /// The current macro regime.
    pub fn regime(&self) -> Regime {
        self.regime
    }

    /// Every regime transition with its timestamp, starting with the
    /// initial calm regime at time zero.
    pub fn regime_history(&self) -> &[(SimTime, Regime)] {
        &self.regime_history
    }

    /// Processes every event scheduled at or before `target_ms`.
    pub fn run_until(&mut self, target_ms: SimTime) {
        engine::run_until(self, target_ms);
    }

    /// Combined activity multiplier of the current regime and the
    /// intraday profile; market makers are exempt (they quote all day).
    fn activity_multiplier(&self, now_ms: SimTime) -> f64 {
        self.config.regime.activity[self.regime.index()]
            * intraday_multiplier(now_ms, self.config.day_length_ms)
    }

    // ------------------------------------------------------------------
    // Noise agents: identical behaviour to the M1 market, at a
    // regime- and session-modulated tempo.
    // ------------------------------------------------------------------

    fn wake_noise(&mut self, agent: usize, now_ms: SimTime) {
        let mut live = std::mem::take(&mut self.noise_resting[agent]);
        live.retain(|id| self.core.exchange.book().order(*id).is_some());

        let should_cancel = !live.is_empty()
            && self
                .core
                .rng
                .bernoulli(self.config.noise.cancel_probability);
        if should_cancel {
            let index = self.core.rng.uniform_int(0, live.len() as i64 - 1) as usize;
            let order_id = live.swap_remove(index);
            self.noise_resting[agent] = live;
            self.core
                .cancel_tracked(agent as AccountId, order_id, now_ms);
        } else {
            if let Some(order_id) = self.place_noise_order(agent, now_ms) {
                live.push(order_id);
            }
            self.noise_resting[agent] = live;
        }

        let rate = self.config.noise.wake_rate_per_second * self.activity_multiplier(now_ms);
        let gap = self.core.rng.poisson_gap_ms(rate);
        let next = now_ms.checked_add(gap).expect("sim time overflow");
        self.core.schedule(next, KIND_NOISE, agent);
    }

    fn place_noise_order(&mut self, agent: usize, now_ms: SimTime) -> Option<OrderId> {
        let side = if self.core.rng.bernoulli(0.5) {
            Side::Buy
        } else {
            Side::Sell
        };
        let aggressive = self
            .core
            .rng
            .bernoulli(self.config.noise.aggressive_probability);
        let view = QuoteView {
            best_bid: self.core.exchange.book().best_bid(),
            best_ask: self.core.exchange.book().best_ask(),
            last_trade: self.core.last_trade_price(),
            ref_price: self.config.ref_price,
        };
        let price = noise_order_price(
            view,
            &mut self.core.rng,
            &self.config.noise,
            side,
            aggressive,
        );
        if price <= 0 {
            return None;
        }
        let quantity = noise_order_quantity(&mut self.core.rng, &self.config.noise);
        let request = LimitOrderRequest {
            account_id: agent as AccountId,
            side,
            limit_price: price,
            quantity,
        };
        self.core.submit_and_track(request, now_ms)
    }

    // ------------------------------------------------------------------
    // Market makers.
    // ------------------------------------------------------------------

    fn wake_mm(&mut self, index: usize, now_ms: SimTime) {
        let account_id = self.mms[index].account_id;
        // Refresh, then cancel everything: quotes are re-placed from
        // scratch each wake around the current mid and inventory.
        self.mms[index]
            .resting
            .retain(|id| self.core.exchange.book().order(*id).is_some());
        for order_id in std::mem::take(&mut self.mms[index].resting) {
            self.core.cancel_tracked(account_id, order_id, now_ms);
        }

        let mm = &self.mms[index];
        let mid = self.core.mark_price();
        let half = ((mm.base_half_spread_ticks
            * self.config.regime.mm_half_spread[self.regime.index()])
        .round() as i64)
            .max(1);
        let inv_dev = self.agent_lots(account_id) - mm.target_lots;
        let skew = -(mm.inv_skew_ticks_per_lot * inv_dev as f64).round() as i64;
        let quote_lots = mm.quote_lots;

        // Two-sided quote around the skewed centre, never crossing the
        // opposite touch.
        let mut bid = mid - half + skew;
        let mut ask = mid + half + skew;
        if let Some(best_ask) = self.core.exchange.book().best_ask() {
            bid = bid.min(best_ask - 1);
        }
        if let Some(best_bid) = self.core.exchange.book().best_bid() {
            ask = ask.max(best_bid + 1);
        }
        bid = bid.max(1);
        if ask <= bid {
            ask = bid + 1;
        }

        let request = LimitOrderRequest {
            account_id,
            side: Side::Buy,
            limit_price: bid,
            quantity: quote_lots * LOT_SIZE,
        };
        if let Some(order_id) = self.core.submit_and_track(request, now_ms) {
            self.mms[index].resting.push(order_id);
        }
        let sell_lots = quote_lots.min(self.sellable_lots(account_id));
        if sell_lots > 0 {
            let request = LimitOrderRequest {
                account_id,
                side: Side::Sell,
                limit_price: ask,
                quantity: sell_lots * LOT_SIZE,
            };
            if let Some(order_id) = self.core.submit_and_track(request, now_ms) {
                self.mms[index].resting.push(order_id);
            }
        }

        // Market makers quote all day: no activity modulation.
        let gap = self
            .core
            .rng
            .poisson_gap_ms(self.mms[index].wake_rate_per_second);
        let next = now_ms.checked_add(gap).expect("sim time overflow");
        self.core.schedule(next, KIND_MM, index);
    }

    // ------------------------------------------------------------------
    // Trend followers.
    // ------------------------------------------------------------------

    fn wake_trend(&mut self, index: usize, now_ms: SimTime) {
        let account_id = self.trends[index].account_id;
        self.trends[index]
            .resting
            .retain(|id| self.core.exchange.book().order(*id).is_some());
        for order_id in std::mem::take(&mut self.trends[index].resting) {
            self.core.cancel_tracked(account_id, order_id, now_ms);
        }

        let (fast_n, slow_n, band_ticks, lots, cap) = {
            let agent = &self.trends[index];
            (
                agent.fast_n,
                agent.slow_n,
                agent.band_ticks,
                agent.lots,
                agent.position_cap_lots,
            )
        };
        if let Some((fast, slow)) = self.moving_averages(fast_n, slow_n) {
            let signal = fast - slow;
            let mid = self.core.mark_price();
            let deviation = self.agent_lots(account_id) - self.seed_lots();
            if signal > band_ticks as f64 {
                let buy_lots = lots.min((cap - deviation).max(0));
                if buy_lots > 0 {
                    let ask = self.core.exchange.book().best_ask().unwrap_or(mid + 1);
                    let request = LimitOrderRequest {
                        account_id,
                        side: Side::Buy,
                        limit_price: ask,
                        quantity: buy_lots * LOT_SIZE,
                    };
                    if let Some(order_id) = self.core.submit_and_track(request, now_ms) {
                        self.trends[index].resting.push(order_id);
                    }
                }
            } else if signal < -band_ticks as f64 {
                let sell_lots = lots
                    .min(self.sellable_lots(account_id))
                    .min((deviation + cap).max(0));
                if sell_lots > 0 {
                    let bid = self
                        .core
                        .exchange
                        .book()
                        .best_bid()
                        .unwrap_or((mid - 1).max(1));
                    let request = LimitOrderRequest {
                        account_id,
                        side: Side::Sell,
                        limit_price: bid,
                        quantity: sell_lots * LOT_SIZE,
                    };
                    if let Some(order_id) = self.core.submit_and_track(request, now_ms) {
                        self.trends[index].resting.push(order_id);
                    }
                }
            }
        }

        let rate = self.trends[index].wake_rate_per_second * self.activity_multiplier(now_ms);
        let gap = self.core.rng.poisson_gap_ms(rate);
        let next = now_ms.checked_add(gap).expect("sim time overflow");
        self.core.schedule(next, KIND_TREND, index);
    }

    // ------------------------------------------------------------------
    // Mean reverters.
    // ------------------------------------------------------------------

    fn wake_mean_rev(&mut self, index: usize, now_ms: SimTime) {
        let account_id = self.mean_revs[index].account_id;
        self.mean_revs[index]
            .resting
            .retain(|id| self.core.exchange.book().order(*id).is_some());
        for order_id in std::mem::take(&mut self.mean_revs[index].resting) {
            self.core.cancel_tracked(account_id, order_id, now_ms);
        }

        let (window, band_ticks, lots, cap) = {
            let agent = &self.mean_revs[index];
            (
                agent.anchor_window,
                agent.band_ticks,
                agent.lots,
                agent.position_cap_lots,
            )
        };
        let tape = self.core.tape();
        if tape.len() >= 50 {
            let take = window.min(tape.len());
            let anchor: f64 = tape[tape.len() - take..]
                .iter()
                .map(|print| print.price as f64)
                .sum::<f64>()
                / take as f64;
            let price = tape[tape.len() - 1].price as f64;
            let deviation_from_anchor = price - anchor;
            let mid = self.core.mark_price();
            let deviation = self.agent_lots(account_id) - self.seed_lots();
            if deviation_from_anchor > band_ticks as f64 {
                // Overextended to the upside: fade it.
                let sell_lots = lots
                    .min(self.sellable_lots(account_id))
                    .min((deviation + cap).max(0));
                if sell_lots > 0 {
                    let bid = self
                        .core
                        .exchange
                        .book()
                        .best_bid()
                        .unwrap_or((mid - 1).max(1));
                    let request = LimitOrderRequest {
                        account_id,
                        side: Side::Sell,
                        limit_price: bid,
                        quantity: sell_lots * LOT_SIZE,
                    };
                    if let Some(order_id) = self.core.submit_and_track(request, now_ms) {
                        self.mean_revs[index].resting.push(order_id);
                    }
                }
            } else if deviation_from_anchor < -band_ticks as f64 {
                let buy_lots = lots.min((cap - deviation).max(0));
                if buy_lots > 0 {
                    let ask = self.core.exchange.book().best_ask().unwrap_or(mid + 1);
                    let request = LimitOrderRequest {
                        account_id,
                        side: Side::Buy,
                        limit_price: ask,
                        quantity: buy_lots * LOT_SIZE,
                    };
                    if let Some(order_id) = self.core.submit_and_track(request, now_ms) {
                        self.mean_revs[index].resting.push(order_id);
                    }
                }
            }
        }

        let rate = self.mean_revs[index].wake_rate_per_second * self.activity_multiplier(now_ms);
        let gap = self.core.rng.poisson_gap_ms(rate);
        let next = now_ms.checked_add(gap).expect("sim time overflow");
        self.core.schedule(next, KIND_MEAN_REV, index);
    }

    // ------------------------------------------------------------------
    // Fundamentalists.
    // ------------------------------------------------------------------

    fn wake_fund(&mut self, index: usize, now_ms: SimTime) {
        let account_id = self.funds[index].account_id;
        self.funds[index]
            .resting
            .retain(|id| self.core.exchange.book().order(*id).is_some());
        for order_id in std::mem::take(&mut self.funds[index].resting) {
            self.core.cancel_tracked(account_id, order_id, now_ms);
        }

        // The private fair-value estimate drifts with the regime and
        // jitters with idiosyncratic news.
        let drift = self.config.regime.fund_drift_ticks[self.regime.index()];
        let vol = self.config.regime.fund_vol_ticks;
        self.funds[index].value_ticks += drift + self.core.rng.standard_normal() * vol;

        let (value, band_ticks, lots, cap) = {
            let agent = &self.funds[index];
            (
                agent.value_ticks,
                agent.band_ticks,
                agent.lots,
                agent.position_cap_lots,
            )
        };
        let mid = self.core.mark_price();
        let mispricing = value - mid as f64; // >0: market undervalued
        let deviation = self.agent_lots(account_id) - self.seed_lots();
        if mispricing > band_ticks as f64 {
            let buy_lots = lots.min((cap - deviation).max(0));
            if buy_lots > 0 {
                let ask = self.core.exchange.book().best_ask().unwrap_or(mid + 1);
                let request = LimitOrderRequest {
                    account_id,
                    side: Side::Buy,
                    limit_price: ask,
                    quantity: buy_lots * LOT_SIZE,
                };
                if let Some(order_id) = self.core.submit_and_track(request, now_ms) {
                    self.funds[index].resting.push(order_id);
                }
            }
        } else if mispricing < -band_ticks as f64 {
            let sell_lots = lots
                .min(self.sellable_lots(account_id))
                .min((deviation + cap).max(0));
            if sell_lots > 0 {
                let bid = self
                    .core
                    .exchange
                    .book()
                    .best_bid()
                    .unwrap_or((mid - 1).max(1));
                let request = LimitOrderRequest {
                    account_id,
                    side: Side::Sell,
                    limit_price: bid,
                    quantity: sell_lots * LOT_SIZE,
                };
                if let Some(order_id) = self.core.submit_and_track(request, now_ms) {
                    self.funds[index].resting.push(order_id);
                }
            }
        }

        let rate = self.funds[index].wake_rate_per_second * self.activity_multiplier(now_ms);
        let gap = self.core.rng.poisson_gap_ms(rate);
        let next = now_ms.checked_add(gap).expect("sim time overflow");
        self.core.schedule(next, KIND_FUND, index);
    }

    // ------------------------------------------------------------------
    // The scenario engine.
    // ------------------------------------------------------------------

    fn wake_regime(&mut self, now_ms: SimTime) {
        // Draw the next, different regime from the current row.
        let row = self.config.regime.switch_rows[self.regime.index()];
        let others = self.regime.others();
        let u = self.core.rng.next_f64();
        let mut next = others[2];
        let mut acc = 0.0;
        for (weight, candidate) in row.iter().zip(others) {
            acc += weight;
            if u < acc {
                next = candidate;
                break;
            }
        }
        self.regime = next;
        self.regime_history.push((now_ms, next));

        let switch_rate = 1000.0 / self.config.regime.mean_sojourn_ms as f64;
        let gap = self.core.rng.poisson_gap_ms(switch_rate);
        let next_time = now_ms.checked_add(gap).expect("sim time overflow");
        self.core.schedule(next_time, KIND_REGIME, 0);
    }

    // ------------------------------------------------------------------
    // Helpers.
    // ------------------------------------------------------------------

    /// Mean price over the last `fast_n` and `slow_n` prints; `None` until
    /// the tape is long enough for the slow window.
    fn moving_averages(&self, fast_n: usize, slow_n: usize) -> Option<(f64, f64)> {
        let tape = self.core.tape();
        if tape.len() < slow_n {
            return None;
        }
        let mean = |n: usize| -> f64 {
            tape[tape.len() - n..]
                .iter()
                .map(|print| print.price as f64)
                .sum::<f64>()
                / n as f64
        };
        Some((mean(fast_n), mean(slow_n)))
    }

    fn seed_lots(&self) -> i64 {
        self.config.agent_seed_shares / LOT_SIZE
    }

    /// Total shares held (settled + today's buys), in lots.
    fn agent_lots(&self, account_id: AccountId) -> i64 {
        self.core
            .exchange
            .account(account_id)
            .map(|account| {
                let position = account.position(&self.core.symbol);
                (position.settled + position.unsettled_buys) / LOT_SIZE
            })
            .unwrap_or(0)
    }

    /// Settled, sellable shares in lots.
    fn sellable_lots(&self, account_id: AccountId) -> i64 {
        self.core
            .exchange
            .account(account_id)
            .map(|account| account.position(&self.core.symbol).sellable / LOT_SIZE)
            .unwrap_or(0)
    }
}

impl MarketDriver for HeteroMarket {
    fn core(&mut self) -> &mut MarketCore {
        &mut self.core
    }

    fn wake(&mut self, kind: u8, index: usize, now_ms: SimTime) {
        match kind {
            KIND_NOISE => self.wake_noise(index, now_ms),
            KIND_MM => self.wake_mm(index, now_ms),
            KIND_TREND => self.wake_trend(index, now_ms),
            KIND_MEAN_REV => self.wake_mean_rev(index, now_ms),
            KIND_FUND => self.wake_fund(index, now_ms),
            KIND_REGIME => self.wake_regime(now_ms),
            other => unreachable!("unknown wake kind {other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats;
    use crate::{Account, AccountId, Exchange};

    fn small_config(seed: u64) -> HeteroMarketConfig {
        HeteroMarketConfig {
            n_noise: 24,
            n_market_makers: 3,
            n_trend: 4,
            n_mean_revert: 4,
            n_fundamental: 2,
            day_length_ms: 120_000,
            seed,
            ..HeteroMarketConfig::default()
        }
    }

    #[test]
    fn same_seed_reproduces_the_heterogeneous_market_exactly() {
        let mut a = HeteroMarket::new(small_config(7));
        let mut b = HeteroMarket::new(small_config(7));
        a.run_until(60_000);
        b.run_until(60_000);
        assert_eq!(a.tape(), b.tape());
        assert_eq!(a.exchange(), b.exchange());
        assert_eq!(a.replay_log(), b.replay_log());
        assert_eq!(a.regime_history(), b.regime_history());
    }

    #[test]
    fn different_seeds_produce_different_markets() {
        let mut a = HeteroMarket::new(small_config(1));
        let mut b = HeteroMarket::new(small_config(2));
        a.run_until(60_000);
        b.run_until(60_000);
        assert_ne!(a.tape(), b.tape());
    }

    #[test]
    fn replay_log_rebuilds_the_identical_exchange() {
        let mut market = HeteroMarket::new(small_config(9));
        market.run_until(240_000);

        let mut rebuilt = Exchange::new(market.config.symbol.clone());
        let n_agents = market.config.n_noise
            + market.config.n_market_makers
            + market.config.n_trend
            + market.config.n_mean_revert
            + market.config.n_fundamental;
        for agent in 0..n_agents {
            let mut account = Account::with_cash(market.config.agent_cash);
            account.seed_settled_position(&market.config.symbol, market.config.agent_seed_shares);
            rebuilt.add_account(agent as AccountId, account).unwrap();
        }
        let processed = rebuilt
            .replay(market.replay_log().to_vec())
            .expect("log keys are unique and ordered");
        assert_eq!(processed.len(), market.replay_log().len());
        assert_eq!(&rebuilt, market.exchange());
    }

    #[test]
    fn regime_engine_switches_regimes_over_a_long_run() {
        let mut market = HeteroMarket::new(HeteroMarketConfig {
            regime: RegimeParams {
                mean_sojourn_ms: 150_000,
                ..RegimeParams::default()
            },
            ..small_config(42)
        });
        // Twenty-four days at two-minute days: many sojourns elapse.
        market.run_until(2_880_000);
        let distinct = market
            .regime_history()
            .iter()
            .map(|(_, regime)| *regime)
            .collect::<std::collections::BTreeSet<_>>();
        eprintln!(
            "regime history: {:?}",
            market
                .regime_history()
                .iter()
                .map(|(t, r)| (t / 1000, format!("{r:?}")))
                .collect::<Vec<_>>()
        );
        assert!(
            distinct.len() >= 3,
            "the chain should visit at least three regimes, got {distinct:?}"
        );
        // The long run also gives even the slow fundamentalists time to
        // find mispricing and trade.
        let cfg = small_config(42);
        let first_fund =
            (cfg.n_noise + cfg.n_market_makers + cfg.n_trend + cfg.n_mean_revert) as AccountId;
        assert!(
            market
                .exchange()
                .trades()
                .iter()
                .any(|trade| trade.buyer_account_id >= first_fund
                    || trade.seller_account_id >= first_fund),
            "fundamentalists traded over a long run"
        );
    }

    #[test]
    fn smoke_run_trades_and_keeps_the_book_consistent() {
        let mut market = HeteroMarket::new(small_config(42));
        market.run_until(240_000);
        assert!(!market.tape().is_empty(), "the market should trade");
        assert_eq!(market.rejected_submits(), 0);
        if let (Some(bid), Some(ask)) = (
            market.exchange().book().best_bid(),
            market.exchange().book().best_ask(),
        ) {
            assert!(bid < ask, "book must never lock or cross");
        }
        for print in market.tape() {
            assert!(print.price > 0 && print.quantity > 0);
        }
        let bars = market.bars(1_000);
        assert!(!bars.is_empty());
        for bar in &bars {
            assert!(bar.high >= bar.low);
            assert!(bar.high >= bar.open && bar.high >= bar.close);
            assert!(bar.low <= bar.open && bar.low <= bar.close);
        }
        // Every strategy population actually participated in the tape.
        let cfg = small_config(42);
        let first_mm = cfg.n_noise as AccountId;
        let first_trend = first_mm + cfg.n_market_makers as AccountId;
        let first_mean_rev = first_trend + cfg.n_trend as AccountId;
        let first_fund = first_mean_rev + cfg.n_mean_revert as AccountId;
        let trades = market.exchange().trades();
        let involved = |lo: AccountId, hi: AccountId| {
            trades.iter().any(|trade| {
                (trade.buyer_account_id >= lo && trade.buyer_account_id < hi)
                    || (trade.seller_account_id >= lo && trade.seller_account_id < hi)
            })
        };
        assert!(involved(first_mm, first_trend), "market makers traded");
        assert!(involved(first_trend, first_mean_rev), "trend agents traded");
        assert!(
            involved(first_mean_rev, first_fund),
            "mean reverters traded"
        );
        assert!(
            involved(first_fund, AccountId::MAX),
            "fundamentalists traded"
        );

        // Sanity: |r| statistics are computable and prices stayed sane.
        let bars = market.bars(1_000);
        let closes: Vec<f64> = bars.iter().map(|bar| bar.close as f64).collect();
        let returns = stats::log_returns(&closes);
        assert!(stats::autocorrelation(&returns, 1).abs() < 0.5);
        let lo = bars.iter().map(|b| b.low).min().unwrap();
        let hi = bars.iter().map(|b| b.high).max().unwrap();
        assert!(
            (200..=3_000).contains(&lo) && (200..=3_000).contains(&hi),
            "price wandered too far: [{lo}, {hi}]"
        );
    }
}
