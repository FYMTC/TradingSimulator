//! Zero-intelligence market simulation on top of the M0 matching core.
//!
//! Each agent owns a Poisson clock: it sleeps for an exponentially
//! distributed gap, wakes up, either cancels one of its live orders or places
//! a new one anchored on the current mid, and goes back to sleep.  There is
//! no intelligence anywhere - prices emerge purely from the interaction of
//! random order flow with the book, which is exactly the claim of the Farmer
//! et al. zero-intelligence line of research that M1 sets out to verify.
//!
//! Everything runs on the shared deterministic [`crate::engine`] core, so a
//! seed reproduces the market exactly.  The run is also recorded as a
//! canonical [`Event`] log that replays to the identical exchange state.

use crate::bar::{Bar, TapePrint, aggregate_bars};
use crate::engine::{self, KIND_SETTLE, MarketCore, MarketDriver};
use crate::rng::Rng;
use crate::{
    AccountId, Exchange, LimitOrderRequest, Money, OrderId, Price, Quantity, Side, SimTime,
};

/// Shares per lot; quantities are placed in whole lots, A-share style.
pub const LOT_SIZE: Quantity = 100;

/// Queue kind of a noise-agent wake-up within the M1 market.
const KIND_NOISE: u8 = 1;

/// Behaviour parameters shared by every noise agent.
#[derive(Clone, Copy, Debug)]
pub struct NoiseAgentParams {
    /// Poisson wake-up intensity, in events per second per agent.
    pub wake_rate_per_second: f64,
    /// Probability of cancelling (instead of quoting) when the agent has at
    /// least one live order.
    pub cancel_probability: f64,
    /// Probability that a new order is aggressive (marketable) rather than a
    /// passive quote around the mid.
    pub aggressive_probability: f64,
    /// Median order size in lots (lognormal median).
    pub size_median_lots: f64,
    /// Lognormal sigma of order size; higher means fatter volume tails.
    pub size_sigma: f64,
    /// Maximum extra ticks an aggressive order crosses beyond the touch.
    pub aggressive_overshoot_max_ticks: i64,
    /// Maximum ticks a passive quote may step back from the same-side
    /// touch; quotes always range from that far behind up to two ticks
    /// inside the spread.
    pub passive_max_quote_offset_ticks: i64,
    /// Hard cap on noise order size in lots.  The lognormal draw is
    /// unbounded; without a cap a monster order eventually exceeds the
    /// seller's remaining position and trips the risk check.
    pub max_order_lots: i64,
}

impl Default for NoiseAgentParams {
    fn default() -> Self {
        // Calibrated against the M1 stylized-facts acceptance harness
        // (tests/stylized_facts.rs): heavy-tailed sizes plus a thin,
        // fast-cancelling book are what make the mid actually diffuse
        // instead of bouncing between the touches.
        Self {
            wake_rate_per_second: 1.0,
            cancel_probability: 0.45,
            aggressive_probability: 0.30,
            size_median_lots: 1.0,
            size_sigma: 2.4,
            aggressive_overshoot_max_ticks: 5,
            passive_max_quote_offset_ticks: 1,
            max_order_lots: 500,
        }
    }
}

/// Full configuration of one zero-intelligence market run.
#[derive(Clone, Debug)]
pub struct NoiseMarketConfig {
    pub symbol: String,
    /// Reference price in ticks used before the first trade forms a mid.
    pub ref_price: Price,
    pub n_agents: usize,
    pub agent_cash: Money,
    /// Settled, sellable shares seeded into every agent (keeps T+1 from
    /// silencing the sell side on day one).
    pub agent_seed_shares: Quantity,
    /// Simulated milliseconds per trading day; T+1 settles on each boundary.
    pub day_length_ms: SimTime,
    pub params: NoiseAgentParams,
    pub seed: u64,
}

impl Default for NoiseMarketConfig {
    fn default() -> Self {
        Self {
            symbol: "600000.SH".to_owned(),
            ref_price: 1_000,
            n_agents: 32,
            agent_cash: 1_000_000_000_000,
            agent_seed_shares: 1_000_000,
            day_length_ms: 4 * 60 * 60 * 1000,
            params: NoiseAgentParams::default(),
            seed: 1,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct AgentState {
    resting: Vec<OrderId>,
}

/// The market view a noise agent prices against.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QuoteView {
    pub best_bid: Option<Price>,
    pub best_ask: Option<Price>,
    pub last_trade: Option<Price>,
    pub ref_price: Price,
}

/// Touch-anchored pricing in the spirit of Farmer et al.: aggressive orders
/// cross the touch by a random amount, passive orders improve, join, or
/// step back from the same-side best quote.  Shared by every market layer
/// that keeps a zero-intelligence crowd for background liquidity.
pub(crate) fn noise_order_price(
    view: QuoteView,
    rng: &mut Rng,
    params: &NoiseAgentParams,
    side: Side,
    aggressive: bool,
) -> Price {
    let QuoteView {
        best_bid,
        best_ask,
        last_trade,
        ref_price,
    } = view;
    let (same_best, opposite_best) = match side {
        Side::Buy => (best_bid, best_ask),
        Side::Sell => (best_ask, best_bid),
    };

    if aggressive && let Some(opposite) = opposite_best {
        let overshoot = rng.uniform_int(0, params.aggressive_overshoot_max_ticks);
        return match side {
            Side::Buy => opposite + overshoot,
            Side::Sell => (opposite - overshoot).max(1),
        };
    }

    // Passive: the offset shifts the quote by -behind..+2 ticks from the
    // same-side best; negative steps behind the touch, positive improves
    // inside the spread (never crossing the opposite touch).
    if let Some(best) = same_best {
        let offset = rng.uniform_int(-params.passive_max_quote_offset_ticks, 2);
        let raw = match side {
            Side::Buy => best + offset,
            Side::Sell => best - offset,
        };
        // A passive order never crosses the opposite touch.
        return match (side, opposite_best) {
            (Side::Buy, Some(ask)) => raw.min(ask - 1).max(1),
            (Side::Sell, Some(bid)) => raw.max(bid + 1),
            _ => raw.max(1),
        };
    }

    // Own side is empty: quote near the last trade (or the initial
    // reference price on a cold start).
    let anchor = last_trade.unwrap_or(ref_price);
    let offset = rng.uniform_int(0, 2);
    match side {
        Side::Buy => (anchor - offset).max(1),
        Side::Sell => anchor + offset,
    }
}

/// Lognormal lot size, floored at one lot and capped at the configured
/// maximum.
pub(crate) fn noise_order_quantity(rng: &mut Rng, params: &NoiseAgentParams) -> Quantity {
    let z = rng.standard_normal();
    let lots = (z * params.size_sigma).exp() * params.size_median_lots;
    (lots.max(1.0).round() as i64).clamp(1, params.max_order_lots) as Quantity * LOT_SIZE
}

/// A zero-intelligence market driven by Poisson-woken noise agents.
#[derive(Clone, Debug)]
pub struct NoiseMarket {
    config: NoiseMarketConfig,
    core: MarketCore,
    agents: Vec<AgentState>,
}

impl NoiseMarket {
    /// Builds the market, funds the accounts, and schedules every agent's
    /// first wake-up plus the first T+1 settlement boundary.
    pub fn new(config: NoiseMarketConfig) -> Self {
        assert!(config.n_agents >= 2, "need at least two agents");
        let n_agents = config.n_agents;
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

        let mut market = Self {
            config,
            core,
            agents: vec![AgentState::default(); n_agents],
        };
        for agent in 0..market.config.n_agents {
            let gap = market
                .core
                .rng
                .poisson_gap_ms(market.config.params.wake_rate_per_second);
            market.core.schedule(gap, KIND_NOISE, agent);
        }
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
    pub fn replay_log(&self) -> &[crate::Event] {
        self.core.replay_log()
    }

    /// Processes every event scheduled at or before `target_ms`.
    pub fn run_until(&mut self, target_ms: SimTime) {
        engine::run_until(self, target_ms);
    }

    fn wake_agent(&mut self, agent: usize, now_ms: SimTime) {
        // Drop ids that have since filled or been cancelled.
        let mut live = std::mem::take(&mut self.agents[agent].resting);
        live.retain(|id| self.core.exchange.book().order(*id).is_some());

        let should_cancel = !live.is_empty()
            && self
                .core
                .rng
                .bernoulli(self.config.params.cancel_probability);
        if should_cancel {
            let index = self.core.rng.uniform_int(0, live.len() as i64 - 1) as usize;
            let order_id = live.swap_remove(index);
            self.agents[agent].resting = live;
            self.core
                .cancel_tracked(agent as AccountId, order_id, now_ms);
        } else {
            if let Some(order_id) = self.place_order(agent, now_ms) {
                live.push(order_id);
            }
            self.agents[agent].resting = live;
        }

        let gap = self
            .core
            .rng
            .poisson_gap_ms(self.config.params.wake_rate_per_second);
        let next = now_ms.checked_add(gap).expect("sim time overflow");
        self.core.schedule(next, KIND_NOISE, agent);
    }

    /// Places one order; returns the order id when any quantity rests.
    fn place_order(&mut self, agent: usize, now_ms: SimTime) -> Option<OrderId> {
        let side = if self.core.rng.bernoulli(0.5) {
            Side::Buy
        } else {
            Side::Sell
        };
        let aggressive = self
            .core
            .rng
            .bernoulli(self.config.params.aggressive_probability);
        let price = self.draw_price(side, aggressive);
        if price <= 0 {
            return None;
        }
        let quantity = self.draw_quantity();

        let request = LimitOrderRequest {
            account_id: agent as AccountId,
            side,
            limit_price: price,
            quantity,
        };
        self.core.submit_and_track(request, now_ms)
    }

    /// Touch-anchored pricing; see [`noise_order_price`].
    fn draw_price(&mut self, side: Side, aggressive: bool) -> Price {
        let view = QuoteView {
            best_bid: self.core.exchange.book().best_bid(),
            best_ask: self.core.exchange.book().best_ask(),
            last_trade: self.core.last_trade_price(),
            ref_price: self.config.ref_price,
        };
        noise_order_price(
            view,
            &mut self.core.rng,
            &self.config.params,
            side,
            aggressive,
        )
    }

    /// Lognormal lot size; see [`noise_order_quantity`].
    fn draw_quantity(&mut self) -> Quantity {
        noise_order_quantity(&mut self.core.rng, &self.config.params)
    }
}

impl MarketDriver for NoiseMarket {
    fn core(&mut self) -> &mut MarketCore {
        &mut self.core
    }

    fn wake(&mut self, kind: u8, index: usize, now_ms: SimTime) {
        debug_assert_eq!(kind, KIND_NOISE, "the M1 market only has noise agents");
        self.wake_agent(index, now_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Account, AccountId, Exchange};

    fn small_config(seed: u64) -> NoiseMarketConfig {
        NoiseMarketConfig {
            n_agents: 16,
            seed,
            day_length_ms: 60_000,
            ..NoiseMarketConfig::default()
        }
    }

    #[test]
    fn same_seed_reproduces_the_market_exactly() {
        let mut a = NoiseMarket::new(small_config(7));
        let mut b = NoiseMarket::new(small_config(7));
        a.run_until(30_000);
        b.run_until(30_000);
        assert_eq!(a.tape(), b.tape());
        assert_eq!(a.exchange(), b.exchange());
        assert_eq!(a.replay_log(), b.replay_log());
    }

    #[test]
    fn different_seeds_produce_different_markets() {
        let mut a = NoiseMarket::new(small_config(1));
        let mut b = NoiseMarket::new(small_config(2));
        a.run_until(30_000);
        b.run_until(30_000);
        assert_ne!(a.tape(), b.tape());
    }

    #[test]
    fn replay_log_rebuilds_the_identical_exchange() {
        let mut market = NoiseMarket::new(small_config(9));
        market.run_until(120_000);

        let mut rebuilt = Exchange::new(market.config.symbol.clone());
        for agent in 0..market.config.n_agents {
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
    fn smoke_run_trades_and_keeps_the_book_consistent() {
        let mut market = NoiseMarket::new(small_config(42));
        market.run_until(60_000);
        assert!(!market.tape().is_empty(), "the market should trade");
        assert!(market.now_ms() >= 60_000);
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
    }
}
