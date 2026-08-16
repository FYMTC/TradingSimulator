//! Zero-intelligence market simulation on top of the M0 matching core.
//!
//! Each agent owns a Poisson clock: it sleeps for an exponentially
//! distributed gap, wakes up, either cancels one of its live orders or places
//! a new one anchored on the current mid, and goes back to sleep.  There is
//! no intelligence anywhere - prices emerge purely from the interaction of
//! random order flow with the book, which is exactly the claim of the Farmer
//! et al. zero-intelligence line of research that M1 sets out to verify.
//!
//! Everything runs on a single thread with one shared [`Rng`], so a seed
//! reproduces the market exactly.  The run is also recorded as a canonical
//! [`Event`] log that replays to the identical exchange state.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::bar::{aggregate_bars, Bar, TapePrint};
use crate::rng::Rng;
use crate::{
    Account, AccountId, Event, EventKind, EventKey, Exchange, LimitOrderRequest, Money, OrderId,
    Price, Quantity, Side, SimTime,
};

/// Shares per lot; quantities are placed in whole lots, A-share style.
pub const LOT_SIZE: Quantity = 100;

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
}

impl Default for NoiseAgentParams {
    fn default() -> Self {
        Self {
            wake_rate_per_second: 1.0,
            cancel_probability: 0.35,
            aggressive_probability: 0.30,
            size_median_lots: 1.0,
            size_sigma: 1.0,
            aggressive_overshoot_max_ticks: 2,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct QueueEntry {
    time_ms: SimTime,
    /// Global tie-breaker guaranteeing a total order over simultaneous
    /// events; this is what makes the run deterministic.
    seq: u64,
    /// Agent index, or `SETTLE_SENTINEL` for the T+1 day boundary.
    agent: usize,
}

const SETTLE_SENTINEL: usize = usize::MAX;

/// A zero-intelligence market driven by Poisson-woken noise agents.
#[derive(Clone, Debug)]
pub struct NoiseMarket {
    config: NoiseMarketConfig,
    exchange: Exchange,
    rng: Rng,
    queue: BinaryHeap<Reverse<QueueEntry>>,
    next_seq: u64,
    now_ms: SimTime,
    last_trade_price: Option<Price>,
    tape: Vec<TapePrint>,
    /// Post-trade spread samples in ticks, for the acceptance harness.
    spread_samples_ticks: Vec<i64>,
    agents: Vec<AgentState>,
    /// Canonical event log; replaying it rebuilds the identical exchange.
    replay_log: Vec<Event>,
    rejected_submits: usize,
}

impl NoiseMarket {
    /// Builds the market, funds the accounts, and schedules every agent's
    /// first wake-up plus the first T+1 settlement boundary.
    pub fn new(config: NoiseMarketConfig) -> Self {
        assert!(config.n_agents >= 2, "need at least two agents");
        let n_agents = config.n_agents;
        let initial_ref = config.ref_price;
        let mut exchange = Exchange::new(config.symbol.clone());
        for agent in 0..n_agents {
            let mut account = Account::with_cash(config.agent_cash);
            account.seed_settled_position(&config.symbol, config.agent_seed_shares);
            exchange
                .add_account(agent as AccountId, account)
                .expect("account ids are unique");
        }

        let rng = Rng::seed_from_u64(config.seed);
        let mut market = Self {
            config,
            exchange,
            rng,
            queue: BinaryHeap::new(),
            next_seq: 0,
            now_ms: 0,
            last_trade_price: None,
            tape: Vec::new(),
            spread_samples_ticks: Vec::new(),
            agents: vec![AgentState::default(); n_agents],
            replay_log: Vec::new(),
            rejected_submits: 0,
        };
        for agent in 0..market.config.n_agents {
            let gap = market.rng.poisson_gap_ms(market.config.params.wake_rate_per_second);
            market.schedule(gap, agent);
        }
        market.schedule(market.config.day_length_ms, SETTLE_SENTINEL);
        market
    }

    /// The underlying exchange.
    pub fn exchange(&self) -> &Exchange {
        &self.exchange
    }

    /// Current simulation time in milliseconds.
    pub fn now_ms(&self) -> SimTime {
        self.now_ms
    }

    /// The executed trade tape in order.
    pub fn tape(&self) -> &[TapePrint] {
        &self.tape
    }

    /// Post-trade spread samples (in ticks) collected during the run.
    pub fn spread_samples_ticks(&self) -> &[i64] {
        &self.spread_samples_ticks
    }

    /// Aggregates the tape into OHLCV bars of `width_ms`.
    pub fn bars(&self, width_ms: SimTime) -> Vec<Bar> {
        aggregate_bars(self.tape.iter().copied(), width_ms)
    }

    /// Submissions rejected by risk checks so far (diagnostics).
    pub fn rejected_submits(&self) -> usize {
        self.rejected_submits
    }

    /// The recorded event log, replayable through [`Exchange::replay`].
    pub fn replay_log(&self) -> &[Event] {
        &self.replay_log
    }

    /// Processes every event scheduled at or before `target_ms`.
    pub fn run_until(&mut self, target_ms: SimTime) {
        while let Some(Reverse(entry)) = self.queue.peek() {
            if entry.time_ms > target_ms {
                break;
            }
            let Reverse(entry) = self.queue.pop().expect("peeked entry exists");
            self.now_ms = self.now_ms.max(entry.time_ms);
            if entry.agent == SETTLE_SENTINEL {
                self.exchange.settle_trading_day();
                self.log_event(entry.time_ms, entry.seq, EventKind::SettleTradingDay);
                self.schedule(
                    entry
                        .time_ms
                        .checked_add(self.config.day_length_ms)
                        .expect("sim time overflow"),
                    SETTLE_SENTINEL,
                );
            } else {
                self.wake_agent(entry.agent, entry.time_ms, entry.seq);
            }
        }
        self.now_ms = self.now_ms.max(target_ms);
    }

    fn schedule(&mut self, time_ms: SimTime, agent: usize) {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.checked_add(1).expect("sequence overflow");
        self.queue.push(Reverse(QueueEntry { time_ms, seq, agent }));
    }

    fn log_event(&mut self, time_ms: SimTime, seq: u64, kind: EventKind) {
        self.replay_log.push(Event {
            key: EventKey {
                sim_time: time_ms,
                source_priority: 1,
                source_seq: seq,
            },
            kind,
        });
    }

    fn wake_agent(&mut self, agent: usize, now_ms: SimTime, seq: u64) {
        // Drop ids that have since filled or been cancelled.
        let mut live = std::mem::take(&mut self.agents[agent].resting);
        live.retain(|id| self.exchange.book().order(*id).is_some());

        let should_cancel = !live.is_empty()
            && self
                .rng
                .bernoulli(self.config.params.cancel_probability);
        if should_cancel {
            let index = self.rng.uniform_int(0, live.len() as i64 - 1) as usize;
            let order_id = live.swap_remove(index);
            self.agents[agent].resting = live;
            let _ = self.exchange.cancel_order(agent as AccountId, order_id);
            self.log_event(
                now_ms,
                seq,
                EventKind::Cancel {
                    account_id: agent as AccountId,
                    order_id,
                },
            );
        } else {
            if let Some(order_id) = self.place_order(agent, now_ms, seq) {
                live.push(order_id);
            }
            self.agents[agent].resting = live;
        }

        let gap = self
            .rng
            .poisson_gap_ms(self.config.params.wake_rate_per_second);
        self.schedule(
            now_ms.checked_add(gap).expect("sim time overflow"),
            agent,
        );
    }

    /// Places one order; returns the order id when any quantity rests.
    fn place_order(&mut self, agent: usize, now_ms: SimTime, seq: u64) -> Option<OrderId> {
        let side = if self.rng.bernoulli(0.5) {
            Side::Buy
        } else {
            Side::Sell
        };
        let aggressive = self
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
        self.log_event(now_ms, seq, EventKind::Submit(request));
        let Ok(result) = self.exchange.submit_limit_order(request) else {
            self.rejected_submits += 1;
            return None;
        };
        for trade in &result.trades {
            self.last_trade_price = Some(trade.price);
            self.tape.push(TapePrint::new(now_ms, trade.price, trade.quantity));
        }
        if !result.trades.is_empty() {
            if let (Some(bid), Some(ask)) =
                (self.exchange.book().best_bid(), self.exchange.book().best_ask())
            {
                self.spread_samples_ticks.push(ask - bid);
            }
        }
        (result.remaining > 0).then_some(result.order_id)
    }

    /// Touch-anchored pricing in the spirit of Farmer et al.: aggressive
    /// orders cross the touch by a small random amount, passive orders
    /// improve, join, or step back from the same-side best quote.  Prints
    /// therefore cluster on one or two price levels around the touch, the
    /// way they do in real limit order markets.
    fn draw_price(&mut self, side: Side, aggressive: bool) -> Price {
        let (same_best, opposite_best) = match side {
            Side::Buy => (
                self.exchange.book().best_bid(),
                self.exchange.book().best_ask(),
            ),
            Side::Sell => (
                self.exchange.book().best_ask(),
                self.exchange.book().best_bid(),
            ),
        };

        if aggressive {
            if let Some(opposite) = opposite_best {
                let overshoot = self
                    .rng
                    .uniform_int(0, self.config.params.aggressive_overshoot_max_ticks);
                return match side {
                    Side::Buy => opposite + overshoot,
                    Side::Sell => (opposite - overshoot).max(1),
                };
            }
        }

        // Passive: the offset shifts the quote by -1..+2 ticks from the
        // same-side best; negative steps behind the touch, positive
        // improves inside the spread (never crossing the opposite touch).
        if let Some(best) = same_best {
            let offset = self.rng.uniform_int(-1, 2);
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
        let anchor = self
            .last_trade_price
            .unwrap_or(self.config.ref_price);
        let offset = self.rng.uniform_int(0, 2);
        match side {
            Side::Buy => (anchor - offset).max(1),
            Side::Sell => anchor + offset,
        }
    }

    /// Lognormal lot size, floored at one lot.
    fn draw_quantity(&mut self) -> Quantity {
        let z = self.rng.standard_normal();
        let lots = (z * self.config.params.size_sigma).exp() * self.config.params.size_median_lots;
        (lots.max(1.0).round() as Quantity) * LOT_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            rebuilt
                .add_account(agent as AccountId, account)
                .unwrap();
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
