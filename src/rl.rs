//! Reinforcement-learning investors on top of the M0/M1 market machinery.
//!
//! M2 keeps the zero-intelligence crowd for background liquidity and adds a
//! population of Q-learning investors.  Each RL agent wakes on its own
//! Poisson clock, observes a small discretised view of the market (spread
//! width and own inventory relative to target), and picks one of five
//! actions: hold, aggressive buy/sell, or a passive buy/sell quote.  The
//! reward is the spread capture of the agent's own fills between
//! decisions - mark-to-market PnL with the drift of held inventory
//! stripped out, so exogenous price moves are never mistaken for skill -
//! minus an inventory-risk penalty, so spread capture and inventory
//! control are *learned* rather than scripted - the micro-level analogue
//! of real investors adapting to the market they face.
//!
//! Every agent also gets a deterministic personality drawn at construction
//! (wake-up rate, risk aversion, inventory target, order size), so the
//! population behaves heterogeneously like a real investor crowd.
//!
//! Learning stays inside the deterministic simulation contract: the shared
//! [`crate::engine`] core (one RNG, strictly ordered events, canonical
//! replay log) means a seed reproduces the entire market bit for bit,
//! including everything the agents learned along the way.

use crate::bar::{Bar, TapePrint, aggregate_bars};
use crate::engine::{self, KIND_SETTLE, MarketCore, MarketDriver};
use crate::sim::{NoiseAgentParams, QuoteView, noise_order_price, noise_order_quantity};
use crate::{
    AccountId, Event, Exchange, LimitOrderRequest, Money, OrderId, Price, Quantity, Side, SimTime,
};

/// Queue kind of a noise-agent wake-up within the M2 market.
const KIND_NOISE: u8 = 1;
/// Queue kind of an RL-agent decision within the M2 market.
const KIND_RL: u8 = 2;

/// Number of discrete market states: 3 spread-width x 3 inventory buckets.
/// Kept deliberately small so ~10^2-10^3 decisions per agent suffice to
/// learn a meaningful policy.
pub const N_STATES: usize = 9;
/// Hold, aggressive buy/sell, passive buy/sell quote.
pub const N_ACTIONS: usize = 5;

/// The action set of every RL investor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Do nothing this decision.
    Hold = 0,
    /// Cross the spread and buy immediately.
    AggressiveBuy = 1,
    /// Cross the spread and sell immediately.
    AggressiveSell = 2,
    /// Join the best bid with a resting limit buy.
    PassiveBuyQuote = 3,
    /// Join the best ask with a resting limit sell.
    PassiveSellQuote = 4,
}

/// Tabular Q-values over the discretised state/action grid.
#[derive(Clone, Debug)]
pub struct QTable {
    values: Vec<f64>,
}

impl QTable {
    pub fn zeros() -> Self {
        Self {
            values: vec![0.0; N_STATES * N_ACTIONS],
        }
    }

    pub fn value(&self, state: usize, action: usize) -> f64 {
        self.values[state * N_ACTIONS + action]
    }

    /// Greedy action; ties resolve to the lowest index, deterministically.
    pub fn best_action(&self, state: usize) -> usize {
        let mut best = 0;
        let mut best_value = f64::NEG_INFINITY;
        for action in 0..N_ACTIONS {
            let value = self.value(state, action);
            if value > best_value {
                best = action;
                best_value = value;
            }
        }
        best
    }

    pub fn max_value(&self, state: usize) -> f64 {
        self.value(state, self.best_action(state))
    }

    /// One step of Q-learning: `Q(s,a) += alpha * (target - Q(s,a))`.
    pub fn update(&mut self, state: usize, action: usize, target: f64, alpha: f64) {
        let index = state * N_ACTIONS + action;
        self.values[index] += alpha * (target - self.values[index]);
    }

    /// Largest absolute Q-value, a measure of how much has been learned.
    pub fn max_abs(&self) -> f64 {
        self.values.iter().fold(0.0, |acc, v| acc.max(v.abs()))
    }
}

/// Learning parameters shared by every RL agent.
#[derive(Clone, Copy, Debug)]
pub struct RlAgentParams {
    /// Q-learning step size.
    pub alpha: f64,
    /// Discount factor per decision.
    pub gamma: f64,
    /// Exploration rate at the first decision.
    pub epsilon_start: f64,
    /// Exploration floor.
    pub epsilon_min: f64,
    /// Multiplicative epsilon decay applied after each decision.
    pub epsilon_decay: f64,
    /// Rewards are clipped to +-this many per-share ticks before the TD
    /// update.  The mark-drift noise is already stripped from the reward
    /// (see [`Pending`]); the clip only guards the residual tails of
    /// fill-timing costs, which are genuine signal and therefore clipped
    /// generously.
    pub reward_clip: f64,
    /// Inventory-risk penalty per lot of deviation from target and per
    /// decision, in per-share tick units.
    pub inventory_penalty_per_lot: f64,
    /// Upper bound on each agent's order size in lots; actual sizes are
    /// drawn per agent from 1..=this at construction.
    pub max_order_lots: Quantity,
}

impl Default for RlAgentParams {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            gamma: 0.95,
            epsilon_start: 0.6,
            epsilon_min: 0.05,
            epsilon_decay: 0.99,
            reward_clip: 30.0,
            inventory_penalty_per_lot: 0.2,
            max_order_lots: 4,
        }
    }
}

/// The decision whose consequence is still open, awaiting its TD update.
#[derive(Clone, Copy, Debug)]
struct Pending {
    state: usize,
    action: usize,
    /// Mark-to-market PnL of own trading measured *before* the action
    /// executed, so the action's own spread cost lands in the reward.
    pnl: Money,
    /// Mark price and inventory (lots vs target) at the same moment, used
    /// to strip the pure mark drift on held inventory out of the reward.
    mark: Price,
    deviation_lots: i64,
}

/// One Q-learning investor with a deterministic personality.
#[derive(Clone, Debug)]
struct RlAgent {
    account_id: AccountId,
    q: QTable,
    epsilon: f64,
    /// Personal wake-up intensity, events per second.
    wake_rate_per_second: f64,
    /// Personal inventory-risk aversion.
    inventory_penalty: f64,
    /// Personal inventory target in lots.
    target_lots: i64,
    /// Personal order size cap in lots.
    max_lots: Quantity,
    /// Personal hard position limit in lots of deviation from target;
    /// real investors have risk limits, and without one a one-sided
    /// quoting policy accumulates inventory without bound.
    position_cap_lots: i64,
    resting: Vec<OrderId>,
    pending: Option<Pending>,
    decisions: u64,
    /// How often each action was chosen, for diagnostics.
    action_counts: [u64; N_ACTIONS],
    rewards: Vec<f64>,
}

/// Read-only view of one RL agent, for tests and diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct RlAgentStats {
    pub account_id: AccountId,
    pub decisions: u64,
    /// How often each action was chosen, indexed by [`Action`] order
    /// (hold, aggressive buy, aggressive sell, passive buy, passive sell).
    pub action_counts: [u64; N_ACTIONS],
    pub epsilon: f64,
    /// Mark-to-market PnL of the agent's own trading at the last trade
    /// price, relative to its initial endowment.
    pub pnl: Money,
    /// Total shares held, in lots.
    pub lots: i64,
    /// Reward per decision, in per-share tick units.
    pub rewards: Vec<f64>,
    /// Largest absolute Q-value of the agent's table.
    pub q_max_abs: f64,
}

/// Full configuration of one mixed noise + RL market run.
#[derive(Clone, Debug)]
pub struct RlMarketConfig {
    pub symbol: String,
    /// Reference price in ticks used before the first trade forms a mid.
    pub ref_price: Price,
    /// Simulated milliseconds per trading day; T+1 settles on each boundary.
    pub day_length_ms: SimTime,
    pub seed: u64,
    pub agent_cash: Money,
    /// Settled, sellable shares seeded into every agent.
    pub agent_seed_shares: Quantity,
    pub n_noise_agents: usize,
    pub n_rl_agents: usize,
    pub noise: NoiseAgentParams,
    /// Base RL wake-up intensity; each agent draws its own rate around it.
    pub rl_wake_rate_per_second: f64,
    pub rl: RlAgentParams,
}

impl Default for RlMarketConfig {
    fn default() -> Self {
        Self {
            symbol: "600000.SH".to_owned(),
            ref_price: 1_000,
            day_length_ms: 4 * 60 * 60 * 1000,
            seed: 1,
            agent_cash: 1_000_000_000_000,
            agent_seed_shares: 1_000_000,
            n_noise_agents: 64,
            n_rl_agents: 8,
            noise: NoiseAgentParams::default(),
            rl_wake_rate_per_second: 1.0,
            rl: RlAgentParams::default(),
        }
    }
}

/// A mixed market: zero-intelligence noise agents supply background
/// liquidity while Q-learning investors trade against them.
#[derive(Clone, Debug)]
pub struct RlMarket {
    config: RlMarketConfig,
    core: MarketCore,
    noise_resting: Vec<Vec<OrderId>>,
    rl_agents: Vec<RlAgent>,
}

impl RlMarket {
    /// Builds the market, funds the accounts, gives every RL agent its
    /// deterministic personality, and schedules all first wake-ups plus
    /// the first T+1 settlement boundary.
    pub fn new(config: RlMarketConfig) -> Self {
        assert!(config.n_noise_agents >= 2, "need at least two noise agents");
        assert!(config.n_rl_agents >= 1, "need at least one RL agent");
        let n_agents = config.n_noise_agents + config.n_rl_agents;
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

        let seed_lots = config.agent_seed_shares / crate::sim::LOT_SIZE;
        let mut rl_agents = Vec::with_capacity(config.n_rl_agents);
        for index in 0..config.n_rl_agents {
            // Deterministic investor personalities: patience, risk
            // aversion, target inventory, and order size all vary.
            let wake_rate = config.rl_wake_rate_per_second * (0.5 + core.rng.next_f64());
            let inventory_penalty =
                config.rl.inventory_penalty_per_lot * (0.5 + core.rng.next_f64());
            let max_lots = core.rng.uniform_int(1, config.rl.max_order_lots);
            // Inventory targets stay within a few orders of the endowment
            // so the position bucket reacts to real accumulation.
            let target_lots = seed_lots + core.rng.uniform_int(-5, 5);
            let position_cap_lots = 5 * max_lots;
            rl_agents.push(RlAgent {
                account_id: (config.n_noise_agents + index) as AccountId,
                q: QTable::zeros(),
                epsilon: config.rl.epsilon_start,
                wake_rate_per_second: wake_rate,
                inventory_penalty,
                target_lots,
                max_lots,
                position_cap_lots,
                resting: Vec::new(),
                pending: None,
                decisions: 0,
                action_counts: [0; N_ACTIONS],
                rewards: Vec::new(),
            });
        }

        let mut market = Self {
            config,
            core,
            noise_resting: vec![Vec::new(); n_agents],
            rl_agents,
        };
        for agent in 0..market.config.n_noise_agents {
            let gap = market
                .core
                .rng
                .poisson_gap_ms(market.config.noise.wake_rate_per_second);
            market.core.schedule(gap, KIND_NOISE, agent);
        }
        let rl_gaps: Vec<SimTime> = market
            .rl_agents
            .iter()
            .map(|agent| market.core.rng.poisson_gap_ms(agent.wake_rate_per_second))
            .collect();
        for (index, gap) in rl_gaps.into_iter().enumerate() {
            market.core.schedule(gap, KIND_RL, index);
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
    pub fn replay_log(&self) -> &[Event] {
        self.core.replay_log()
    }

    /// Snapshot of every RL agent's learning state and wealth.
    pub fn rl_stats(&self) -> Vec<RlAgentStats> {
        let mark = self.core.mark_price();
        self.rl_agents
            .iter()
            .map(|agent| RlAgentStats {
                account_id: agent.account_id,
                decisions: agent.decisions,
                action_counts: agent.action_counts,
                epsilon: agent.epsilon,
                pnl: self.agent_pnl(agent.account_id, mark),
                lots: self.agent_lots(agent.account_id),
                rewards: agent.rewards.clone(),
                q_max_abs: agent.q.max_abs(),
            })
            .collect()
    }

    /// Processes every event scheduled at or before `target_ms`.
    pub fn run_until(&mut self, target_ms: SimTime) {
        engine::run_until(self, target_ms);
    }

    // ------------------------------------------------------------------
    // Noise agents: identical behaviour to the M1 market.
    // ------------------------------------------------------------------

    fn wake_noise(&mut self, agent: usize, now_ms: SimTime) {
        // Drop ids that have since filled or been cancelled.
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

        let gap = self
            .core
            .rng
            .poisson_gap_ms(self.config.noise.wake_rate_per_second);
        let next = now_ms.checked_add(gap).expect("sim time overflow");
        self.core.schedule(next, KIND_NOISE, agent);
    }

    /// Places one noise order; returns the order id when any quantity rests.
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
    // RL agents: observe, learn, act.
    // ------------------------------------------------------------------

    fn wake_rl_agent(&mut self, index: usize, now_ms: SimTime) {
        // 1. Observe the market and own account.
        let state = self.observe(index);
        let mark = self.core.mark_price();
        let account_id = self.rl_agents[index].account_id;
        let pnl = self.agent_pnl(account_id, mark);
        let lots = self.agent_lots(account_id);

        // 2. Close the previous decision's TD update.  The reward is the
        //    spread capture of own fills (PnL change minus the pure mark
        //    drift on the inventory held at the previous decision) minus
        //    an inventory-risk penalty.  Stripping the drift matters: the
        //    mid jumps fat-tailed distances, and unfiltered drift on
        //    inventory is unlearnable noise that would drown the signal.
        if let Some(pending) = self.rl_agents[index].pending.take() {
            let total = money_to_reward(pnl - pending.pnl);
            let drift = pending.deviation_lots as f64 * (mark - pending.mark) as f64;
            let spread_capture = total - drift;
            let penalty = self.rl_agents[index].inventory_penalty
                * (lots - self.rl_agents[index].target_lots).abs() as f64;
            let reward = (spread_capture - penalty)
                .clamp(-self.config.rl.reward_clip, self.config.rl.reward_clip);
            let next_max = self.rl_agents[index].q.max_value(state);
            let target = reward + self.config.rl.gamma * next_max;
            let alpha = self.config.rl.alpha;
            self.rl_agents[index]
                .q
                .update(pending.state, pending.action, target, alpha);
            self.rl_agents[index].rewards.push(reward);
        }

        // 3. Act: epsilon-greedy on the current state, then start from a
        //    clean slate so the new action is the only live intention.
        let action = self.select_rl_action(index, state);
        self.rl_agents[index].decisions += 1;
        self.rl_agents[index].action_counts[action] += 1;
        self.rl_agents[index].epsilon = (self.rl_agents[index].epsilon
            * self.config.rl.epsilon_decay)
            .max(self.config.rl.epsilon_min);
        self.cancel_rl_resting(index, now_ms);
        self.execute_rl_action(index, action, now_ms);
        let target_lots = self.rl_agents[index].target_lots;
        self.rl_agents[index].pending = Some(Pending {
            state,
            action,
            pnl,
            mark,
            deviation_lots: lots - target_lots,
        });

        // 4. Sleep until the next decision.
        let rate = self.rl_agents[index].wake_rate_per_second;
        let gap = self.core.rng.poisson_gap_ms(rate);
        let next = now_ms.checked_add(gap).expect("sim time overflow");
        self.core.schedule(next, KIND_RL, index);
    }

    /// Discretises the market view plus own inventory into a state index:
    /// spread width x own inventory bucket.  The two features that carry
    /// the learnable structure of this market - quoting earns more when
    /// the spread is wide, and inventory wants managing around the target.
    fn observe(&self, index: usize) -> usize {
        let bid = self.core.exchange.book().best_bid();
        let ask = self.core.exchange.book().best_ask();

        // Spread width bucket in ticks.
        let spread_bucket: usize = match (bid, ask) {
            (Some(bid), Some(ask)) => match ask - bid {
                1 => 0,
                2 | 3 => 1,
                _ => 2,
            },
            _ => 2,
        };

        // Own inventory relative to the personal target.
        let agent = &self.rl_agents[index];
        let lots = self.agent_lots(agent.account_id);
        let tolerance = agent.max_lots;
        let deviation = lots - agent.target_lots;
        let position_bucket: i64 = if deviation < -tolerance {
            -1
        } else if deviation > tolerance {
            1
        } else {
            0
        };

        spread_bucket * 3 + (position_bucket + 1) as usize
    }

    fn select_rl_action(&mut self, index: usize, state: usize) -> usize {
        let agent = &self.rl_agents[index];
        if self.core.rng.next_f64() < agent.epsilon {
            self.core.rng.uniform_int(0, N_ACTIONS as i64 - 1) as usize
        } else {
            agent.q.best_action(state)
        }
    }

    /// Cancels every live resting order of the agent.
    fn cancel_rl_resting(&mut self, index: usize, now_ms: SimTime) {
        let account_id = self.rl_agents[index].account_id;
        let mut resting = std::mem::take(&mut self.rl_agents[index].resting);
        resting.retain(|id| self.core.exchange.book().order(*id).is_some());
        for order_id in resting.drain(..) {
            self.core.cancel_tracked(account_id, order_id, now_ms);
        }
    }

    /// Translates the chosen action into at most one limit order, clamped
    /// by the agent's hard position limit (order sizes shrink to fit, and
    /// vanish entirely beyond it).
    fn execute_rl_action(&mut self, index: usize, action: usize, now_ms: SimTime) {
        if action == Action::Hold as usize {
            return;
        }
        let (account_id, max_lots, target_lots, position_cap_lots) = {
            let agent = &self.rl_agents[index];
            (
                agent.account_id,
                agent.max_lots,
                agent.target_lots,
                agent.position_cap_lots,
            )
        };
        let mark = self
            .core
            .last_trade_price()
            .unwrap_or(self.config.ref_price);
        let bid = self.core.exchange.book().best_bid();
        let ask = self.core.exchange.book().best_ask();
        let sellable_lots = self
            .core
            .exchange
            .account(account_id)
            .map(|account| account.position(&self.config.symbol).sellable / crate::sim::LOT_SIZE)
            .unwrap_or(0);
        let deviation = self.agent_lots(account_id) - target_lots;

        let (side, price, lots) = if action == Action::AggressiveBuy as usize {
            // Room left before the long-side position limit.
            let lots = max_lots.min((position_cap_lots - deviation).max(0) as Quantity);
            if lots == 0 {
                return;
            }
            (Side::Buy, ask.unwrap_or(mark + 1), lots)
        } else if action == Action::AggressiveSell as usize {
            let lots = max_lots
                .min(sellable_lots)
                .min((deviation + position_cap_lots).max(0) as Quantity);
            if lots == 0 {
                return;
            }
            (Side::Sell, bid.unwrap_or((mark - 1).max(1)), lots)
        } else if action == Action::PassiveBuyQuote as usize {
            let lots = max_lots.min((position_cap_lots - deviation).max(0) as Quantity);
            if lots == 0 {
                return;
            }
            // Join the bid; never cross the ask.
            let mut price = bid.unwrap_or((mark - 1).max(1));
            if let Some(ask) = ask {
                price = price.min(ask - 1);
            }
            (Side::Buy, price.max(1), lots)
        } else if action == Action::PassiveSellQuote as usize {
            let lots = max_lots
                .min(sellable_lots)
                .min((deviation + position_cap_lots).max(0) as Quantity);
            if lots == 0 {
                return;
            }
            // Join the ask; never cross the bid.
            let mut price = ask.unwrap_or(mark + 1);
            if let Some(bid) = bid {
                price = price.max(bid + 1);
            }
            (Side::Sell, price, lots)
        } else {
            return;
        };
        if price <= 0 || lots <= 0 {
            return;
        }

        let request = LimitOrderRequest {
            account_id,
            side,
            limit_price: price,
            quantity: lots * crate::sim::LOT_SIZE,
        };
        if let Some(order_id) = self.core.submit_and_track(request, now_ms) {
            self.rl_agents[index].resting.push(order_id);
        }
    }

    // ------------------------------------------------------------------
    // Account helpers.
    // ------------------------------------------------------------------

    /// Total shares held (settled + today's buys), in lots.
    fn agent_lots(&self, account_id: AccountId) -> i64 {
        self.core
            .exchange
            .account(account_id)
            .map(|account| {
                let position = account.position(&self.config.symbol);
                (position.settled + position.unsettled_buys) / crate::sim::LOT_SIZE
            })
            .unwrap_or(0)
    }

    /// Mark-to-market PnL of the agent's *own trading* relative to its
    /// initial endowment: change in cash (free + reserved) plus change in
    /// shares held, valued at the mark price.  Seeding the reward this way
    /// keeps the exogenous drift of the endowed position out of the
    /// learning signal - agents are credited only with the consequences of
    /// their actions.  Reservations count at face value, so a resting buy
    /// above the mark slightly overstates PnL, bounded by the spread: the
    /// exact cost structure the agents should learn to avoid.
    fn agent_pnl(&self, account_id: AccountId, mark: Price) -> Money {
        match self.core.exchange.account(account_id) {
            Some(account) => {
                let position = account.position(&self.config.symbol);
                let shares =
                    position.settled + position.unsettled_buys - self.config.agent_seed_shares;
                account.cash_available + account.cash_reserved - self.config.agent_cash
                    + Money::from(shares) * Money::from(mark)
            }
            None => 0,
        }
    }
}

impl MarketDriver for RlMarket {
    fn core(&mut self) -> &mut MarketCore {
        &mut self.core
    }

    fn wake(&mut self, kind: u8, index: usize, now_ms: SimTime) {
        match kind {
            KIND_NOISE => self.wake_noise(index, now_ms),
            KIND_RL => self.wake_rl_agent(index, now_ms),
            other => unreachable!("unknown wake kind {other}"),
        }
    }
}

/// Converts a money delta into per-share tick units so Q-values keep a
/// human scale regardless of lot sizes.
fn money_to_reward(delta: Money) -> f64 {
    delta as f64 / crate::sim::LOT_SIZE as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats;
    use crate::{Account, AccountId, Exchange};

    fn small_config(seed: u64) -> RlMarketConfig {
        RlMarketConfig {
            n_noise_agents: 24,
            n_rl_agents: 6,
            day_length_ms: 60_000,
            seed,
            ..RlMarketConfig::default()
        }
    }

    #[test]
    fn same_seed_reproduces_the_learning_market_exactly() {
        let mut a = RlMarket::new(small_config(7));
        let mut b = RlMarket::new(small_config(7));
        a.run_until(30_000);
        b.run_until(30_000);
        assert_eq!(a.tape(), b.tape());
        assert_eq!(a.exchange(), b.exchange());
        assert_eq!(a.replay_log(), b.replay_log());
        assert_eq!(a.rl_stats(), b.rl_stats());
    }

    #[test]
    fn different_seeds_produce_different_markets() {
        let mut a = RlMarket::new(small_config(1));
        let mut b = RlMarket::new(small_config(2));
        a.run_until(30_000);
        b.run_until(30_000);
        assert_ne!(a.tape(), b.tape());
    }

    #[test]
    fn replay_log_rebuilds_the_identical_exchange() {
        let mut market = RlMarket::new(small_config(9));
        market.run_until(120_000);

        let mut rebuilt = Exchange::new(market.config.symbol.clone());
        for agent in 0..(market.config.n_noise_agents + market.config.n_rl_agents) {
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
    fn agents_learn_to_time_their_trading() {
        let mut market = RlMarket::new(RlMarketConfig {
            n_noise_agents: 24,
            n_rl_agents: 6,
            seed: 2026,
            ..RlMarketConfig::default()
        });
        market.run_until(1_800_000);
        let stats = market.rl_stats();

        assert!(stats.iter().all(|s| s.decisions >= 100));
        // Inventory stays near target: the position cap plus the learned
        // policy keep accumulation bounded.
        assert!(
            stats.iter().all(|s| (s.lots - 10_000).abs() <= 25),
            "inventory deviations should stay small, got {:?}",
            stats.iter().map(|s| s.lots - 10_000).collect::<Vec<_>>()
        );
        // Exploration has annealed well below its starting level.
        assert!(
            stats.iter().all(|s| s.epsilon < 0.4),
            "epsilon should decay, got {:?}",
            stats.iter().map(|s| s.epsilon).collect::<Vec<_>>()
        );
        // Q-values actually moved away from the zero initialisation.
        assert!(
            stats.iter().all(|s| s.q_max_abs > 0.01),
            "Q tables should be non-trivial, got {:?}",
            stats.iter().map(|s| s.q_max_abs).collect::<Vec<_>>()
        );

        // The money test: average reward in the second half of training
        // beats the first half, where random exploration bleeds spread.
        let mut early = Vec::new();
        let mut late = Vec::new();
        for s in &stats {
            let half = s.rewards.len() / 2;
            if half == 0 {
                continue;
            }
            early.extend_from_slice(&s.rewards[..half]);
            late.extend_from_slice(&s.rewards[half..]);
        }
        let mean_early = stats::mean(&early);
        let mean_late = stats::mean(&late);
        eprintln!(
            "decisions: {:?}, mean reward early/late: {mean_early:.3}/{mean_late:.3}",
            stats.iter().map(|s| s.decisions).collect::<Vec<_>>()
        );
        assert!(
            mean_late > mean_early,
            "learning should improve average reward: early {mean_early:.3} vs late {mean_late:.3}"
        );
    }

    #[test]
    fn smoke_run_keeps_the_book_consistent() {
        let mut market = RlMarket::new(small_config(42));
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
        // RL agents actually participated in the tape.
        let first_rl = small_config(42).n_noise_agents as AccountId;
        assert!(market.exchange().trades().iter().any(|trade| {
            trade.buyer_account_id >= first_rl || trade.seller_account_id >= first_rl
        }));
    }
}
