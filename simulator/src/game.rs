//! M4 game layer: the player's trading loop on top of the M3 market.
//!
//! [`GameMarket`] wraps the heterogeneous population and adds everything a
//! game session needs on the back end:
//!
//! - **A player account** with [`GameMarket::submit_player_order`],
//!   [`GameMarket::cancel_player_order`] and a full
//!   [`GameMarket::snapshot`] view - the player trades against the very
//!   same agent crowd, order book and tape the simulations use, with
//!   fills reported immediately.
//! - **A-share price limits**: the daily ±`limit_bp` band gates every
//!   order (agents and player alike) and re-centres on each session's
//!   close at the T+1 settlement boundary.
//! - **A manipulator** - the classic "pump and dump" stock operator: it
//!   accumulates quietly at the bid, pumps aggressively through the
//!   offers, then distributes its inventory back into the crowd.  Its
//!   wake-ups run on the same strictly ordered event queue, so the whole
//!   session - player orders, manipulation, agent crowd - stays
//!   deterministic and replayable from one seed.
//! - **A virtual clock** ([`GameMarket::advance_ms`]) the future front
//!   end can drive frame by frame or fast-forward.

use crate::engine::{self, MarketCore, MarketDriver};
use crate::hetero::{HeteroMarket, HeteroMarketConfig, Regime};
use crate::sim::LOT_SIZE;
use crate::{
    AccountId, CancelResult, Event, Exchange, ExchangeError, LevelSnapshot, LimitOrderRequest,
    Money, OrderId, Price, Quantity, RestingOrder, Side, SimTime, SubmitResult,
};

/// Queue kind of the manipulator's wake-up; kinds 1-6 belong to the
/// underlying heterogeneous market.
const KIND_MANIP: u8 = 7;

/// Phase of the manipulator's pump-and-dump cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManipPhase {
    /// Quietly buying at/inside the bid, building the inventory.
    Accumulate,
    /// Sweeping the offers to drive the price up.
    Pump,
    /// Selling the inventory back into the crowd.
    Distribute,
    /// Cycle finished; stays dormant.
    Done,
}

/// Schedule of one pump-and-dump cycle.
#[derive(Clone, Debug)]
pub struct ManipParams {
    /// Duration of each phase, in simulated milliseconds.
    pub accumulate_ms: SimTime,
    pub pump_ms: SimTime,
    pub distribute_ms: SimTime,
    /// Wake-up intensity, events per second, shared by all phases.
    pub wake_rate_per_second: f64,
    /// Lots bought per wake while accumulating (passive, at the bid).
    pub accumulate_lots: i64,
    /// Lots swept per wake while pumping (aggressive, through the offers).
    pub pump_lots: i64,
    /// Lots sold per wake while distributing (into the bid).
    pub distribute_lots: i64,
}

impl Default for ManipParams {
    fn default() -> Self {
        Self {
            accumulate_ms: 400_000,
            pump_ms: 150_000,
            distribute_ms: 300_000,
            wake_rate_per_second: 1.5,
            accumulate_lots: 5,
            pump_lots: 10,
            distribute_lots: 15,
        }
    }
}

/// Full configuration of one game session.
#[derive(Clone, Debug)]
pub struct GameConfig {
    /// The underlying heterogeneous market (agent population, regimes).
    pub market: HeteroMarketConfig,
    pub player_cash: Money,
    /// Settled, sellable shares seeded into the player, so the sell side
    /// works from the first minute despite T+1.
    pub player_seed_shares: Quantity,
    pub manip_cash: Money,
    pub manip_seed_shares: Quantity,
    /// Daily price-limit band in basis points (main board 10% = 1000).
    pub limit_bp: i64,
    pub manip: ManipParams,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            market: HeteroMarketConfig::default(),
            player_cash: 10_000_000,
            player_seed_shares: 20_000,
            manip_cash: 1_000_000_000_000,
            manip_seed_shares: 1_000_000,
            limit_bp: 1_000,
            manip: ManipParams::default(),
        }
    }
}

/// The manipulator's live state.
#[derive(Clone, Debug)]
struct Manipulator {
    account_id: AccountId,
    phase: ManipPhase,
    phase_start_ms: SimTime,
    /// Mark price when the pump began, for diagnostics.
    pump_start_mark: Price,
    /// Highest mark seen during the pump.
    pump_peak_mark: Price,
    /// Total shares actually sold during distribution.
    distributed_shares: Quantity,
}

/// What the player sees: one consistent cut of the market and account.
#[derive(Clone, Debug)]
pub struct PlayerState {
    pub cash_available: Money,
    pub cash_reserved: Money,
    pub settled: Quantity,
    pub unsettled_buys: Quantity,
    pub sellable: Quantity,
    /// The player's resting limit orders.
    pub open_orders: Vec<RestingOrder>,
    /// Cash + all shares marked at the current mid.
    pub equity: Money,
}

#[derive(Clone, Debug)]
pub struct GameSnapshot {
    pub now_ms: SimTime,
    pub regime: Regime,
    pub best_bid: Option<Price>,
    pub best_ask: Option<Price>,
    /// Mid quote (or the available touch / last trade on a one-sided
    /// book); the mark used for equity.
    pub mark: Price,
    pub last_trade: Option<Price>,
    /// Daily price-limit band, `(lower, upper)`.
    pub price_limits: Option<(Price, Price)>,
    pub depth_bids: Vec<LevelSnapshot>,
    pub depth_asks: Vec<LevelSnapshot>,
    pub player: PlayerState,
    pub manip_phase: ManipPhase,
}

/// A playable market: heterogeneous agents + a manipulator + the player.
#[derive(Clone, Debug)]
pub struct GameMarket {
    config: GameConfig,
    inner: HeteroMarket,
    player_id: AccountId,
    manip: Manipulator,
    /// Resting player order ids, pruned lazily.
    player_resting: Vec<OrderId>,
}

impl GameMarket {
    /// Builds the session: agent market, player and manipulator
    /// accounts, the price-limit band, and the manipulator's first
    /// wake-up.
    pub fn new(config: GameConfig) -> Self {
        let agents = (config.market.n_noise
            + config.market.n_market_makers
            + config.market.n_trend
            + config.market.n_mean_revert
            + config.market.n_fundamental) as AccountId;
        let mut inner = HeteroMarket::new(config.market.clone());
        let player_id = agents;
        let manip_id = agents + 1;
        inner
            .core()
            .add_funded_account(player_id, config.player_cash, config.player_seed_shares);
        inner
            .core()
            .add_funded_account(manip_id, config.manip_cash, config.manip_seed_shares);
        inner
            .core()
            .exchange
            .set_price_limits(config.market.ref_price, config.limit_bp);

        // The manipulator opens in quiet accumulation, on its own clock.
        let first_gap = inner
            .core()
            .rng
            .poisson_gap_ms(config.manip.wake_rate_per_second);
        inner.core().schedule(first_gap, KIND_MANIP, 0);
        let start_mark = inner.core().mark_price();
        Self {
            manip: Manipulator {
                account_id: manip_id,
                phase: ManipPhase::Accumulate,
                phase_start_ms: 0,
                pump_start_mark: start_mark,
                pump_peak_mark: start_mark,
                distributed_shares: 0,
            },
            config,
            inner,
            player_id,
            player_resting: Vec::new(),
        }
    }

    // ------------------------------------------------------------------
    // Player API.
    // ------------------------------------------------------------------

    pub fn player_account_id(&self) -> AccountId {
        self.player_id
    }

    /// Submits a limit order for the player; `quantity_lots` is in whole
    /// lots (A-share board lots).  Fills are reported inline, exactly as
    /// for any other participant.
    pub fn submit_player_order(
        &mut self,
        side: Side,
        limit_price: Price,
        quantity_lots: i64,
    ) -> Result<SubmitResult, ExchangeError> {
        assert!(quantity_lots > 0, "quantity must be positive");
        let request = LimitOrderRequest {
            account_id: self.player_id,
            side,
            limit_price,
            quantity: quantity_lots * LOT_SIZE,
        };
        let now = self.inner.core().now_ms();
        let result = self.inner.core().submit_verbose(request, now)?;
        if result.remaining > 0 {
            self.player_resting.push(result.order_id);
        }
        Ok(result)
    }

    /// Cancels one resting player order.
    pub fn cancel_player_order(
        &mut self,
        order_id: OrderId,
    ) -> Result<CancelResult, ExchangeError> {
        let now = self.inner.core().now_ms();
        let result = self
            .inner
            .core()
            .exchange
            .cancel_order(self.player_id, order_id)?;
        self.inner.core().log_event(
            now,
            crate::EventKind::Cancel {
                account_id: self.player_id,
                order_id,
            },
        );
        self.player_resting.retain(|id| *id != order_id);
        Ok(result)
    }

    /// Advances the virtual clock by `ms`, processing every scheduled
    /// event (agent wake-ups, regime switches, settlements, the
    /// manipulator) in strict order.
    pub fn advance_ms(&mut self, ms: SimTime) {
        let target = self
            .inner
            .core()
            .now_ms()
            .checked_add(ms)
            .expect("sim time overflow");
        engine::run_until(self, target);
        // Prune filled/cancelled ids from the player's tracking list.
        let live: Vec<OrderId> = self
            .player_resting
            .iter()
            .filter(|id| self.inner.core().exchange.book().order(**id).is_some())
            .copied()
            .collect();
        self.player_resting = live;
    }

    /// One consistent view of the market and the player's account.
    pub fn snapshot(&self) -> GameSnapshot {
        let core = self.inner.core_view();
        let mut open_orders: Vec<RestingOrder> = self
            .player_resting
            .iter()
            .filter_map(|id| core.exchange.book().order(*id))
            .collect();
        open_orders.sort_by_key(|order| order.id);

        let account = core.exchange.account(self.player_id);
        let position = account
            .map(|account| account.position(&core.symbol))
            .unwrap_or_default();
        let shares = position.settled + position.unsettled_buys;
        let equity = account.map_or(0, |account| account.cash_available)
            + account.map_or(0, |account| account.cash_reserved)
            + Money::from(shares) * Money::from(core.mark_price());
        GameSnapshot {
            now_ms: core.now_ms(),
            regime: self.inner.regime(),
            best_bid: core.exchange.book().best_bid(),
            best_ask: core.exchange.book().best_ask(),
            mark: core.mark_price(),
            last_trade: core.last_trade_price(),
            price_limits: core
                .exchange
                .price_limits()
                .map(|limits| (limits.lower, limits.upper)),
            depth_bids: core.exchange.book().depth(Side::Buy, 10),
            depth_asks: core.exchange.book().depth(Side::Sell, 10),
            player: PlayerState {
                cash_available: account.map_or(0, |a| a.cash_available),
                cash_reserved: account.map_or(0, |a| a.cash_reserved),
                settled: position.settled,
                unsettled_buys: position.unsettled_buys,
                sellable: position.sellable,
                open_orders,
                equity,
            },
            manip_phase: self.manip.phase,
        }
    }

    // ------------------------------------------------------------------
    // Market views (thin forwards of the underlying heterogeneous run).
    // ------------------------------------------------------------------

    pub fn exchange(&self) -> &Exchange {
        self.inner.exchange()
    }

    pub fn now_ms(&self) -> SimTime {
        self.inner.now_ms()
    }

    pub fn tape(&self) -> &[crate::bar::TapePrint] {
        self.inner.tape()
    }

    /// Aggregates the tape into OHLCV bars of `width_ms`.
    pub fn bars(&self, width_ms: SimTime) -> Vec<crate::bar::Bar> {
        self.inner.bars(width_ms)
    }

    /// The canonical event log including player and manipulator orders.
    pub fn replay_log(&self) -> &[Event] {
        self.inner.replay_log()
    }

    pub fn manip_phase(&self) -> ManipPhase {
        self.manip.phase
    }

    /// The underlying market configuration (probe diagnostics).
    pub fn config_market(&self) -> &HeteroMarketConfig {
        &self.config.market
    }

    /// Mark at the moment the pump phase began (diagnostics).
    pub fn manip_pump_start_mark(&self) -> Price {
        self.manip.pump_start_mark
    }

    /// Highest mark seen during the pump (diagnostics).
    pub fn manip_pump_peak_mark(&self) -> Price {
        self.manip.pump_peak_mark
    }

    /// Shares the manipulator actually sold during distribution.
    pub fn manip_distributed_shares(&self) -> Quantity {
        self.manip.distributed_shares
    }

    // ------------------------------------------------------------------
    // The manipulator.
    // ------------------------------------------------------------------

    fn wake_manip(&mut self, now_ms: SimTime) {
        let account_id = self.manip.account_id;
        let mark = self.inner.core().mark_price();
        match self.manip.phase {
            ManipPhase::Accumulate => {
                // Join the bid (or quote just below the mid on a cold
                // book): build inventory without moving the price.
                let bid = self
                    .inner
                    .core()
                    .exchange
                    .book()
                    .best_bid()
                    .unwrap_or((mark - 1).max(1));
                let request = LimitOrderRequest {
                    account_id,
                    side: Side::Buy,
                    limit_price: bid,
                    quantity: self.config.manip.accumulate_lots * LOT_SIZE,
                };
                self.inner.core().submit_and_track(request, now_ms);
                // Inventory is measured by position, not by resting ids;
                // quote refresh is left to the crowd's own flow.
                let held = self.total_lots(account_id);
                let seed = (self.config.manip_seed_shares / LOT_SIZE) as i64;
                let target = seed + 150; // inventory goal of the cycle
                let elapsed = now_ms.saturating_sub(self.manip.phase_start_ms);
                if held >= target || elapsed >= self.config.manip.accumulate_ms {
                    self.manip.phase = ManipPhase::Pump;
                    self.manip.phase_start_ms = now_ms;
                    self.manip.pump_start_mark = mark;
                    self.manip.pump_peak_mark = mark;
                }
            }
            ManipPhase::Pump => {
                // Sweep through the offers: a marketable buy priced a few
                // ticks through the touch, capped at limit up.
                let (lower, upper) = self
                    .inner
                    .core()
                    .exchange
                    .price_limits()
                    .map(|limits| (limits.lower, limits.upper))
                    .unwrap_or((1, mark + 3));
                let sweep = (mark + 3).clamp(lower, upper);
                let lots = self.config.manip.pump_lots;
                let request = LimitOrderRequest {
                    account_id,
                    side: Side::Buy,
                    limit_price: sweep,
                    quantity: lots * LOT_SIZE,
                };
                self.inner.core().submit_and_track(request, now_ms);
                let mark_now = self.inner.core().mark_price();
                if mark_now > self.manip.pump_peak_mark {
                    self.manip.pump_peak_mark = mark_now;
                }
                let elapsed = now_ms.saturating_sub(self.manip.phase_start_ms);
                if elapsed >= self.config.manip.pump_ms {
                    self.manip.phase = ManipPhase::Distribute;
                    self.manip.phase_start_ms = now_ms;
                }
            }
            ManipPhase::Distribute => {
                // Sell into the bid until the acquired inventory is gone.
                // Inventory bought today is T+1-locked, so distribution
                // waits across settlement boundaries like a real
                // operator working a multi-day exit.
                let core = self.inner.core();
                let position = core
                    .exchange
                    .account(account_id)
                    .map(|account| account.position(&core.symbol))
                    .unwrap_or_default();
                let seed = (self.config.manip_seed_shares / LOT_SIZE) as i64;
                let acquired_sellable = (position.sellable / LOT_SIZE - seed).max(0);
                let acquired_unsettled = position.unsettled_buys / LOT_SIZE;
                if acquired_sellable > 0 {
                    let lots = self.config.manip.distribute_lots.min(acquired_sellable);
                    let bid = core.exchange.book().best_bid().unwrap_or((mark - 1).max(1));
                    let request = LimitOrderRequest {
                        account_id,
                        side: Side::Sell,
                        limit_price: bid,
                        quantity: lots * LOT_SIZE,
                    };
                    if let Ok(result) = core.submit_verbose(request, now_ms) {
                        let sold: Quantity = result.trades.iter().map(|t| t.quantity).sum();
                        self.manip.distributed_shares += sold;
                    }
                }
                let elapsed = now_ms.saturating_sub(self.manip.phase_start_ms);
                let inventory_left = acquired_sellable + acquired_unsettled;
                if elapsed >= self.config.manip.distribute_ms
                    || (inventory_left == 0 && elapsed > 0)
                {
                    self.manip.phase = ManipPhase::Done;
                }
            }
            ManipPhase::Done => {
                return; // no further wake-ups
            }
        }

        let rate = self.config.manip.wake_rate_per_second;
        let gap = self.inner.core().rng.poisson_gap_ms(rate);
        let next = now_ms.checked_add(gap).expect("sim time overflow");
        self.inner.core().schedule(next, KIND_MANIP, 0);
    }

    // ------------------------------------------------------------------
    // Helpers.
    // ------------------------------------------------------------------

    /// Total shares held (settled + today's buys), in lots.
    fn total_lots(&self, account_id: AccountId) -> i64 {
        let core = self.inner.core_view();
        core.exchange
            .account(account_id)
            .map(|account| {
                let position = account.position(&core.symbol);
                (position.settled + position.unsettled_buys) / LOT_SIZE
            })
            .unwrap_or(0)
    }
}

impl MarketDriver for GameMarket {
    fn core(&mut self) -> &mut MarketCore {
        self.inner.core()
    }

    fn wake(&mut self, kind: u8, index: usize, now_ms: SimTime) {
        if kind == KIND_MANIP {
            self.wake_manip(now_ms);
        } else {
            // Every agent kind of the underlying heterogeneous market.
            self.inner.wake(kind, index, now_ms);
        }
    }
}
