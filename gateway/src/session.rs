//! One game session: a [`GameMarket`] plus the virtual-clock state that
//! drives it.
//!
//! A session is owned by exactly one async task, so the market stays
//! single-threaded and its determinism contract is untouched - the
//! gateway only ever calls [`Session::advance`], [`Session::order`] and
//! [`Session::cancel`] from that task.  Nothing here touches sockets,
//! which keeps the whole loop unit-testable.

use trading_simulator::game::{GameConfig, GameMarket, GameSnapshot};
use trading_simulator::hetero::HeteroMarketConfig;
use trading_simulator::{Price, SimTime};

use crate::protocol::{
    Ack, BarDto, ClientMsg, FillDto, LevelDto, ManipTag, OpenOrderDto, PlayerDto, RegimeTag,
    ServerMsg, SideTagWire, Snapshot, TapeDto,
};

/// Width of one OHLCV bar on the wire, simulated milliseconds.
pub const BAR_WIDTH_MS: SimTime = 5_000;
/// Recent tape prints carried per snapshot.
pub const TAPE_WINDOW: usize = 40;
/// Bars carried per snapshot.
pub const BAR_WINDOW: usize = 180;
/// Real milliseconds between ticks.
pub const TICK_MS: u64 = 100;
/// Default virtual-clock multiplier (1 = real time).
pub const DEFAULT_SPEED: f64 = 1.0;

/// A playable session with its own seed and virtual clock.
pub struct Session {
    market: GameMarket,
    speed: f64,
    /// Simulated milliseconds already advanced during the last tick;
    /// fractional speeds accumulate here so 0.5x still moves the clock.
    sim_remainder_ms: f64,
}

impl Session {
    /// Creates a session from a client-provided seed (falling back to a
    /// fixed one), warm-starting the book so the UI opens on a live
    /// market instead of an empty tape.
    pub fn new(seed: Option<u64>) -> Self {
        let config = GameConfig {
            market: HeteroMarketConfig {
                seed: seed.unwrap_or(2026),
                ..HeteroMarketConfig::default()
            },
            ..GameConfig::default()
        };
        let mut market = GameMarket::new(config);
        // Warm start: two simulated minutes of agent activity before the
        // first snapshot, so candles and depth exist from tick one.
        market.advance_ms(120_000);
        Self {
            market,
            speed: DEFAULT_SPEED,
            sim_remainder_ms: 0.0,
        }
    }

    pub fn speed(&self) -> f64 {
        self.speed
    }

    pub fn now_ms(&self) -> SimTime {
        self.market.now_ms()
    }

    /// One real-time tick: advances the virtual clock by
    /// `TICK_MS * speed` simulated milliseconds.
    pub fn advance(&mut self) {
        self.sim_remainder_ms += TICK_MS as f64 * self.speed;
        let whole = self.sim_remainder_ms.floor() as u64;
        if whole > 0 {
            self.sim_remainder_ms -= whole as f64;
            self.market.advance_ms(whole);
        }
    }

    /// Applies one client message; order/cancel requests produce an ack.
    pub fn apply(&mut self, msg: ClientMsg, seq: u64) -> Option<ServerMsg> {
        match msg {
            ClientMsg::Order {
                side,
                price,
                lots,
            } => Some(self.order(side.into(), price, lots, seq)),
            ClientMsg::Cancel { order_id } => Some(self.cancel(order_id, seq)),
            ClientMsg::Speed { multiplier } => {
                self.speed = multiplier.clamp(0.0, 100.0);
                None
            }
        }
    }

    fn order(
        &mut self,
        side: trading_simulator::Side,
        price: Price,
        lots: i64,
        seq: u64,
    ) -> ServerMsg {
        if lots <= 0 {
            return ServerMsg::Ack(Ack {
                seq,
                ok: false,
                order_id: None,
                fills: Vec::new(),
                error: Some("lots must be positive".to_owned()),
            });
        }
        match self.market.submit_player_order(side, price, lots) {
            Ok(result) => ServerMsg::Ack(Ack {
                seq,
                ok: true,
                order_id: (result.remaining > 0).then_some(result.order_id),
                fills: result
                    .trades
                    .iter()
                    .map(|trade| FillDto {
                        price: trade.price,
                        quantity: trade.quantity,
                    })
                    .collect(),
                error: None,
            }),
            Err(error) => ServerMsg::Ack(Ack {
                seq,
                ok: false,
                order_id: None,
                fills: Vec::new(),
                error: Some(format!("{error:?}")),
            }),
        }
    }

    fn cancel(&mut self, order_id: u64, seq: u64) -> ServerMsg {
        match self.market.cancel_player_order(order_id) {
            Ok(result) => ServerMsg::Ack(Ack {
                seq,
                ok: true,
                order_id: Some(result.order_id),
                fills: Vec::new(),
                error: None,
            }),
            Err(error) => ServerMsg::Ack(Ack {
                seq,
                ok: false,
                order_id: None,
                fills: Vec::new(),
                error: Some(format!("{error:?}")),
            }),
        }
    }

    /// Builds the full snapshot for the wire.
    pub fn snapshot(&self) -> ServerMsg {
        let view = self.market.snapshot();
        ServerMsg::Snapshot(self.to_dto(&view))
    }

    fn to_dto(&self, view: &GameSnapshot) -> Snapshot {
        let bars: Vec<BarDto> = self
            .market
            .bars(BAR_WIDTH_MS)
            .into_iter()
            .rev()
            .take(BAR_WINDOW)
            .rev()
            .map(|bar| BarDto {
                t: bar.start_ms,
                open: bar.open,
                high: bar.high,
                low: bar.low,
                close: bar.close,
                volume: bar.volume,
            })
            .collect();
        let tape: Vec<TapeDto> = self
            .market
            .tape()
            .iter()
            .rev()
            .take(TAPE_WINDOW)
            .rev()
            .map(|print| TapeDto {
                t: print.time_ms,
                price: print.price,
                quantity: print.quantity,
            })
            .collect();
        Snapshot {
            now_ms: view.now_ms,
            regime: RegimeTag::from(view.regime),
            best_bid: view.best_bid,
            best_ask: view.best_ask,
            mark: view.mark,
            last_trade: view.last_trade,
            limit: view.price_limits,
            bids: view.depth_bids.into_iter().map(LevelDto::from).collect(),
            asks: view.depth_asks.into_iter().map(LevelDto::from).collect(),
            bars,
            tape,
            manip_phase: ManipTag::from(view.manip_phase),
            player: PlayerDto {
                cash_available: view.player.cash_available,
                cash_reserved: view.player.cash_reserved,
                settled: view.player.settled,
                unsettled_buys: view.player.unsettled_buys,
                sellable: view.player.sellable,
                equity: view.player.equity,
                open_orders: view
                    .player
                    .open_orders
                    .iter()
                    .map(|order| OpenOrderDto {
                        order_id: order.id,
                        side: match order.side {
                            trading_simulator::Side::Buy => SideTagWire::Buy,
                            trading_simulator::Side::Sell => SideTagWire::Sell,
                        },
                        price: order.limit_price,
                        remaining: order.remaining,
                    })
                    .collect(),
            },
        }
    }
}

/// Splits one snapshot payload for tests asserting on inner fields.
#[cfg(test)]
pub(crate) fn unwrap_snapshot(msg: &ServerMsg) -> &Snapshot {
    match msg {
        ServerMsg::Snapshot(snapshot) => snapshot,
        other => panic!("expected a snapshot, got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use crate::protocol::{Ack, ClientMsg, SideTag};

    use super::*;

    fn session() -> Session {
        // Fresh session per test: deterministic seed, warm start.
        Session::new(Some(7))
    }

    #[test]
    fn warm_start_opens_on_a_live_market() {
        let session = session();
        let snapshot = unwrap_snapshot(&session.snapshot());
        assert!(session.now_ms() >= 120_000);
        assert!(snapshot.best_bid.is_some() && snapshot.best_ask.is_some());
        assert!(!snapshot.bars.is_empty(), "candles exist from tick one");
        assert!(!snapshot.tape.is_empty());
        assert!(snapshot.limit.is_some());
    }

    #[test]
    fn advance_moves_the_clock_at_the_configured_speed() {
        let mut session = session();
        let start = session.now_ms();
        session.apply(
            ClientMsg::Speed { multiplier: 5.0 },
            0,
        );
        session.advance(); // 100ms real * 5x = 500ms sim
        assert_eq!(session.now_ms() - start, 500);
    }

    #[test]
    fn zero_speed_pauses_the_clock() {
        let mut session = session();
        let start = session.now_ms();
        session.apply(ClientMsg::Speed { multiplier: 0.0 }, 0);
        session.advance();
        session.advance();
        assert_eq!(session.now_ms(), start);
    }

    #[test]
    fn fractional_speed_accumulates_remainders() {
        let mut session = session();
        let start = session.now_ms();
        session.apply(ClientMsg::Speed { multiplier: 0.25 }, 0);
        for _ in 0..4 {
            session.advance(); // 25ms sim each, only every second lands.
        }
        assert_eq!(session.now_ms() - start, 100);
    }

    #[test]
    fn orders_acknowledge_fills_and_rejections() {
        let mut session = session();
        let snapshot = unwrap_snapshot(&session.snapshot());
        let ask = snapshot.best_ask.unwrap();

        let ack = session.apply(
            ClientMsg::Order {
                side: SideTag::Buy,
                price: ask,
                lots: 1,
            },
            1,
        );
        match ack {
            ServerMsg::Ack(Ack {
                seq,
                ok,
                fills,
                error,
                ..
            }) => {
                assert_eq!(seq, 1);
                assert!(ok, "{error:?}");
                assert!(!fills.is_empty(), "crossing order fills immediately");
            }
            other => panic!("expected an ack, got {other:?}"),
        }

        // An out-of-band price is rejected with the band in the reason.
        let (lower, upper) = snapshot.limit.unwrap();
        let ack = session.apply(
            ClientMsg::Order {
                side: SideTag::Buy,
                price: upper + 1,
                lots: 1,
            },
            2,
        );
        match ack {
            ServerMsg::Ack(Ack {
                seq,
                ok,
                error,
                ..
            }) => {
                assert_eq!(seq, 2);
                assert!(!ok);
                let reason = error.unwrap();
                assert!(reason.contains("PriceOutsideLimits"), "{reason}");
                let _ = lower;
            }
            other => panic!("expected an ack, got {other:?}"),
        }

        // Non-positive size is rejected without touching the market.
        let ack = session.apply(
            ClientMsg::Order {
                side: SideTag::Sell,
                price: ask,
                lots: 0,
            },
            3,
        );
        match ack {
            ServerMsg::Ack(Ack { ok, error, .. }) => {
                assert!(!ok);
                assert!(error.unwrap().contains("positive"));
            }
            other => panic!("expected an ack, got {other:?}"),
        }
    }

    #[test]
    fn resting_orders_can_be_cancelled_through_the_protocol() {
        let mut session = session();
        let snapshot = unwrap_snapshot(&session.snapshot());
        let price = snapshot.best_bid.unwrap();

        let order_id = match session.apply(
            ClientMsg::Order {
                side: SideTag::Buy,
                price,
                lots: 1,
            },
            1,
        ) {
            ServerMsg::Ack(Ack {
                order_id: Some(id),
                ..
            }) => id,
            other => panic!("expected a resting order, got {other:?}"),
        };

        let snapshot = unwrap_snapshot(&session.snapshot());
        assert_eq!(snapshot.player.open_orders.len(), 1);
        assert_eq!(snapshot.player.open_orders[0].order_id, order_id);

        session.apply(ClientMsg::Cancel { order_id }, 2);
        let snapshot = unwrap_snapshot(&session.snapshot());
        assert!(snapshot.player.open_orders.is_empty());
    }

    #[test]
    fn same_seed_sessions_stay_in_lockstep() {
        let run = || {
            let mut session = Session::new(Some(99));
            for _ in 0..10 {
                session.advance();
            }
            serde_json::to_string(&session.snapshot()).unwrap()
        };
        assert_eq!(run(), run());
    }
}
