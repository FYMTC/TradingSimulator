//! JSON protocol between the browser front end and one game session.
//!
//! Client -> server messages drive the session (orders, cancels, clock
//! speed); server -> client messages are full snapshots at the tick rate
//! plus per-request acknowledgements.  Prices are integer ticks and
//! quantities integer shares exactly as in the core, so the wire format
//! never introduces floating-point ambiguity.

use serde::{Deserialize, Serialize};
use trading_simulator::game::ManipPhase;
use trading_simulator::{LevelSnapshot, Price, Quantity, SimTime};

/// One message from the browser.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMsg {
    /// Place a limit order (quantity in whole lots).
    Order {
        side: SideTag,
        price: Price,
        lots: i64,
    },
    /// Cancel one resting order.
    Cancel { order_id: u64 },
    /// Set the virtual clock multiplier (0 pauses, 1 = real time).
    Speed { multiplier: f64 },
}

/// One message to the browser.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMsg {
    /// Full market + player snapshot, broadcast every tick.
    Snapshot(Snapshot),
    /// Outcome of a client request, in request order.
    Ack(Ack),
}

/// Buy/sell discriminator on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SideTag {
    Buy,
    Sell,
}

impl From<SideTag> for trading_simulator::Side {
    fn from(tag: SideTag) -> Self {
        match tag {
            SideTag::Buy => Self::Buy,
            SideTag::Sell => Self::Sell,
        }
    }
}

/// One executed fill inside an acknowledgement.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct FillDto {
    pub price: Price,
    pub quantity: Quantity,
}

/// Outcome of an order or cancel request.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Ack {
    /// Echo of the sequence number the client stamped on the request.
    pub seq: u64,
    pub ok: bool,
    /// Present when the order was accepted (resting id, 0 if fully
    /// filled immediately).
    pub order_id: Option<u64>,
    pub fills: Vec<FillDto>,
    /// Rejection reason when `ok` is false.
    pub error: Option<String>,
}

/// A depth level on the wire.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct LevelDto {
    pub price: Price,
    pub quantity: Quantity,
    pub orders: usize,
}

impl From<LevelSnapshot> for LevelDto {
    fn from(level: LevelSnapshot) -> Self {
        Self {
            price: level.price,
            quantity: level.quantity,
            orders: level.order_count,
        }
    }
}

/// One OHLCV bar on the wire.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct BarDto {
    /// Bar start time, simulated milliseconds.
    pub t: SimTime,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: Quantity,
}

/// One recent tape print on the wire.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct TapeDto {
    pub t: SimTime,
    pub price: Price,
    pub quantity: Quantity,
}

/// Phase tag mirrored to the UI (the manipulator's story arc).
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ManipTag {
    Accumulate,
    Pump,
    Distribute,
    Done,
}

impl From<ManipPhase> for ManipTag {
    fn from(phase: ManipPhase) -> Self {
        match phase {
            ManipPhase::Accumulate => Self::Accumulate,
            ManipPhase::Pump => Self::Pump,
            ManipPhase::Distribute => Self::Distribute,
            ManipPhase::Done => Self::Done,
        }
    }
}

/// Macro regime tag mirrored to the UI.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RegimeTag {
    Calm,
    Bull,
    Bear,
    Crisis,
}

impl From<trading_simulator::hetero::Regime> for RegimeTag {
    fn from(regime: trading_simulator::hetero::Regime) -> Self {
        use trading_simulator::hetero::Regime as R;
        match regime {
            R::Calm => Self::Calm,
            R::Bull => Self::Bull,
            R::Bear => Self::Bear,
            R::Crisis => Self::Crisis,
        }
    }
}

/// The player's account and open orders.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct PlayerDto {
    pub cash_available: i128,
    pub cash_reserved: i128,
    pub settled: Quantity,
    pub unsettled_buys: Quantity,
    pub sellable: Quantity,
    pub equity: i128,
    pub open_orders: Vec<OpenOrderDto>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct OpenOrderDto {
    pub order_id: u64,
    pub side: SideTagWire,
    pub price: Price,
    pub remaining: Quantity,
}

/// Resting-order side, serialized (client orders only).
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SideTagWire {
    Buy,
    Sell,
}

/// The full snapshot broadcast every tick.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Snapshot {
    pub now_ms: SimTime,
    pub regime: RegimeTag,
    pub best_bid: Option<Price>,
    pub best_ask: Option<Price>,
    pub mark: Price,
    pub last_trade: Option<Price>,
    /// `(lower, upper)` of the daily price-limit band.
    pub limit: Option<(Price, Price)>,
    pub bids: Vec<LevelDto>,
    pub asks: Vec<LevelDto>,
    pub bars: Vec<BarDto>,
    pub tape: Vec<TapeDto>,
    pub manip_phase: ManipTag,
    pub player: PlayerDto,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_messages_round_trip_through_json() {
        let cases = vec![
            (
                r#"{"type":"order","side":"buy","price":1000,"lots":2}"#,
                ClientMsg::Order {
                    side: SideTag::Buy,
                    price: 1000,
                    lots: 2,
                },
            ),
            (
                r#"{"type":"order","side":"sell","price":999,"lots":1}"#,
                ClientMsg::Order {
                    side: SideTag::Sell,
                    price: 999,
                    lots: 1,
                },
            ),
            (
                r#"{"type":"cancel","order_id":7}"#,
                ClientMsg::Cancel { order_id: 7 },
            ),
            (
                r#"{"type":"speed","multiplier":0}"#,
                ClientMsg::Speed { multiplier: 0.0 },
            ),
        ];
        for (wire, expected) in cases {
            assert_eq!(serde_json::from_str::<ClientMsg>(wire).unwrap(), expected);
        }
    }

    #[test]
    fn snapshots_serialize_with_snake_case_tags() {
        let snapshot = Snapshot {
            now_ms: 1_000,
            regime: RegimeTag::Calm,
            best_bid: Some(999),
            best_ask: Some(1_001),
            mark: 1_000,
            last_trade: None,
            limit: Some((900, 1_100)),
            bids: vec![LevelDto {
                price: 999,
                quantity: 300,
                orders: 2,
            }],
            asks: vec![],
            bars: vec![BarDto {
                t: 0,
                open: 1_000,
                high: 1_002,
                low: 998,
                close: 1_000,
                volume: 1_200,
            }],
            tape: vec![],
            manip_phase: ManipTag::Accumulate,
            player: PlayerDto {
                cash_available: 9_000_000,
                cash_reserved: 200_000,
                settled: 20_000,
                unsettled_buys: 0,
                sellable: 20_000,
                equity: 29_200_000,
                open_orders: vec![OpenOrderDto {
                    order_id: 3,
                    side: SideTagWire::Buy,
                    price: 999,
                    remaining: 100,
                }],
            },
        };
        let wire = serde_json::to_string(&ServerMsg::Snapshot(snapshot)).unwrap();
        assert!(wire.contains(r#""manip_phase":"accumulate""#), "{wire}");
        assert!(wire.contains(r#""regime":"calm""#), "{wire}");
        assert!(wire.contains(r#""best_bid":999"#), "{wire}");
    }
}
