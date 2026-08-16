//! Deterministic M0 matching core.
//!
//! The crate intentionally uses only the Rust standard library.  It provides a
//! single-instrument, price-time-priority order book plus the minimum account
//! state required to model cash reservation and A-share-style T+1 sellability.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap};

pub type AccountId = u64;
pub type OrderId = u64;
pub type TradeId = u64;
pub type Price = i64;
pub type Quantity = i64;
pub type Money = i128;
pub type SimTime = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    fn opposite(self) -> Self {
        match self {
            Self::Buy => Self::Sell,
            Self::Sell => Self::Buy,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LimitOrderRequest {
    pub account_id: AccountId,
    pub side: Side,
    /// Integer ticks; the tick size belongs to instrument configuration.
    pub limit_price: Price,
    pub quantity: Quantity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RestingOrder {
    pub id: OrderId,
    pub account_id: AccountId,
    pub side: Side,
    pub limit_price: Price,
    pub remaining: Quantity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IncomingOrder {
    id: OrderId,
    account_id: AccountId,
    side: Side,
    limit_price: Price,
    remaining: Quantity,
}

impl From<IncomingOrder> for RestingOrder {
    fn from(value: IncomingOrder) -> Self {
        Self {
            id: value.id,
            account_id: value.account_id,
            side: value.side,
            limit_price: value.limit_price,
            remaining: value.remaining,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NodeHandle {
    slot: usize,
    generation: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OrderNode {
    order: RestingOrder,
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct PriceLevel {
    head: Option<usize>,
    tail: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BookMatch {
    maker: RestingOrder,
    taker: IncomingOrder,
    execution_price: Price,
    quantity: Quantity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelSnapshot {
    pub price: Price,
    pub quantity: Quantity,
    pub order_count: usize,
}

/// A price-time-priority order book.
///
/// Price levels are sorted in `BTreeMap`s. Orders at each level live in an
/// intrusive linked list stored in an arena. `OrderId -> NodeHandle` makes
/// cancellation O(1) after the order lookup, while the generation protects the
/// arena from stale-handle reuse.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OrderBook {
    bids: BTreeMap<Price, PriceLevel>,
    asks: BTreeMap<Price, PriceLevel>,
    nodes: Vec<Option<OrderNode>>,
    generations: Vec<u64>,
    free_nodes: Vec<usize>,
    by_order: HashMap<OrderId, NodeHandle>,
}

impl OrderBook {
    pub fn best_bid(&self) -> Option<Price> {
        self.bids.last_key_value().map(|(price, _)| *price)
    }

    pub fn best_ask(&self) -> Option<Price> {
        self.asks.first_key_value().map(|(price, _)| *price)
    }

    pub fn resting_order_count(&self) -> usize {
        self.by_order.len()
    }

    pub fn order(&self, order_id: OrderId) -> Option<RestingOrder> {
        let handle = self.by_order.get(&order_id)?;
        if !self.is_live(*handle) {
            return None;
        }
        self.nodes[handle.slot].as_ref().map(|node| node.order)
    }

    /// Returns best-to-worst visible levels for `side`.
    pub fn depth(&self, side: Side, max_levels: usize) -> Vec<LevelSnapshot> {
        let mut snapshots = Vec::with_capacity(max_levels);
        match side {
            Side::Buy => {
                for (price, level) in self.bids.iter().rev().take(max_levels) {
                    snapshots.push(self.snapshot_level(*price, level));
                }
            }
            Side::Sell => {
                for (price, level) in self.asks.iter().take(max_levels) {
                    snapshots.push(self.snapshot_level(*price, level));
                }
            }
        }
        snapshots
    }

    fn snapshot_level(&self, price: Price, level: &PriceLevel) -> LevelSnapshot {
        let mut quantity = 0;
        let mut order_count = 0;
        let mut cursor = level.head;
        while let Some(slot) = cursor {
            let node = self.nodes[slot]
                .as_ref()
                .expect("price level references a live node");
            quantity += node.order.remaining;
            order_count += 1;
            cursor = node.next;
        }
        LevelSnapshot {
            price,
            quantity,
            order_count,
        }
    }

    fn levels(&self, side: Side) -> &BTreeMap<Price, PriceLevel> {
        match side {
            Side::Buy => &self.bids,
            Side::Sell => &self.asks,
        }
    }

    fn levels_mut(&mut self, side: Side) -> &mut BTreeMap<Price, PriceLevel> {
        match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        }
    }

    fn is_live(&self, handle: NodeHandle) -> bool {
        self.generations
            .get(handle.slot)
            .is_some_and(|generation| *generation == handle.generation)
            && self.nodes.get(handle.slot).is_some_and(Option::is_some)
    }

    fn allocate_node(&mut self, order: RestingOrder) -> NodeHandle {
        if let Some(slot) = self.free_nodes.pop() {
            debug_assert!(self.nodes[slot].is_none());
            self.nodes[slot] = Some(OrderNode {
                order,
                prev: None,
                next: None,
            });
            NodeHandle {
                slot,
                generation: self.generations[slot],
            }
        } else {
            let slot = self.nodes.len();
            self.nodes.push(Some(OrderNode {
                order,
                prev: None,
                next: None,
            }));
            self.generations.push(0);
            NodeHandle {
                slot,
                generation: 0,
            }
        }
    }

    fn add_resting(&mut self, order: RestingOrder) {
        debug_assert!(order.remaining > 0);
        let handle = self.allocate_node(order);
        let tail = self
            .levels(order.side)
            .get(&order.limit_price)
            .and_then(|level| level.tail);

        self.nodes[handle.slot]
            .as_mut()
            .expect("new node is live")
            .prev = tail;
        if let Some(tail_slot) = tail {
            self.nodes[tail_slot]
                .as_mut()
                .expect("level tail is live")
                .next = Some(handle.slot);
        }

        let level = self
            .levels_mut(order.side)
            .entry(order.limit_price)
            .or_default();
        if level.head.is_none() {
            level.head = Some(handle.slot);
        }
        level.tail = Some(handle.slot);
        self.by_order.insert(order.id, handle);
    }

    fn best_opposite_price(&self, incoming_side: Side) -> Option<Price> {
        match incoming_side {
            Side::Buy => self.asks.first_key_value().map(|(price, _)| *price),
            Side::Sell => self.bids.last_key_value().map(|(price, _)| *price),
        }
    }

    fn crosses(incoming_side: Side, limit_price: Price, opposite_price: Price) -> bool {
        match incoming_side {
            Side::Buy => limit_price >= opposite_price,
            Side::Sell => limit_price <= opposite_price,
        }
    }

    fn level_head(&self, side: Side, price: Price) -> Option<usize> {
        self.levels(side).get(&price).and_then(|level| level.head)
    }

    fn remove_node(&mut self, handle: NodeHandle) -> Option<RestingOrder> {
        if !self.is_live(handle) {
            return None;
        }

        let node = self.nodes[handle.slot]
            .as_ref()
            .expect("live handle has a node");
        let order = node.order;
        let prev = node.prev;
        let next = node.next;

        if let Some(prev_slot) = prev {
            self.nodes[prev_slot]
                .as_mut()
                .expect("previous node is live")
                .next = next;
        }
        if let Some(next_slot) = next {
            self.nodes[next_slot]
                .as_mut()
                .expect("next node is live")
                .prev = prev;
        }

        let level_is_empty = {
            let level = self
                .levels_mut(order.side)
                .get_mut(&order.limit_price)
                .expect("node price level exists");
            if level.head == Some(handle.slot) {
                level.head = next;
            }
            if level.tail == Some(handle.slot) {
                level.tail = prev;
            }
            level.head.is_none()
        };
        if level_is_empty {
            self.levels_mut(order.side).remove(&order.limit_price);
        }

        self.nodes[handle.slot] = None;
        self.generations[handle.slot] = self.generations[handle.slot].wrapping_add(1);
        self.free_nodes.push(handle.slot);
        self.by_order.remove(&order.id);
        Some(order)
    }

    fn match_incoming(&mut self, incoming: &mut IncomingOrder) -> Vec<BookMatch> {
        let mut matches = Vec::new();
        while incoming.remaining > 0 {
            let Some(best_price) = self.best_opposite_price(incoming.side) else {
                break;
            };
            if !Self::crosses(incoming.side, incoming.limit_price, best_price) {
                break;
            }

            let maker_side = incoming.side.opposite();
            let maker_slot = self
                .level_head(maker_side, best_price)
                .expect("best price has a head order");
            let maker = self.nodes[maker_slot]
                .as_ref()
                .expect("maker node is live")
                .order;
            let quantity = incoming.remaining.min(maker.remaining);

            incoming.remaining -= quantity;
            if quantity == maker.remaining {
                let handle = NodeHandle {
                    slot: maker_slot,
                    generation: self.generations[maker_slot],
                };
                self.remove_node(handle)
                    .expect("maker is removable through its live handle");
            } else {
                self.nodes[maker_slot]
                    .as_mut()
                    .expect("maker node is live")
                    .order
                    .remaining -= quantity;
            }

            matches.push(BookMatch {
                maker,
                taker: *incoming,
                execution_price: best_price,
                quantity,
            });
        }

        if incoming.remaining > 0 {
            self.add_resting((*incoming).into());
        }
        matches
    }

    fn cancel(&mut self, order_id: OrderId) -> Option<RestingOrder> {
        let handle = *self.by_order.get(&order_id)?;
        self.remove_node(handle)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Position {
    /// Shares currently settled and owned. Some of them may be locked in sells.
    pub settled: Quantity,
    /// Settled shares not locked in active sell orders.
    pub sellable: Quantity,
    /// Shares bought today; they become sellable at `settle_trading_day`.
    pub unsettled_buys: Quantity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Account {
    pub cash_available: Money,
    pub cash_reserved: Money,
    positions: BTreeMap<String, Position>,
}

impl Account {
    pub fn with_cash(cash_available: Money) -> Self {
        assert!(cash_available >= 0, "initial cash cannot be negative");
        Self {
            cash_available,
            cash_reserved: 0,
            positions: BTreeMap::new(),
        }
    }

    pub fn position(&self, symbol: &str) -> Position {
        self.positions.get(symbol).copied().unwrap_or_default()
    }

    pub fn seed_settled_position(&mut self, symbol: impl Into<String>, quantity: Quantity) {
        assert!(quantity >= 0, "initial position cannot be negative");
        let position = self.positions.entry(symbol.into()).or_default();
        position.settled += quantity;
        position.sellable += quantity;
    }

    fn reserve_buy(&mut self, limit_price: Price, quantity: Quantity) -> Result<(), ExchangeError> {
        let required = notional(limit_price, quantity)?;
        if self.cash_available < required {
            return Err(ExchangeError::InsufficientCash {
                required,
                available: self.cash_available,
            });
        }
        self.cash_available -= required;
        self.cash_reserved += required;
        Ok(())
    }

    fn release_buy(&mut self, limit_price: Price, quantity: Quantity) -> Result<(), ExchangeError> {
        let released = notional(limit_price, quantity)?;
        debug_assert!(self.cash_reserved >= released);
        self.cash_reserved -= released;
        self.cash_available += released;
        Ok(())
    }

    fn settle_buy_fill(
        &mut self,
        symbol: &str,
        limit_price: Price,
        execution_price: Price,
        quantity: Quantity,
    ) -> Result<(), ExchangeError> {
        let reserved = notional(limit_price, quantity)?;
        let paid = notional(execution_price, quantity)?;
        debug_assert!(reserved >= paid);
        debug_assert!(self.cash_reserved >= reserved);
        self.cash_reserved -= reserved;
        self.cash_available += reserved - paid;
        self.positions
            .entry(symbol.to_owned())
            .or_default()
            .unsettled_buys += quantity;
        Ok(())
    }

    fn reserve_sell(&mut self, symbol: &str, quantity: Quantity) -> Result<(), ExchangeError> {
        let position = self.positions.entry(symbol.to_owned()).or_default();
        if position.sellable < quantity {
            return Err(ExchangeError::InsufficientSellable {
                requested: quantity,
                sellable: position.sellable,
            });
        }
        position.sellable -= quantity;
        Ok(())
    }

    fn release_sell(&mut self, symbol: &str, quantity: Quantity) {
        self.positions
            .entry(symbol.to_owned())
            .or_default()
            .sellable += quantity;
    }

    fn settle_sell_fill(
        &mut self,
        symbol: &str,
        execution_price: Price,
        quantity: Quantity,
    ) -> Result<(), ExchangeError> {
        let proceeds = notional(execution_price, quantity)?;
        let position = self.positions.entry(symbol.to_owned()).or_default();
        debug_assert!(position.settled >= quantity);
        position.settled -= quantity;
        self.cash_available += proceeds;
        Ok(())
    }

    fn settle_trading_day(&mut self, symbol: &str) {
        let position = self.positions.entry(symbol.to_owned()).or_default();
        position.settled += position.unsettled_buys;
        position.sellable += position.unsettled_buys;
        position.unsettled_buys = 0;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExchangeError {
    DuplicateAccount(AccountId),
    UnknownAccount(AccountId),
    InvalidPrice(Price),
    InvalidQuantity(Quantity),
    InsufficientCash {
        required: Money,
        available: Money,
    },
    InsufficientSellable {
        requested: Quantity,
        sellable: Quantity,
    },
    UnknownOrder(OrderId),
    OrderOwnedByDifferentAccount {
        order_id: OrderId,
        owner: AccountId,
    },
    OrderIdOverflow,
    TradeIdOverflow,
    DuplicateEventKey(EventKey),
}

fn notional(price: Price, quantity: Quantity) -> Result<Money, ExchangeError> {
    if price <= 0 {
        return Err(ExchangeError::InvalidPrice(price));
    }
    if quantity <= 0 {
        return Err(ExchangeError::InvalidQuantity(quantity));
    }
    Ok(Money::from(price) * Money::from(quantity))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trade {
    pub id: TradeId,
    pub price: Price,
    pub quantity: Quantity,
    pub maker_order_id: OrderId,
    pub taker_order_id: OrderId,
    pub buyer_account_id: AccountId,
    pub seller_account_id: AccountId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmitResult {
    pub order_id: OrderId,
    pub remaining: Quantity,
    pub trades: Vec<Trade>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CancelResult {
    pub order_id: OrderId,
    pub released_quantity: Quantity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exchange {
    symbol: String,
    accounts: BTreeMap<AccountId, Account>,
    book: OrderBook,
    trades: Vec<Trade>,
    next_order_id: OrderId,
    next_trade_id: TradeId,
}

impl Exchange {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            accounts: BTreeMap::new(),
            book: OrderBook::default(),
            trades: Vec::new(),
            next_order_id: 1,
            next_trade_id: 1,
        }
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    pub fn book(&self) -> &OrderBook {
        &self.book
    }

    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    pub fn account(&self, account_id: AccountId) -> Option<&Account> {
        self.accounts.get(&account_id)
    }

    pub fn add_account(
        &mut self,
        account_id: AccountId,
        account: Account,
    ) -> Result<(), ExchangeError> {
        if self.accounts.contains_key(&account_id) {
            return Err(ExchangeError::DuplicateAccount(account_id));
        }
        self.accounts.insert(account_id, account);
        Ok(())
    }

    pub fn submit_limit_order(
        &mut self,
        request: LimitOrderRequest,
    ) -> Result<SubmitResult, ExchangeError> {
        notional(request.limit_price, request.quantity)?;
        let order_id = self.next_order_id;
        let next_order_id = self
            .next_order_id
            .checked_add(1)
            .ok_or(ExchangeError::OrderIdOverflow)?;

        let account = self
            .accounts
            .get_mut(&request.account_id)
            .ok_or(ExchangeError::UnknownAccount(request.account_id))?;
        match request.side {
            Side::Buy => account.reserve_buy(request.limit_price, request.quantity)?,
            Side::Sell => account.reserve_sell(&self.symbol, request.quantity)?,
        }
        self.next_order_id = next_order_id;

        let mut incoming = IncomingOrder {
            id: order_id,
            account_id: request.account_id,
            side: request.side,
            limit_price: request.limit_price,
            remaining: request.quantity,
        };
        let matches = self.book.match_incoming(&mut incoming);
        let mut trades = Vec::with_capacity(matches.len());
        for book_match in matches {
            let trade = self.settle_match(book_match)?;
            self.trades.push(trade);
            trades.push(trade);
        }

        Ok(SubmitResult {
            order_id,
            remaining: incoming.remaining,
            trades,
        })
    }

    fn settle_match(&mut self, book_match: BookMatch) -> Result<Trade, ExchangeError> {
        let (buyer_id, buyer_limit, seller_id) = match book_match.taker.side {
            Side::Buy => (
                book_match.taker.account_id,
                book_match.taker.limit_price,
                book_match.maker.account_id,
            ),
            Side::Sell => (
                book_match.maker.account_id,
                book_match.maker.limit_price,
                book_match.taker.account_id,
            ),
        };

        if buyer_id == seller_id {
            let account = self
                .accounts
                .get_mut(&buyer_id)
                .ok_or(ExchangeError::UnknownAccount(buyer_id))?;
            account.settle_buy_fill(
                &self.symbol,
                buyer_limit,
                book_match.execution_price,
                book_match.quantity,
            )?;
            account.settle_sell_fill(
                &self.symbol,
                book_match.execution_price,
                book_match.quantity,
            )?;
        } else {
            self.accounts
                .get_mut(&buyer_id)
                .ok_or(ExchangeError::UnknownAccount(buyer_id))?
                .settle_buy_fill(
                    &self.symbol,
                    buyer_limit,
                    book_match.execution_price,
                    book_match.quantity,
                )?;
            self.accounts
                .get_mut(&seller_id)
                .ok_or(ExchangeError::UnknownAccount(seller_id))?
                .settle_sell_fill(
                    &self.symbol,
                    book_match.execution_price,
                    book_match.quantity,
                )?;
        }

        let id = self.next_trade_id;
        self.next_trade_id = self
            .next_trade_id
            .checked_add(1)
            .ok_or(ExchangeError::TradeIdOverflow)?;
        Ok(Trade {
            id,
            price: book_match.execution_price,
            quantity: book_match.quantity,
            maker_order_id: book_match.maker.id,
            taker_order_id: book_match.taker.id,
            buyer_account_id: buyer_id,
            seller_account_id: seller_id,
        })
    }

    pub fn cancel_order(
        &mut self,
        account_id: AccountId,
        order_id: OrderId,
    ) -> Result<CancelResult, ExchangeError> {
        let resting = self
            .book
            .order(order_id)
            .ok_or(ExchangeError::UnknownOrder(order_id))?;
        if resting.account_id != account_id {
            return Err(ExchangeError::OrderOwnedByDifferentAccount {
                order_id,
                owner: resting.account_id,
            });
        }
        let cancelled = self
            .book
            .cancel(order_id)
            .expect("order was present immediately before cancellation");
        let account = self
            .accounts
            .get_mut(&account_id)
            .ok_or(ExchangeError::UnknownAccount(account_id))?;
        match cancelled.side {
            Side::Buy => account.release_buy(cancelled.limit_price, cancelled.remaining)?,
            Side::Sell => account.release_sell(&self.symbol, cancelled.remaining),
        }
        Ok(CancelResult {
            order_id,
            released_quantity: cancelled.remaining,
        })
    }

    /// Performs the T+1 settlement boundary for the exchange's instrument.
    pub fn settle_trading_day(&mut self) {
        for account in self.accounts.values_mut() {
            account.settle_trading_day(&self.symbol);
        }
    }

    /// Replays a uniquely keyed event log in canonical key order.
    pub fn replay(&mut self, mut events: Vec<Event>) -> Result<Vec<ProcessedEvent>, ExchangeError> {
        events.sort_by_key(|event| event.key);
        for pair in events.windows(2) {
            if pair[0].key == pair[1].key {
                return Err(ExchangeError::DuplicateEventKey(pair[0].key));
            }
        }

        let mut processed = Vec::with_capacity(events.len());
        for event in events {
            let outcome = match event.kind {
                EventKind::Submit(request) => {
                    EventOutcome::Submitted(self.submit_limit_order(request))
                }
                EventKind::Cancel {
                    account_id,
                    order_id,
                } => EventOutcome::Cancelled(self.cancel_order(account_id, order_id)),
                EventKind::SettleTradingDay => {
                    self.settle_trading_day();
                    EventOutcome::Settled
                }
            };
            processed.push(ProcessedEvent { event, outcome });
        }
        Ok(processed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EventKey {
    pub sim_time: SimTime,
    pub source_priority: u8,
    pub source_seq: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EventKind {
    Submit(LimitOrderRequest),
    Cancel {
        account_id: AccountId,
        order_id: OrderId,
    },
    SettleTradingDay,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    pub key: EventKey,
    pub kind: EventKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EventOutcome {
    Submitted(Result<SubmitResult, ExchangeError>),
    Cancelled(Result<CancelResult, ExchangeError>),
    Settled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessedEvent {
    pub event: Event,
    pub outcome: EventOutcome,
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYMBOL: &str = "600000.SH";

    fn funded_exchange() -> Exchange {
        let mut exchange = Exchange::new(SYMBOL);
        exchange
            .add_account(1, Account::with_cash(100_000))
            .unwrap();
        exchange
            .add_account(2, Account::with_cash(100_000))
            .unwrap();
        exchange
            .add_account(3, Account::with_cash(100_000))
            .unwrap();
        exchange
            .add_account(4, Account::with_cash(100_000))
            .unwrap();
        exchange
            .accounts
            .get_mut(&1)
            .unwrap()
            .seed_settled_position(SYMBOL, 100);
        exchange
            .accounts
            .get_mut(&2)
            .unwrap()
            .seed_settled_position(SYMBOL, 100);
        exchange
            .accounts
            .get_mut(&3)
            .unwrap()
            .seed_settled_position(SYMBOL, 100);
        exchange
    }

    fn order(
        account_id: AccountId,
        side: Side,
        price: Price,
        quantity: Quantity,
    ) -> LimitOrderRequest {
        LimitOrderRequest {
            account_id,
            side,
            limit_price: price,
            quantity,
        }
    }

    #[test]
    fn matches_best_price_before_arrival_time() {
        let mut exchange = funded_exchange();
        let higher_ask = exchange
            .submit_limit_order(order(1, Side::Sell, 101, 5))
            .unwrap();
        let lower_ask = exchange
            .submit_limit_order(order(2, Side::Sell, 100, 5))
            .unwrap();

        let result = exchange
            .submit_limit_order(order(3, Side::Buy, 102, 7))
            .unwrap();

        assert_eq!(result.remaining, 0);
        assert_eq!(result.trades.len(), 2);
        assert_eq!(result.trades[0].price, 100);
        assert_eq!(result.trades[0].quantity, 5);
        assert_eq!(result.trades[0].maker_order_id, lower_ask.order_id);
        assert_eq!(result.trades[1].price, 101);
        assert_eq!(result.trades[1].quantity, 2);
        assert_eq!(result.trades[1].maker_order_id, higher_ask.order_id);
        assert_eq!(
            exchange
                .book()
                .order(higher_ask.order_id)
                .unwrap()
                .remaining,
            3
        );
        assert_eq!(exchange.book().best_ask(), Some(101));
    }

    #[test]
    fn preserves_time_priority_within_a_price_level() {
        let mut exchange = funded_exchange();
        let first = exchange
            .submit_limit_order(order(1, Side::Sell, 100, 4))
            .unwrap();
        let second = exchange
            .submit_limit_order(order(2, Side::Sell, 100, 4))
            .unwrap();

        let result = exchange
            .submit_limit_order(order(3, Side::Buy, 100, 6))
            .unwrap();

        assert_eq!(result.trades[0].maker_order_id, first.order_id);
        assert_eq!(result.trades[1].maker_order_id, second.order_id);
        assert_eq!(exchange.book().order(second.order_id).unwrap().remaining, 2);
    }

    #[test]
    fn cancellation_releases_the_exact_unfilled_cash_reservation() {
        let mut exchange = funded_exchange();
        let buy = exchange
            .submit_limit_order(order(3, Side::Buy, 100, 10))
            .unwrap();
        assert_eq!(exchange.account(3).unwrap().cash_available, 99_000);
        assert_eq!(exchange.account(3).unwrap().cash_reserved, 1_000);

        exchange
            .submit_limit_order(order(1, Side::Sell, 99, 4))
            .unwrap();
        assert_eq!(exchange.account(3).unwrap().cash_available, 99_000);
        assert_eq!(exchange.account(3).unwrap().cash_reserved, 600);

        let cancelled = exchange.cancel_order(3, buy.order_id).unwrap();
        assert_eq!(cancelled.released_quantity, 6);
        assert_eq!(exchange.account(3).unwrap().cash_available, 99_600);
        assert_eq!(exchange.account(3).unwrap().cash_reserved, 0);
        assert_eq!(exchange.book().resting_order_count(), 0);
    }

    #[test]
    fn t_plus_one_blocks_same_day_resale_until_settlement() {
        let mut exchange = funded_exchange();
        exchange
            .submit_limit_order(order(1, Side::Sell, 100, 10))
            .unwrap();
        exchange
            .submit_limit_order(order(4, Side::Buy, 100, 10))
            .unwrap();

        assert_eq!(
            exchange.account(4).unwrap().position(SYMBOL).unsettled_buys,
            10
        );
        assert_eq!(
            exchange.submit_limit_order(order(4, Side::Sell, 100, 10)),
            Err(ExchangeError::InsufficientSellable {
                requested: 10,
                sellable: 0,
            })
        );

        exchange.settle_trading_day();
        let resale = exchange
            .submit_limit_order(order(4, Side::Sell, 100, 10))
            .unwrap();
        assert_eq!(resale.remaining, 10);
        assert_eq!(exchange.account(4).unwrap().position(SYMBOL).sellable, 0);
    }

    #[test]
    fn rejected_submission_does_not_consume_an_order_id() {
        let mut exchange = funded_exchange();
        assert_eq!(
            exchange.submit_limit_order(order(4, Side::Buy, 100, 1_001)),
            Err(ExchangeError::InsufficientCash {
                required: 100_100,
                available: 100_000,
            })
        );
        assert_eq!(exchange.account(4).unwrap().cash_available, 100_000);
        assert_eq!(exchange.account(4).unwrap().cash_reserved, 0);

        let accepted = exchange
            .submit_limit_order(order(1, Side::Sell, 100, 1))
            .unwrap();
        assert_eq!(accepted.order_id, 1);
    }

    #[test]
    fn duplicate_account_does_not_replace_the_original_account() {
        let mut exchange = Exchange::new(SYMBOL);
        exchange.add_account(1, Account::with_cash(10)).unwrap();
        assert_eq!(
            exchange.add_account(1, Account::with_cash(99)),
            Err(ExchangeError::DuplicateAccount(1))
        );
        assert_eq!(exchange.account(1).unwrap().cash_available, 10);
    }

    #[test]
    fn depth_is_best_to_worst_and_aggregates_a_level() {
        let mut exchange = funded_exchange();
        exchange
            .submit_limit_order(order(1, Side::Buy, 99, 4))
            .unwrap();
        exchange
            .submit_limit_order(order(2, Side::Buy, 99, 6))
            .unwrap();
        exchange
            .submit_limit_order(order(3, Side::Buy, 98, 5))
            .unwrap();

        assert_eq!(
            exchange.book().depth(Side::Buy, 10),
            vec![
                LevelSnapshot {
                    price: 99,
                    quantity: 10,
                    order_count: 2,
                },
                LevelSnapshot {
                    price: 98,
                    quantity: 5,
                    order_count: 1,
                },
            ]
        );
    }

    #[test]
    fn replay_sorts_events_and_reproduces_the_same_state() {
        let events = vec![
            Event {
                key: EventKey {
                    sim_time: 20,
                    source_priority: 1,
                    source_seq: 1,
                },
                kind: EventKind::Submit(order(3, Side::Buy, 101, 4)),
            },
            Event {
                key: EventKey {
                    sim_time: 10,
                    source_priority: 1,
                    source_seq: 2,
                },
                kind: EventKind::Submit(order(2, Side::Sell, 100, 5)),
            },
            Event {
                key: EventKey {
                    sim_time: 30,
                    source_priority: 1,
                    source_seq: 3,
                },
                kind: EventKind::SettleTradingDay,
            },
        ];

        let mut first = funded_exchange();
        let first_log = first.replay(events.clone()).unwrap();
        let mut second = funded_exchange();
        let second_log = second.replay(events.into_iter().rev().collect()).unwrap();

        assert_eq!(first_log, second_log);
        assert_eq!(first, second);
        assert_eq!(
            first.trades(),
            &[Trade {
                id: 1,
                price: 100,
                quantity: 4,
                maker_order_id: 1,
                taker_order_id: 2,
                buyer_account_id: 3,
                seller_account_id: 2,
            }]
        );
    }

    #[test]
    fn replay_rejects_ambiguous_event_keys() {
        let mut exchange = funded_exchange();
        let key = EventKey {
            sim_time: 10,
            source_priority: 1,
            source_seq: 1,
        };
        let error = exchange
            .replay(vec![
                Event {
                    key,
                    kind: EventKind::SettleTradingDay,
                },
                Event {
                    key,
                    kind: EventKind::SettleTradingDay,
                },
            ])
            .unwrap_err();
        assert_eq!(error, ExchangeError::DuplicateEventKey(key));
    }
}
