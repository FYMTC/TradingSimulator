//! Shared discrete-event market core used by every simulation layer.
//!
//! M1 (noise), M2 (RL investors) and M3 (heterogeneous agents + regime
//! engine) differ only in what an agent does when it wakes; the machinery
//! around it - the virtual clock, the strictly ordered event queue, the
//! canonical replay log, trade-tape and spread recording - is identical.
//! This module owns that machinery so each layer stays a thin population
//! definition, and the determinism contract (single shared RNG, unique
//! event keys, bit-exact replay) is implemented exactly once.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::bar::TapePrint;
use crate::rng::Rng;
use crate::{
    Account, AccountId, Event, EventKey, EventKind, Exchange, LimitOrderRequest, Money, OrderId,
    Price, Quantity, SimTime,
};

/// Queue kind reserved for the internal T+1 day-settlement boundary;
/// every market numbers its own agent kinds starting from 1.
pub(crate) const KIND_SETTLE: u8 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct QueueEntry {
    pub time_ms: SimTime,
    /// Global tie-breaker guaranteeing a total order over simultaneous
    /// events; this is what makes every run deterministic.
    pub seq: u64,
    pub kind: u8,
    pub index: usize,
}

/// The shared simulation state every market drives.
#[derive(Clone, Debug)]
pub(crate) struct MarketCore {
    pub(crate) symbol: String,
    pub(crate) ref_price: Price,
    pub(crate) day_length_ms: SimTime,
    pub(crate) exchange: Exchange,
    pub(crate) rng: Rng,
    queue: BinaryHeap<Reverse<QueueEntry>>,
    next_seq: u64,
    /// Event-log sequence, allocated at log time so one wake may emit
    /// several uniquely keyed events (e.g. cancels + a submit).
    next_event_seq: u64,
    now_ms: SimTime,
    last_trade_price: Option<Price>,
    tape: Vec<TapePrint>,
    /// Post-trade spread samples in ticks, for the acceptance harnesses.
    spread_samples_ticks: Vec<i64>,
    /// Canonical event log; replaying it rebuilds the identical exchange.
    replay_log: Vec<Event>,
    rejected_submits: usize,
}

impl MarketCore {
    pub(crate) fn new(symbol: String, ref_price: Price, day_length_ms: SimTime, seed: u64) -> Self {
        let exchange = Exchange::new(symbol.clone());
        Self {
            symbol,
            ref_price,
            day_length_ms,
            exchange,
            rng: Rng::seed_from_u64(seed),
            queue: BinaryHeap::new(),
            next_seq: 0,
            next_event_seq: 0,
            now_ms: 0,
            last_trade_price: None,
            tape: Vec::new(),
            spread_samples_ticks: Vec::new(),
            replay_log: Vec::new(),
            rejected_submits: 0,
        }
    }

    /// Funds an account with cash and a settled, sellable seed position.
    pub(crate) fn add_funded_account(
        &mut self,
        account_id: AccountId,
        cash: Money,
        seed_shares: Quantity,
    ) {
        let mut account = Account::with_cash(cash);
        account.seed_settled_position(&self.symbol, seed_shares);
        self.exchange
            .add_account(account_id, account)
            .expect("account ids are unique");
    }

    pub(crate) fn now_ms(&self) -> SimTime {
        self.now_ms
    }

    pub(crate) fn last_trade_price(&self) -> Option<Price> {
        self.last_trade_price
    }

    pub(crate) fn tape(&self) -> &[TapePrint] {
        &self.tape
    }

    pub(crate) fn spread_samples_ticks(&self) -> &[i64] {
        &self.spread_samples_ticks
    }

    pub(crate) fn replay_log(&self) -> &[Event] {
        &self.replay_log
    }

    pub(crate) fn rejected_submits(&self) -> usize {
        self.rejected_submits
    }

    pub(crate) fn schedule(&mut self, time_ms: SimTime, kind: u8, index: usize) {
        let seq = self.next_seq;
        self.next_seq = seq.checked_add(1).expect("sequence overflow");
        self.queue.push(Reverse(QueueEntry {
            time_ms,
            seq,
            kind,
            index,
        }));
    }

    pub(crate) fn log_event(&mut self, time_ms: SimTime, kind: EventKind) {
        let seq = self.next_event_seq;
        self.next_event_seq = seq.checked_add(1).expect("event sequence overflow");
        self.replay_log.push(Event {
            key: EventKey {
                sim_time: time_ms,
                source_priority: 1,
                source_seq: seq,
            },
            kind,
        });
    }

    /// Cancels one resting order and logs the cancellation.
    pub(crate) fn cancel_tracked(
        &mut self,
        account_id: AccountId,
        order_id: OrderId,
        now_ms: SimTime,
    ) {
        let _ = self.exchange.cancel_order(account_id, order_id);
        self.log_event(
            now_ms,
            EventKind::Cancel {
                account_id,
                order_id,
            },
        );
    }

    /// Logs, submits, and records tape/spread side effects; returns the
    /// order id when any quantity rests.
    pub(crate) fn submit_and_track(
        &mut self,
        request: LimitOrderRequest,
        now_ms: SimTime,
    ) -> Option<OrderId> {
        self.log_event(now_ms, EventKind::Submit(request));
        let Ok(result) = self.exchange.submit_limit_order(request) else {
            self.rejected_submits += 1;
            return None;
        };
        for trade in &result.trades {
            self.last_trade_price = Some(trade.price);
            self.tape
                .push(TapePrint::new(now_ms, trade.price, trade.quantity));
        }
        if !result.trades.is_empty()
            && let (Some(bid), Some(ask)) = (
                self.exchange.book().best_bid(),
                self.exchange.book().best_ask(),
            )
        {
            self.spread_samples_ticks.push(ask - bid);
        }
        (result.remaining > 0).then_some(result.order_id)
    }

    /// Mark price for PnL accounting: the mid quote when both sides are
    /// quoted, the available touch otherwise, falling back to the last
    /// trade or the reference price on a cold book.  Marking at the mid
    /// (rather than the last print) keeps an agent's own executions from
    /// moving its mark: crossing the spread shows up as an immediate,
    /// visible cost and filling passively as an immediate, visible gain.
    pub(crate) fn mark_price(&self) -> Price {
        let bid = self.exchange.book().best_bid();
        let ask = self.exchange.book().best_ask();
        match (bid, ask) {
            (Some(bid), Some(ask)) => (bid + ask) / 2,
            (Some(price), None) | (None, Some(price)) => price,
            (None, None) => self.last_trade_price.unwrap_or(self.ref_price),
        }
    }

    /// Pops the next event due at or before `target_ms`, advancing the
    /// virtual clock; `None` means nothing is due.
    pub(crate) fn pop_due(&mut self, target_ms: SimTime) -> Option<QueueEntry> {
        let due = match self.queue.peek() {
            Some(Reverse(entry)) => entry.time_ms <= target_ms,
            None => false,
        };
        if !due {
            return None;
        }
        let Reverse(entry) = self.queue.pop().expect("peeked entry exists");
        self.now_ms = self.now_ms.max(entry.time_ms);
        Some(entry)
    }

    pub(crate) fn advance_to(&mut self, target_ms: SimTime) {
        self.now_ms = self.now_ms.max(target_ms);
    }

    /// Executes the T+1 settlement boundary and reschedules the next one.
    pub(crate) fn handle_settle(&mut self, time_ms: SimTime) {
        self.exchange.settle_trading_day();
        self.log_event(time_ms, EventKind::SettleTradingDay);
        let next = time_ms
            .checked_add(self.day_length_ms)
            .expect("sim time overflow");
        self.schedule(next, KIND_SETTLE, 0);
    }
}

/// A market population that can be woken by the shared event loop.
pub(crate) trait MarketDriver {
    fn core(&mut self) -> &mut MarketCore;
    /// React to one scheduled wake-up.
    fn wake(&mut self, kind: u8, index: usize, now_ms: SimTime);
}

/// Drives the virtual clock forward, dispatching wakes in strict order.
pub(crate) fn run_until<M: MarketDriver>(market: &mut M, target_ms: SimTime) {
    while let Some(entry) = market.core().pop_due(target_ms) {
        if entry.kind == KIND_SETTLE {
            market.core().handle_settle(entry.time_ms);
        } else {
            market.wake(entry.kind, entry.index, entry.time_ms);
        }
    }
    market.core().advance_to(target_ms);
}
