//! OHLCV bar aggregation from the trade tape.
//!
//! Bars are derived exclusively from executed prints, never from quoted
//! prices: the K-line is an emergent consequence of matching, which is the
//! founding principle of the whole simulator.  Empty intervals produce no bar,
//! mirroring how real market data feeds elide silent seconds.

use crate::{Price, Quantity, SimTime};

/// One completed OHLCV bar over `[start_ms, start_ms + width_ms)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bar {
    /// Inclusive start of the bar interval in simulation milliseconds.
    pub start_ms: SimTime,
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: Quantity,
}

/// Incremental aggregator that turns a stream of prints into fixed-width bars.
#[derive(Clone, Debug)]
pub struct BarAggregator {
    width_ms: SimTime,
    current: Option<Bar>,
}

impl BarAggregator {
    /// Creates an aggregator whose bars each span `width_ms` of sim time.
    pub fn new(width_ms: SimTime) -> Self {
        assert!(width_ms > 0, "bar width must be positive");
        Self { width_ms, current: None }
    }

    /// Ingests one print; returns the bar that just closed, if any.
    pub fn push(&mut self, time_ms: SimTime, price: Price, quantity: Quantity) -> Option<Bar> {
        let index = time_ms / self.width_ms;
        let closed = match self.current {
            Some(bar) if bar.start_ms / self.width_ms == index => {
                let bar = &mut self.current.as_mut().expect("checked Some above");
                bar.high = bar.high.max(price);
                bar.low = bar.low.min(price);
                bar.close = price;
                bar.volume += quantity;
                None
            }
            Some(bar) => Some(bar),
            None => None,
        };
        if closed.is_some() || self.current.is_none() {
            self.current = Some(Bar {
                start_ms: index * self.width_ms,
                open: price,
                high: price,
                low: price,
                close: price,
                volume: quantity,
            });
        }
        closed
    }

    /// Flushes the trailing partial bar once the session is over.
    pub fn finish(&mut self) -> Option<Bar> {
        self.current.take()
    }
}

/// Builds the full bar series from a trade tape in one pass.
pub fn aggregate_bars<I, T>(tape: I, width_ms: SimTime) -> Vec<Bar>
where
    I: IntoIterator<Item = T>,
    T: Into<TapePrint>,
{
    let mut aggregator = BarAggregator::new(width_ms);
    let mut bars = Vec::new();
    for print in tape {
        let TapePrint {
            time_ms,
            price,
            quantity,
        } = print.into();
        if let Some(bar) = aggregator.push(time_ms, price, quantity) {
            bars.push(bar);
        }
    }
    if let Some(bar) = aggregator.finish() {
        bars.push(bar);
    }
    bars
}

/// A timed print on the trade tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TapePrint {
    pub time_ms: SimTime,
    pub price: Price,
    pub quantity: Quantity,
}

impl TapePrint {
    /// Creates a tape print.
    pub const fn new(time_ms: SimTime, price: Price, quantity: Quantity) -> Self {
        Self {
            time_ms,
            price,
            quantity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bars_roll_on_width_boundaries_with_correct_ohlcv() {
        let mut agg = BarAggregator::new(1_000);
        assert_eq!(agg.push(100, 10, 1), None);
        assert_eq!(agg.push(500, 12, 2), None);
        assert_eq!(agg.push(900, 11, 3), None);

        let closed = agg.push(1_200, 13, 4).expect("first bar closes");
        assert_eq!(
            closed,
            Bar {
                start_ms: 0,
                open: 10,
                high: 12,
                low: 10,
                close: 11,
                volume: 6,
            }
        );

        // 1900 ms is still inside the second bar's interval.
        assert_eq!(agg.push(1_900, 9, 1), None);

        let second = agg.finish().expect("second bar closes at finish");
        assert_eq!(
            second,
            Bar {
                start_ms: 1_000,
                open: 13,
                high: 13,
                low: 9,
                close: 9,
                volume: 5,
            }
        );
        assert_eq!(agg.finish(), None);
    }

    #[test]
    fn silent_intervals_are_skipped_not_filled() {
        let bars = aggregate_bars(
            [
                TapePrint::new(0, 10, 1),
                TapePrint::new(500, 11, 1),
                // The whole [1000, 3000) window has no prints.
                TapePrint::new(3_100, 12, 2),
            ],
            1_000,
        );
        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].start_ms, 0);
        assert_eq!(bars[0].close, 11);
        assert_eq!(bars[1].start_ms, 3_000);
        assert_eq!(bars[1].open, 12);
    }
}
