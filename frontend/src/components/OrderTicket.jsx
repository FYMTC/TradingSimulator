import { useState, useEffect } from 'react';

// Limit-order ticket: price in ticks, quantity in whole lots.  Defaults
// track the touch so a click trades immediately, per A-share habits.
export default function OrderTicket({ bestBid, bestAsk, onOrder, disabled }) {
  const [price, setPrice] = useState('');
  const [lots, setLots] = useState('1');

  useEffect(() => {
    if (price === '' && bestAsk) setPrice(String(bestAsk));
  }, [bestAsk, price]);

  const submit = (side) => {
    const p = parseInt(price, 10);
    const n = parseInt(lots, 10);
    if (!Number.isFinite(p) || !Number.isFinite(n) || n <= 0) return;
    onOrder(side, p, n);
  };

  const adjust = (delta) => {
    const p = parseInt(price === '' ? '0' : price, 10);
    setPrice(String(Math.max(1, p + delta)));
  };

  return (
    <div className="order-ticket">
      <div className="panel-title">委托下单</div>
      <label className="ticket-field">
        <span>价格</span>
        <div className="price-stepper">
          <button type="button" onClick={() => adjust(-1)} disabled={disabled}>
            −
          </button>
          <input
            value={price}
            onChange={(e) => setPrice(e.target.value.replace(/[^\d]/g, ''))}
            inputMode="numeric"
          />
          <button type="button" onClick={() => adjust(1)} disabled={disabled}>
            +
          </button>
        </div>
      </label>
      <div className="touch-buttons">
        <button type="button" onClick={() => bestBid && setPrice(String(bestBid))} disabled={disabled}>
          买一 {bestBid ?? '--'}
        </button>
        <button type="button" onClick={() => bestAsk && setPrice(String(bestAsk))} disabled={disabled}>
          卖一 {bestAsk ?? '--'}
        </button>
      </div>
      <label className="ticket-field">
        <span>数量（手）</span>
        <input
          value={lots}
          onChange={(e) => setLots(e.target.value.replace(/[^\d]/g, ''))}
          inputMode="numeric"
        />
      </label>
      <div className="ticket-actions">
        <button
          type="button"
          className="buy"
          onClick={() => submit('buy')}
          disabled={disabled}
        >
          买入
        </button>
        <button
          type="button"
          className="sell"
          onClick={() => submit('sell')}
          disabled={disabled}
        >
          卖出
        </button>
      </div>
    </div>
  );
}
