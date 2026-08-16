import { useState } from 'react';
import { useSession } from './useSession';
import CandleChart from './components/CandleChart';
import DepthLadder from './components/DepthLadder';
import OrderTicket from './components/OrderTicket';
import PlayerPanel from './components/PlayerPanel';
import TapeFeed from './components/TapeFeed';
import ControlBar from './components/ControlBar';

// Ack copy: distinguish immediate fills from a resting order.
function describeAck(event) {
  if (!event.ok) return `被拒：${event.error}`;
  const parts = [];
  if (event.fills.length > 0) {
    parts.push(
      `成交 ${event.fills.map((fill) => `${fill.price}×${fill.quantity}`).join('，')}`,
    );
  }
  if (event.order_id) parts.push(`挂单 #${event.order_id} 等待中`);
  return parts.join('；') || '已提交';
}

export default function App() {
  const { snapshot, events, connected, order, cancel, setSpeed } = useSession();
  const [speed, setSpeedState] = useState(1);

  const applySpeed = (value) => {
    setSpeedState(value);
    setSpeed(value);
  };

  return (
    <div className="terminal">
      <header className="terminal-header">
        <div className="symbol-block">
          <span className="symbol">600000.SH</span>
          <span className={`mark ${snapshot?.last_trade ? '' : 'muted'}`}>
            {snapshot?.mark ?? '--'}
          </span>
          {snapshot?.best_bid && snapshot?.best_ask && (
            <span className="spread muted">
              价差 {snapshot.best_ask - snapshot.best_bid}
            </span>
          )}
        </div>
        <ControlBar
          connected={connected}
          regime={snapshot?.regime}
          manipPhase={snapshot?.manip_phase}
          speed={speed}
          onSpeed={applySpeed}
        />
      </header>

      <main className="terminal-grid">
        <section className="col-chart">
          <CandleChart bars={snapshot?.bars} limit={snapshot?.limit} />
        </section>
        <aside className="col-side">
          <DepthLadder
            bids={snapshot?.bids ?? []}
            asks={snapshot?.asks ?? []}
            lastTrade={snapshot?.last_trade}
          />
          <OrderTicket
            bestBid={snapshot?.best_bid}
            bestAsk={snapshot?.best_ask}
            onOrder={order}
            disabled={!connected}
          />
        </aside>
        <section className="col-bottom">
          <TapeFeed tape={snapshot?.tape} />
          <PlayerPanel player={snapshot?.player} onCancel={cancel} />
          <div className="event-feed">
            <div className="panel-title">回执</div>
            {events.length === 0 ? (
              <div className="muted small">下单后的成交通知会出现在这里</div>
            ) : (
              events
                .slice()
                .reverse()
                .map((event) => (
                  <div key={event.seq} className={`event ${event.ok ? 'ok' : 'reject'}`}>
                    #{event.seq} {describeAck(event)}
                  </div>
                ))
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
