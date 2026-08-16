// Ten-level depth ladder with quantity heat-mapping, asks inverted on
// top so the touch sits in the middle next to the last price.
export default function DepthLadder({ bids, asks, lastTrade }) {
  const render = (levels, side) => {
    const maxQty = Math.max(1, ...levels.map((level) => level.quantity));
    return levels.map((level, i) => (
      <div className={`depth-row ${side}`} key={`${side}-${level.price}-${i}`}>
        <span className="depth-price">{level.price}</span>
        <span className="depth-qty">{level.quantity.toLocaleString()}</span>
        <div
          className={`depth-heat ${side}`}
          style={{ width: `${(level.quantity / maxQty) * 100}%` }}
        />
      </div>
    ));
  };

  return (
    <div className="depth-ladder">
      <div className="panel-title">盘口十档</div>
      {render([...asks].reverse(), 'ask')}
      <div className="depth-touch">
        <span className="touch-label">最新</span>
        <span className="touch-price">{lastTrade ?? '--'}</span>
      </div>
      {render(bids, 'bid')}
    </div>
  );
}
