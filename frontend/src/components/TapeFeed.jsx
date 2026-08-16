// Recent tape prints, newest last, side inferred from tick direction.
export default function TapeFeed({ tape }) {
  const prints = [...(tape ?? [])].reverse().slice(0, 30);
  return (
    <div className="tape-feed">
      <div className="panel-title">逐笔成交</div>
      {prints.length === 0 ? (
        <div className="muted small">暂无成交</div>
      ) : (
        prints.map((print, i) => {
          const prev = prints[i - 1]?.price ?? print.price;
          const cls = print.price > prev ? 'up' : print.price < prev ? 'down' : 'flat';
          return (
            <div className="tape-row" key={`${print.t}-${i}`}>
              <span className="tape-time">
                {String(Math.floor(print.t / 60000) % 60).padStart(2, '0')}:
                {String(Math.floor(print.t / 1000) % 60).padStart(2, '0')}
              </span>
              <span className={cls}>{print.price}</span>
              <span className="muted">{print.quantity.toLocaleString()}</span>
            </div>
          );
        })
      )}
    </div>
  );
}
