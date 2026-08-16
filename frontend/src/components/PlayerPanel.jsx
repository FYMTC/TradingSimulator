const fmtMoney = (value) => {
  const abs = Math.abs(Number(value ?? 0));
  const sign = Number(value ?? 0) < 0 ? '-' : '';
  if (abs >= 1e8) return `${sign}${(abs / 1e8).toFixed(2)} 亿`;
  if (abs >= 1e4) return `${sign}${(abs / 1e4).toFixed(2)} 万`;
  return `${sign}${abs.toLocaleString()}`;
};

// Player account: cash, T+1 position detail, open orders with cancels.
export default function PlayerPanel({ player, onCancel }) {
  if (!player) {
    return (
      <div className="player-panel">
        <div className="panel-title">我的账户</div>
        <div className="muted">等待会话建立…</div>
      </div>
    );
  }
  const rows = [
    ['可用资金', fmtMoney(player.cash_available)],
    ['冻结资金', fmtMoney(player.cash_reserved)],
    ['总权益', fmtMoney(player.equity)],
    ['可卖持仓', `${player.sellable.toLocaleString()} 股`],
    ['待交收（T+1）', `${player.unsettled_buys.toLocaleString()} 股`],
  ];
  return (
    <div className="player-panel">
      <div className="panel-title">我的账户</div>
      <table className="account-table">
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <td className="muted">{label}</td>
              <td className="value">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="panel-title">挂单</div>
      {player.open_orders.length === 0 ? (
        <div className="muted small">暂无挂单</div>
      ) : (
        <table className="orders-table">
          <tbody>
            {player.open_orders.map((order) => (
              <tr key={order.order_id}>
                <td className={order.side === 'buy' ? 'up' : 'down'}>
                  {order.side === 'buy' ? '买' : '卖'} {order.price}
                </td>
                <td>{order.remaining.toLocaleString()} 股</td>
                <td>
                  <button
                    type="button"
                    className="cancel"
                    onClick={() => onCancel(order.order_id)}
                  >
                    撤单
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
