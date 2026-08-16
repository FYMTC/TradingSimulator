const SPEEDS = [
  { label: '⏸ 暂停', value: 0 },
  { label: '1×', value: 1 },
  { label: '5×', value: 5 },
  { label: '20×', value: 20 },
];

const REGIME_LABELS = {
  calm: '平静',
  bull: '牛市',
  bear: '熊市',
  crisis: '危机',
};

// The manipulator's arc as the game's narrative layer.
const MANIP_STORIES = {
  accumulate: { text: '疑似主力资金正在低调吸筹…', cls: 'watch' },
  pump: { text: '警示：异常资金连续扫单，股价快速拉升！', cls: 'alert' },
  distribute: { text: '主力高位放量出货，注意接盘风险。', cls: 'warn' },
  done: { text: '操纵周期结束，市场回归平静。', cls: 'quiet' },
};

export default function ControlBar({ connected, regime, manipPhase, speed, onSpeed }) {
  const story = MANIP_STORIES[manipPhase] ?? null;
  return (
    <div className="control-bar">
      <span className={`conn ${connected ? 'on' : 'off'}`}>
        {connected ? '● 已连接' : '○ 连接中…'}
      </span>
      {regime && (
        <span className="chip">
          宏观：<b>{REGIME_LABELS[regime] ?? regime}</b>
        </span>
      )}
      {story && <span className={`story ${story.cls}`}>{story.text}</span>}
      <div className="speed-group">
        {SPEEDS.map((option) => (
          <button
            key={option.value}
            type="button"
            className={speed === option.value ? 'active' : ''}
            onClick={() => onSpeed(option.value)}
            disabled={!connected}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}
