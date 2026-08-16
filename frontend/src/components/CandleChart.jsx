import { useEffect, useRef } from 'react';

// Canvas candlestick + volume chart with the daily price-limit band
// shaded in.  Redraws on every snapshot (10 Hz); incremental drawing is
// unnecessary at this payload size, and a full redraw avoids drift.
export default function CandleChart({ bars, limit }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !bars || bars.length === 0) return;
    const parent = canvas.parentElement;
    const cssWidth = parent.clientWidth;
    const cssHeight = parent.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = cssWidth * dpr;
    canvas.height = cssHeight * dpr;
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const padLeft = 8;
    const padRight = 56;
    const padTop = 10;
    const volHeight = cssHeight * 0.22;
    const priceHeight = cssHeight - volHeight - padTop - 24;
    const plotWidth = cssWidth - padLeft - padRight;

    // Price scale over visible bars and the limit band.
    let lo = Infinity;
    let hi = -Infinity;
    for (const bar of bars) {
      lo = Math.min(lo, bar.low);
      hi = Math.max(hi, bar.high);
    }
    if (limit) {
      lo = Math.min(lo, limit[0]);
      hi = Math.max(hi, limit[1]);
    }
    const pad = Math.max(2, (hi - lo) * 0.05);
    lo -= pad;
    hi += pad;
    const y = (price) => padTop + ((hi - price) / (hi - lo)) * priceHeight;
    const step = plotWidth / bars.length;
    const bodyW = Math.max(1, Math.floor(step * 0.62));

    // Limit band shading: legal trading range for the session.
    if (limit) {
      ctx.fillStyle = 'rgba(120, 140, 180, 0.07)';
      ctx.fillRect(padLeft, y(limit[1]), plotWidth, y(limit[0]) - y(limit[1]));
      ctx.strokeStyle = 'rgba(140, 160, 200, 0.35)';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      for (const edge of limit) {
        ctx.beginPath();
        ctx.moveTo(padLeft, y(edge));
        ctx.lineTo(padLeft + plotWidth, y(edge));
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(150, 165, 200, 0.8)';
      ctx.font = '10px system-ui';
      ctx.fillText(String(limit[1]), padLeft + plotWidth + 4, y(limit[1]) + 3);
      ctx.fillText(String(limit[0]), padLeft + plotWidth + 4, y(limit[0]) + 3);
    }

    // Gridlines and price labels.
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.fillStyle = 'rgba(160, 174, 192, 0.8)';
    ctx.font = '10px system-ui';
    for (let i = 0; i <= 4; i++) {
      const price = lo + ((hi - lo) * i) / 4;
      const yy = y(price);
      ctx.beginPath();
      ctx.moveTo(padLeft, yy);
      ctx.lineTo(padLeft + plotWidth, yy);
      ctx.stroke();
      ctx.fillText(String(Math.round(price)), padLeft + plotWidth + 4, yy + 3);
    }

    // Volume scale.
    let maxVol = 0;
    for (const bar of bars) maxVol = Math.max(maxVol, bar.volume);
    const volTop = padTop + priceHeight + 18;
    const volH = volHeight - 6;

    // Candles + volume bars.
    bars.forEach((bar, i) => {
      const cx = padLeft + i * step + step / 2;
      const up = bar.close >= bar.open;
      const color = up ? '#e0566b' : '#3fb984'; // A-share convention: red up, green down.
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(cx) + 0.5, y(bar.high));
      ctx.lineTo(Math.round(cx) + 0.5, y(bar.low));
      ctx.stroke();
      const top = y(Math.max(bar.open, bar.close));
      const bottom = y(Math.min(bar.open, bar.close));
      ctx.fillRect(cx - bodyW / 2, top, bodyW, Math.max(1, bottom - top));
      const vh = maxVol > 0 ? (bar.volume / maxVol) * volH : 0;
      ctx.globalAlpha = 0.45;
      ctx.fillRect(cx - bodyW / 2, volTop + volH - vh, bodyW, vh);
      ctx.globalAlpha = 1;
    });
  }, [bars, limit]);

  return (
    <div className="chart-frame">
      <canvas ref={canvasRef} />
    </div>
  );
}
