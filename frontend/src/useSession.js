import { useEffect, useRef, useState, useCallback } from 'react';

// One WebSocket session against the Rust gateway.  The server is the
// single source of truth: it broadcasts full snapshots at 10 Hz and
// acknowledges each request in order.  We only keep the latest snapshot
// plus a small event feed (acks/fills/rejects) for the UI.
export function useSession() {
  const [snapshot, setSnapshot] = useState(null);
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef(null);

  useEffect(() => {
    const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
    const socket = new WebSocket(url);
    socketRef.current = socket;

    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onerror = () => setConnected(false);
    socket.onmessage = (frame) => {
      const msg = JSON.parse(frame.data);
      if (msg.type === 'snapshot') {
        setSnapshot(msg);
      } else if (msg.type === 'ack') {
        setEvents((prev) => [...prev.slice(-8), msg]);
      }
    };

    return () => socket.close();
  }, []);

  const send = useCallback((msg) => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(msg));
    }
  }, []);

  const order = useCallback(
    (side, price, lots) => send({ type: 'order', side, price, lots }),
    [send],
  );
  const cancel = useCallback((order_id) => send({ type: 'cancel', order_id }), [send]);
  const setSpeed = useCallback((multiplier) => send({ type: 'speed', multiplier }), [send]);

  return { snapshot, events, connected, order, cancel, setSpeed };
}
