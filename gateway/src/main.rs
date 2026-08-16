//! WebSocket gateway and static-asset host for the TradingSimulator game.
//!
//! Each WebSocket connection owns one [`Session`] (its own deterministic
//! seed and virtual clock) inside a single tokio task; the market is
//! never shared across threads, so its replayability contract is
//! intact.  The same server hosts the built front end from
//! `frontend/dist`, so one binary serves the whole game.
//!
//! IO stays isolated from simulation exactly as the design report
//! prescribes: client messages and the 10 Hz clock tick meet in one
//! `select!` loop, snapshots are serialised after the session advances,
//! and a slow client can only ever slow its own session.

mod protocol;
mod session;

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{ConnectInfo, Query, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tower_http::services::ServeDir;

use protocol::{Ack, ClientMsg, ServerMsg};
use session::{Session, TICK_MS};

/// Query parameters on the WebSocket URL.
#[derive(Debug, Default, Deserialize)]
struct WsQuery {
    /// Optional session seed; omitted falls back to the default.
    seed: Option<u64>,
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(8080);
    let static_dir = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("frontend/dist"));

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .fallback_service(ServeDir::new(static_dir));

    let listener = TcpListener_bind(port).await;
    eprintln!(
        "gateway listening on http://127.0.0.1:{port}, static assets from frontend/dist"
    );
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .expect("gateway server error");
}

async fn TcpListener_bind(port: u16) -> tokio::net::TcpListener {
    tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("bind gateway port")
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    Query(query): Query<WsQuery>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| async move {
        eprintln!("session opened: {addr}");
        run_session(socket, query.seed).await;
        eprintln!("session closed: {addr}");
    })
}

/// Drives one session: the 10 Hz tick clock and client messages meet in
/// a single loop, so the market is only ever touched in task order.
async fn run_session(socket: WebSocket, seed: Option<u64>) {
    let (mut sender, mut receiver) = socket.split();
    let mut session = Session::new(seed);

    // Opening snapshot so the UI can render before the first tick.
    let hello = session.snapshot();
    if send(&mut sender, &hello).await.is_err() {
        return;
    }

    let mut tick = tokio::time::interval(std::time::Duration::from_millis(TICK_MS));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let mut client_seq: u64 = 0;
    loop {
        tokio::select! {
            _ = tick.tick() => {
                session.advance();
                let snapshot = session.snapshot();
                if send(&mut sender, &snapshot).await.is_err() {
                    break;
                }
            }
            incoming = receiver.next() => {
                let Some(Ok(message)) = incoming else { break };
                let Message::Text(text) = message else { continue };
                match serde_json::from_str::<ClientMsg>(&text) {
                    Ok(parsed) => {
                        client_seq += 1;
                        if let Some(reply) = session.apply(parsed, client_seq)
                            && send(&mut sender, &reply).await.is_err()
                        {
                            break;
                        }
                    }
                    Err(_) => {
                        let nak = ServerMsg::Ack(Ack {
                            seq: client_seq,
                            ok: false,
                            order_id: None,
                            fills: Vec::new(),
                            error: Some(format!("unparseable message: {text}")),
                        });
                        if send(&mut sender, &nak).await.is_err() {
                            break;
                        }
                    }
                }
            }
        }
    }
    let _ = sender.close().await;
}

async fn send(
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    msg: &ServerMsg,
) -> Result<(), axum::Error> {
    sender
        .send(Message::Text(serde_json::to_string(msg).unwrap().into()))
        .await
}
