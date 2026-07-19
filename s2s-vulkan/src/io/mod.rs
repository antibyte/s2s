//! Network I/O modes: raw WebSocket PCM + minimal Realtime-compatible API.

pub mod realtime;
pub mod websocket;

pub use realtime::run_realtime_server;
pub use websocket::run_websocket_server;
