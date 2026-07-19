//! Audio helpers: PCM conversion, WAV encode, resampling, device I/O.

pub mod local;
pub mod pcm;

pub use local::{list_devices, run_local_io};
