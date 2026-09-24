# AIY Voice Kit push-to-talk client

`../scripts/voicekit_s2s_client.py` is an optional Raspberry Pi client for the
local Speech Lab WebSocket. GPIO 23 starts and ends recording; GPIO 25 shows
recording, processing, playback and errors. Local VAD ends an utterance after
silence. Audio is captured and played as 16 kHz, mono, signed 16-bit PCM.

## Requirements

- AIY Voice Kit or equivalent button, LED, microphone and speaker wiring on
  GPIO 23 and 25.
- Python 3.10+, `gpiozero` and `websockets`; `webrtcvad` is optional. Install
  ALSA utilities for `arecord`, `aplay` and `amixer`.
- A Speech Lab WebSocket reachable through **local loopback**. The standalone
  s2s listener is bound to loopback; do not expose port 8765 to a network.
  For a separate Pi, use an authenticated transport such as an SSH tunnel to
  the host loopback endpoint. Do not place credentials in `S2S_WS_URL`. The AuraGo `/speech-lab/` browser route requires
  an AuraGo admin session and is not a credential-free device endpoint.

```bash
python3 -m pip install gpiozero websockets
S2S_WS_URL=ws://127.0.0.1:8765/ws \
S2S_CAPTURE_DEVICE=capture \
S2S_PLAY_DEVICE=playback \
python3 scripts/voicekit_s2s_client.py
```

The client writes microphone capture errors to `/tmp/voicekit-arecord.err`.
`S2S_MIC_GAIN`, `S2S_END_SILENCE_MS`, `S2S_NO_SPEECH_TIMEOUT_MS` and
`S2S_PROCESS_TIMEOUT_SEC` tune the local turn behavior. A second button press
ends the current utterance. SIGINT and SIGTERM close the WebSocket and stop
ALSA subprocesses.

Run the hardware-independent state checks with
`python3 -m unittest discover -s tests -p 'test_voicekit_s2s_client.py'`.
These checks do not establish audio or GPIO behavior on a physical Pi.