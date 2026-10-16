# Changelog

## 0.1.0 (2026-10-15)

- Initial release.
- Conditions file `conditions_v1` with five conditions: BL1 (telephone band 300-3,400 Hz at 8 kHz),
  RS2 (4 kHz resampling), CD1 (G.711 mu-law at 8 kHz), CD6 (Opus 12 kbit/s, VoIP mode) and RV2
  (synthetic room reverberation, RT60 0.6 s).
- `rirs_v1`: 60 synthetic shoebox-room impulse responses (20 each at RT60 0.3, 0.6 and 1.0 s) with their
  generation parameters, dedicated to the public domain (CC0 1.0).
- Renderer with delay compensation, active-speech level matching and a clipping guard; ESTOI and SI-SDR
  measurements; command line `arabic-channel-sim`.
- Scripts that build the `Bayanat-Labs/arabic-asr-channel-pairs` dataset from FLEURS Egyptian Arabic
  and score Whisper models on it with arabicwer.
