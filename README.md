# arabic-channel-sim

Render clean 16 kHz speech under fixed, documented channel conditions: a telephone band-limit, 4 kHz
resampling, G.711 mu-law, Opus at 12 kbit/s and synthetic room reverberation. Every render is aligned to its
parent, level-matched, checked and measured (ESTOI, SI-SDR, spectral roll-off), and is a deterministic
function of the parent audio, the condition id and the parent id.

It was written to test Arabic speech recognition robustness and is used to build the dataset
[Bayanat-Labs/arabic-asr-channel-pairs](https://huggingface.co/datasets/Bayanat-Labs/arabic-asr-channel-pairs)
(FLEURS Egyptian Arabic recordings, clean and under each condition). The renderer itself works on any
16 kHz mono speech.

The conditions are simulations. They do not reproduce network jitter, packet loss, noise, echo cancellation
or handset acoustics, and renders should not be described as real telephone or call-centre audio.

## Install

```bash
pip install "git+https://github.com/Bayanat-Labs/arabic-channel-sim@v0.1.0"
# with ESTOI (pystoi) and the impulse-response generator (pyroomacoustics):
pip install "arabic-channel-sim[metrics,rir] @ git+https://github.com/Bayanat-Labs/arabic-channel-sim@v0.1.0"
```

Once the package is published on PyPI, `pip install arabic-channel-sim` will also work. Python 3.9 or later.
The codec conditions (`CD1`, `CD6`) call an `ffmpeg` binary on your `PATH` that has the `pcm_mulaw` and
`libopus` encoders (`ffmpeg -hide_banner -encoders | grep -E "mulaw|libopus"`).

## Command line

```bash
arabic-channel-sim list
arabic-channel-sim render --condition CD1 --input clean.wav --output clean__CD1.flac
arabic-channel-sim render --condition all --input clean.wav --output renders/
arabic-channel-sim measure clean.wav clean__CD1.flac
```

`render` prints one JSON record per output with the seed, the delay removed, the level-matching gain,
whether the clipping guard lowered it, the share of clipped samples and, for reverberation, the impulse
response used. `--parent-id` sets the id used for the seed (default: the input file name).

## Python

```python
import soundfile as sf
import arabic_channel_sim as acs
from arabic_channel_sim.dsp import quantise16

x, fs = sf.read("clean.wav")          # 16 kHz mono
x = quantise16(x)                     # the renderer works on the 16-bit grid
y16, record = acs.render(x, "RV2", "clean.wav")
sf.write("clean__RV2.flac", y16, 16000, subtype="PCM_16")

print(record["rir_id"], record["gain_db"])
print(acs.estoi(x, y16 / 32768), acs.si_sdr(x, y16 / 32768))   # estoi needs the [metrics] extra
```

## Conditions (`conditions_v1`)

| Id | Condition | Steps |
|---|---|---|
| `BL1` | Telephone band | resample to 8 kHz; band-pass 300-3,400 Hz (linear-phase FIR, 511 taps, Kaiser beta 8); resample to 16 kHz |
| `RS2` | 4 kHz resampling | resample to 4 kHz; resample to 16 kHz |
| `CD1` | G.711 mu-law | as `BL1` to 8 kHz; ffmpeg `pcm_mulaw` encode and decode; resample to 16 kHz |
| `CD6` | Opus 12 kbit/s | ffmpeg `libopus` at 16 kHz, 12 kbit/s, VBR, VoIP mode, 20 ms frames, complexity 10; decode with libopus |
| `RV2` | Room reverberation | convolve with one of the 20 `rirs_v1` rooms at RT60 0.6 s, chosen by seed; align on the direct path |

The file is [`src/arabic_channel_sim/data/conditions_v1.yaml`](src/arabic_channel_sim/data/conditions_v1.yaml);
the renderer executes its steps as written. Resampling uses `scipy.signal.resample_poly` (Kaiser window,
beta 5) and the band-pass filter is applied centred, so neither adds delay.

After the steps, every variant is:

1. **aligned**: shifted by the lag of the cross-correlation peak with the parent (search ±100 ms); `RV2` is
   aligned on the impulse response's direct path instead;
2. **cut or zero-padded** to the parent's exact length (for `RV2` the reverberant tail past the end is cut and
   its energy recorded);
3. **level-matched**: its RMS over the parent's active-speech frames (20 ms frames within 30 dB of the
   loudest) is set equal to the parent's;
4. **clip-guarded**: if that would put more than 0.1% of samples at 16-bit full scale (possible when the parent
   is itself clipped), the gain is lowered just enough and the record says `level_capped: true`;
5. **quantised** to 16-bit (round to nearest, no dither).

Per-variant seed: `int(sha256(parent_id + condition_id).hexdigest()[:8], 16)`. It only chooses the room for
`RV2`; the other conditions involve no randomness. Condition ids are permanent: a changed recipe gets a new id.

## Impulse responses (`rirs_v1`)

Sixty synthetic shoebox rooms, twenty at each RT60 target of 0.3, 0.6 and 1.0 s, made with the
pyroomacoustics image-source method (no air absorption, no ray tracing): room size drawn between
3 x 3 x 2.5 m and 10 x 8 x 4 m, source and microphone 1-3 m apart and at least 0.5 m from the walls. Wall
absorption starts from the Sabine estimate and is rescaled until the measured T30 (Schroeder integration,
-5 to -35 dB) is within 5% of the target; a room outside 10% would be skipped. Every room's seed, geometry,
absorption steps, measured RT60 and direct-to-reverberant ratio are in
[`params.jsonl`](src/arabic_channel_sim/data/rirs_v1/params.jsonl), and the WAV files (32-bit float, 16 kHz)
ship with the package. `arabic-channel-sim rirs --out DIR` regenerates the set from the conditions file with identical samples (the WAV files can differ byte for byte, because the WAV header records a timestamp). No
recorded impulse responses are used.

## Rebuilding the dataset

The scripts in [`scripts/`](scripts/) build `Bayanat-Labs/arabic-asr-channel-pairs` from FLEURS:

```bash
python scripts/fetch_fleurs.py --out raw                       # FLEURS ar_eg dev + test at a pinned revision, SHA-256 checked
python scripts/build_pilot.py --fleurs raw --out pilot         # selection, rendering, checks, ESTOI and SI-SDR
python scripts/transcribe.py --pilot pilot --out eval/hyps --model large-v3-turbo
python scripts/transcribe.py --pilot pilot --out eval/hyps --model small
python scripts/score_wer.py --pairs pilot/rows.jsonl --hyps eval/hyps --out eval
python scripts/package_hf.py --pilot pilot --eval eval --fleurs-record raw/fetch.json --out hf
python scripts/make_card.py --root hf
```

They need, in addition to this package: `huggingface_hub`, `pyarrow`, `pystoi`, `matplotlib`,
`faster-whisper==1.2.1` and [arabicwer](https://github.com/Bayanat-Labs/arabicwer)
(`pip install "git+https://github.com/Bayanat-Labs/arabicwer@v0.1.0"`). The FLEURS archives for dev and test
are about 400 MB.

## Tests

```bash
pip install -e ".[test]"
python -m pytest -q
```

The `CD1` and `CD6` tests are skipped when ffmpeg or its encoders are missing.

## Licence

- Code: Apache-2.0 ([LICENSE](LICENSE)).
- `conditions_v1.yaml` and `rirs_v1/`: CC0 1.0, public domain ([LICENSE-CC0](LICENSE-CC0)).
- Documentation: CC BY 4.0 ([LICENSE-docs](LICENSE-docs)).

See [NOTICE](NOTICE) for dependencies. The dataset built with these scripts has its own licence terms
(audio CC BY 4.0, text CC BY-SA 4.0); see its card.

## Cite

Bayanat Labs (2026). arabic-channel-sim, version 0.1.0. https://github.com/Bayanat-Labs/arabic-channel-sim.
Apache-2.0. A [CITATION.cff](CITATION.cff) file is included.

## Contact

Questions, corrections and issues: hello@bayanatlabs.com or the repository's issue tracker.
