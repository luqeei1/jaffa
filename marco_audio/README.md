# Marco – audio subsystem (Ahmad)

Marco's ears and voice: a ReSpeaker XVF3800 4-mic USB array used for sound
direction finding, "Marco"/"Polo" keyword spotting and voice playback.
This folder will become the audio part of the group GitHub repo.

## Folder layout

```
marco_audio/
├── README.md
├── requirements.txt
├── requirements-sim.txt  extra package for sim/ only
├── tools/                  stand-alone scripts, no ROS needed
│   ├── mic_array.py        shared constants: sample rate, channel map, mic positions
│   ├── check_device.py     finds the board, checks 6 ch @ 16 kHz can be opened
│   ├── record.py           records all 6 channels to .npy + .wav
│   ├── analyse_clap.py     measures mic-to-mic delays and a direction from a clap
│   ├── simulate_clap.py    makes a fake clap from a known angle, to test the analysis
│   ├── chip_doa.py         reads the chip's own direction estimates over USB
│   ├── measure_directions.py  guided test: board turned to 8 angles, N claps each
│   ├── analyse_directions.py  error per method and direction -> results.csv, plot
│   ├── make_angle_guide.py rebuilds angle_guide.pdf
│   ├── doa_phat.py         GCC-PHAT, robust least-squares fit, SRP-PHAT, SoundEvent fields
│   └── test_doa_phat.py    self-tests for doa_phat.py (ideal delays, no hardware)
├── sim/                    room simulation (pyroomacoustics) to compare the methods
│   ├── benchmark.py        all methods x 4 reverberation times x 24 directions x 1/2 m
│   ├── tune.py             analysis window length and PHAT strength
│   ├── distance.py         error at 1, 2 and 4 m
│   ├── calibration.py      does confidence / azimuth_std match the real errors?
│   ├── explain_figure.py   "why PHAT works" figure for one simulated clap
│   ├── plot_benchmark.py   error vs reverberation figure
│   ├── plot_style.py       shared figure style, one colour per method
│   └── out/                results (.csv, .txt) and figures (.png)
├── docs/references.bib     checked references (BibTeX) for the reports
├── angle_guide.pdf         print at 100%: put the board on it for the direction test
└── recordings/             saved audio and plots (sim_clap_60_analysis.png = expected result)
```

## 1. Python setup (Windows)

Install Python 3.10, 3.11 or 3.12 from python.org (tick "Add python.exe to PATH").
The robot will run ROS 2 Humble, which uses Python 3.10, so the code avoids
anything newer than 3.10 supports.

In PowerShell, inside `marco_audio`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # prompt now starts with (.venv)
pip install -r requirements.txt
```

If PowerShell refuses to run the activate script:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

## 2. Check what firmware the board has

```powershell
cd tools
python check_device.py
```

If a ReSpeaker line shows **6** in the `in ch` column and `OK`, it already has the
6-channel firmware - skip to step 4. If it shows **2**, flash it (step 3).

## 3. Flash the 16 kHz 6-channel firmware

1. Download the firmware repo
   <https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY> with
   **Code → Download ZIP** (never right-click "save as" on a .bin: it saves a web
   page, not the file). Find `respeaker_xvf3800_usb_dfu_firmware_v2.1.1_16k6ch.bin`.
2. Download dfu-util 0.11 (`dfu-util-0.11-binaries.tar.xz` from
   <https://dfu-util.sourceforge.net/>), extract it, and add the `win64` folder to `Path`.
   Check with `dfu-util -V`.
3. Plug the board in using the USB-C port next to the 3.5 mm jack, then run `dfu-util -l`.
   If it says `LIBUSB_ERROR_NOT_SUPPORTED`, Windows needs the WinUSB driver on the
   board's DFU interface:
   - Open Zadig → Options → List All Devices → pick the reSpeaker entry.
   - If there are several "(Interface N)" entries, pick the DFU/control one. **Do not**
     replace a driver shown as `usbaudio2` (that's the microphone itself) or
     `usbccgp` (the parent of the whole device).
   - Choose WinUSB → Install Driver → unplug and replug → `dfu-util -l` again.
     You should see two lines: `reSpeaker DFU Upgrade` (alt=1) and `reSpeaker DFU Factory` (alt=0).
4. Flash:
   ```powershell
   dfu-util -R -e -a 1 -D "C:\path\to\respeaker_xvf3800_usb_dfu_firmware_v2.1.1_16k6ch.bin"
   ```
   `-a 1` writes to the Upgrade slot (the Factory slot is the safe backup);
   `-R` resets the board afterwards.
5. Run `python check_device.py` again: it should now show 6 input channels.

Easier alternative: the reSpeaker Console app
(<https://github.com/respeaker/respeaker-console/releases>) bundles dfu-util;
it still needs the Zadig step.

**If something goes wrong:** unplug, hold the Mute button, plug back in (red LED
blinks = Safe Mode), then flash again.

## 4. Record and analyse a clap

Windows settings first (Sound settings → More sound settings → Recording →
reSpeaker → Properties):
- Advanced tab: default format should read **6 channel, 16000 Hz**; tick
  "Allow applications to take exclusive control".
- Turn off any "Audio enhancements".

Lay the board flat, Seeed logo **up** (the mic holes are on that side).

```powershell
python record.py --device <index> --name clap_front
python analyse_clap.py ..\recordings\clap_front.npy
```

Try `--exclusive` on `record.py` if shared mode fails.
Sanity check before trusting real data:

```powershell
python simulate_clap.py --angle 60
python analyse_clap.py ..\recordings\sim_clap_60.npy --true-angle 60
```

## 5. Tap test: which corner is which mic?

The chip reports its own mic coordinates, but nothing says which physical
sound hole is mic 0, or which way round the frame runs. On the cased board the
mic holes are the four small slots near the edge. Snap your fingers 2–3 cm above
one slot (don't touch the board or desk: vibration travels through the circuit
board far faster than through air and reaches all four mics almost at once),
record, analyse, and note "First to hear the clap". Repeat for each slot.

**Result (24 Oct 2026).** Board flat, logo up, seen from above, USB on the right:

```
        mic3 ●───────● mic0          90°
             │       │                ↑
             │       │ ▐ USB    180° ←   → 0°
             │       │                ↓
        mic2 ●───────● mic1          270°
```
(the USB socket sits halfway between the mic0 and mic1 slots)

- **0° points out of the USB edge**, between mic0 and mic1.
- Angles increase **anticlockwise** seen from above (logo side), so 90° is the
  mic0/mic3 side.
- The chip's own coordinate list (`AEC_MIC_ARRAY_GEO`) is **mirrored** relative
  to this view, so `mic_array.py` flips the sign of y. Anything read from the
  chip itself (e.g. its own DOA) needs converting before comparing.
- Evidence: snap above each slot → mic0, mic1, mic2 as labelled (each with the
  diagonal mic last by ≈250 µs); clap opposite the USB → ≈180°; clap from the
  mic0/mic3 side → ≈90°.
- This matches the ROS convention (REP 103: x forward, y left, z up). Mount the
  board logo up with the USB edge facing the robot's front and the angles are
  already in the robot's frame.

## 6. Direction accuracy test (8 directions)

Instead of walking round the board, **turn the board** and clap from one fixed
spot: what matters is the sound's direction relative to the board.

1. Print `angle_guide.pdf` at 100% (check the scale bar is 10 cm). No printer:
   fold a square sheet in half, in half again, then along both diagonals; the
   creases give 8 lines at 45°. Label them as on the PDF (the line pointing at
   you is 0°, then 45°, 90°… going **clockwise**).
2. Centre the board on it, logo up. Sit or stand at the "YOU" edge, about 1 m
   away. Measure the distance once.
3. Check the chip can be read (optional):
   `python chip_doa.py` (Ctrl+C to stop)
4. Dry run without hardware: `python measure_directions.py --simulate`
5. Real run:
   ```powershell
   python measure_directions.py --device 15 --distance 1.0
   python analyse_directions.py ..\recordings\<session folder it prints>
   ```
   For each angle: turn the board, press Enter, clap 3 times about a second
   apart, at board height, same spot every time. It checks it heard 3 claps.

Output in the session folder: `results.csv` (one row per clap),
`summary.txt` and `direction_test.png`.

## 7. Direction finding with GCC-PHAT and SRP-PHAT (`doa_phat.py`)

`analyse_clap.py` finds each mic pair's delay with plain cross-correlation. That works for
click-like sounds but fails on tonal ones (a ringing clap, a vowel), where the correlation
peak is broad and echoes push it sideways. `doa_phat.py` adds three alternatives:

- **GCC-PHAT** [1]: the same correlation computed in the frequency domain after dividing
  each frequency by its own loudness (phase transform), so every frequency between 300 Hz
  and 6 kHz gets an equal say and the peak stays sharp. Delays are refined to 1/16 sample.
  A least-squares fit turns the 6 pair delays into a direction; `|u|` = cos(elevation).
- **GCC-PHAT robust**: the same fit, but it leaves out a dead or covered mic (weak
  correlation peaks) or a single pair whose delay is a clear outlier.
- **SRP-PHAT** [2]: tries every direction (1° azimuth x 5° elevation) and adds up the six
  GCC-PHAT curves at the delays each direction predicts; the best total wins.

Each method also returns the `SoundEvent` fields `azimuth_std_rad` (precision, from a
per-mic timing-noise model) and `confidence` (0-1: do the delays agree, is `|u|` possible,
are the peaks strong). Check it works:

```powershell
python tools\test_doa_phat.py        # should end with ALL PASSED
```

Use it on a recording (raw mics are channels 2-5; a window from 5 ms before to 30 ms after
the onset works best, because echoes arrive after the direct sound):

```python
import numpy as np, doa_phat as dp
seg = x[onset - 80 : onset + 480, 2:6]
r = dp.localise(seg)
print(np.degrees(r["gcc_phat"]["azimuth_rad"]), r["gcc_phat"]["confidence"])
print(np.degrees(r["srp_phat"]["azimuth_rad"]), r["srp_phat"]["confidence"])
```

## 8. Room simulation (`sim/`)

Simulated rooms show how each method copes with echoes before testing in real rooms.
`pyroomacoustics` [3] models each wall reflection as a mirror-image copy of the source [4].

```powershell
pip install -r requirements-sim.txt   # pyroomacoustics, only needed for sim/
python sim\benchmark.py --quick       # about a minute; drop --quick for the full run
python sim\plot_benchmark.py          # -> sim\out\sim_error_vs_rt60.png
```

Result (28 Oct 2026, 768 simulated events per method): plain cross-correlation reached 50°
error on ringing claps in echoey rooms; GCC-PHAT and SRP-PHAT never exceeded 5.1°. On
click-like sounds all methods are within a few degrees. The simulation leaves out mic
mismatch, the board's case and table reflections, so real recordings decide.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Only 2 channels after flashing | Device Manager → uninstall every reSpeaker entry → unplug/replug so Windows reloads the driver |
| Raw channels (ch2–ch5) silent | Same as above (known Windows issue with the 6-ch firmware) |
| No sound device after Zadig | Zadig replaced the audio driver: uninstall in Device Manager, replug |
| `CLIPPED` in record.py | Clap further away, or more softly |
| Angle wrong but `|u|` ≈ 1 | Mic order/orientation wrong: do the tap test |
| `|u|` far from 1, delays marked "impossible" | Bad recording (echo, clipping, wrong channels) |

## Sources

[1] C. H. Knapp and G. C. Carter, "The generalized correlation method for estimation of
time delay," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 24, no. 4, pp. 320–327, 1976.
[2] J. H. DiBiase, H. F. Silverman and M. S. Brandstein, "Robust localization in reverberant
rooms," in *Microphone Arrays*, M. Brandstein and D. Ward, Eds. Springer, 2001, pp. 157–180.
[3] R. Scheibler, E. Bezzam and I. Dokmanić, "Pyroomacoustics: A Python package for audio room
simulation and array processing algorithms," in *Proc. IEEE ICASSP*, 2018, pp. 351–355.
[4] J. B. Allen and D. A. Berkley, "Image method for efficiently simulating small-room
acoustics," *J. Acoust. Soc. Am.*, vol. 65, no. 4, pp. 943–950, 1979.
Full list with DOIs: `docs/references.bib`.

### Hardware and firmware sources

- Seeed Studio wiki, *Getting Started with ReSpeaker XVF3800*:
  <https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/> (firmware files,
  flashing, Safe Mode, mic geometry, Windows troubleshooting)
- Seeed, `python_control/xvf_host.py` in
  <https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY> (USB control
  protocol and command ids used by `chip_doa.py`); `xmos_firmwares/usb/readme.md`
  (firmware changelog and 6-channel layout)
