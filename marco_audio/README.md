# Marco – audio subsystem (Ahmad)

Marco's ears and voice: a ReSpeaker XVF3800 4-mic USB array used for sound
direction finding, "Marco"/"Polo" keyword spotting and voice playback.
This folder is the audio part of the group repo.

It is two things at once: a set of stand-alone Python scripts for recording and testing
(sections 1 to 9, Windows, no ROS), and a ROS 2 Humble package that runs the same code on the
robot (sections 10 to 12). **To run it on the robot's Jetson, follow
[`docs/JETSON_RUNSHEET.md`](docs/JETSON_RUNSHEET.md).**

## Folder layout

```
marco_audio/
├── README.md
├── requirements.txt
├── requirements-sim.txt  extra package for sim/ only
├── package.xml, setup.py, setup.cfg, resource/   make this folder a ROS 2 package (section 10)
├── marco_audio/            the ROS 2 nodes
│   ├── audio_node.py       Marco's ears: publishes one SoundEvent message per sound
│   └── speech_node.py      Marco's voice: plays a clip when asked
├── launch/
│   └── marco_audio.launch.py  starts both nodes and the board's transform (section 11)
├── clips/                  the voice clips speech_node plays (placeholder beeps for now)
├── test/                   checks for the nodes and the launch file, no ROS needed
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
│   ├── test_doa_phat.py    self-tests for doa_phat.py (ideal delays, no hardware)
│   ├── analyse_phat_session.py  all methods on a measure_directions session -> results_phat.csv
│   ├── live_doa.py         live direction finder: every method + chip, saves each run, --replay
│   ├── keyword_spotter.py  tells the words "marco" and "polo" from every other sound (Vosk)
│   ├── export_clips.py     small clips.npz of a run's sounds, to share or send for checking
│   └── make_placeholder_clips.py  rebuilds the four placeholder clips in clips/
├── sim/                    room simulation (pyroomacoustics) to compare the methods
│   ├── benchmark.py        all methods x 4 reverberation times x 24 directions x 1/2 m
│   ├── tune.py             analysis window length and PHAT strength
│   ├── distance.py         error at 1, 2 and 4 m
│   ├── calibration.py      does confidence / azimuth_std match the real errors?
│   ├── explain_figure.py   "why PHAT works" figure for one simulated clap
│   ├── plot_benchmark.py   error vs reverberation figure
│   ├── plot_style.py       shared figure style, one colour per method
│   └── out/                results (.csv, .txt) and figures (.png)
├── docs/
│   ├── JETSON_RUNSHEET.md  step by step on the robot's Jetson, and the interface for other nodes
│   └── references.bib      checked references (BibTeX) for the reports
├── angle_guide.pdf         print at 100%: put the board on it for the direction test
└── recordings/             saved audio and plots (sim_clap_60_analysis.png = expected result)

../marco_audio_msgs/        the SoundEvent message type, a small ROS 2 package of its own
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
   python measure_directions.py --device <index> --distance 1.0
   python analyse_directions.py ..\recordings\<session folder it prints>
   ```
   For each angle: turn the board, press Enter, clap 3 times about a second
   apart, at board height, same spot every time. It checks it heard 3 claps.
   `<index>` is the WASAPI line marked OK in `check_device.py`. It can change when
   other sound devices are plugged in or out (15 until 5 Nov, then 14).
   Tape the clap spot exactly on the 0° line: a spot a few cm to one side shifts
   every angle the same way (about −3.7° on 28 Oct). `live_doa.py --guide`
   (section 8) runs the same test live.

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

Use it on a recording (raw mics are channels 2-5). Cut a window from 5 ms before to 2 ms
after the onset: echoes arrive after the direct sound, and one from a nearby wall arrives
within about 2 ms (section 8). `analyse_phat_session.py` still defaults to 30 ms, so add
`--post-ms 2` to it.

```python
import numpy as np, doa_phat as dp
seg = x[onset - 80 : onset + 32, 2:6]      # 5 ms before to 2 ms after, at 16 kHz
r = dp.localise(seg)
print(np.degrees(r["gcc_phat"]["azimuth_rad"]), r["gcc_phat"]["confidence"])
print(np.degrees(r["srp_phat"]["azimuth_rad"]), r["srp_phat"]["confidence"])
```

## 8. Live direction finder (`live_doa.py`)

`live_doa.py` listens to the board all the time. For every clap or shout it prints the
direction from each method (GCC-PHAT, GCC-PHAT robust, SRP-PHAT, plain cross-correlation)
next to the chip's two estimates, with each method's confidence. It calls the same functions
as the offline scripts and saves the whole run, so the run can be replayed later without the
board and give identical answers.

```powershell
python check_device.py                    # note the device number: it can change
python live_doa.py --device <index> --guide --distance 1.0 --min-db -60 --note "room, setup"
python live_doa.py --replay ..\recordings\<run folder>                # same answers, no board
python live_doa.py --replay ..\recordings\<run folder> --post-ms 30   # try another window
python export_clips.py ..\recordings\<run folder>                     # small clips.npz to share
```

Guided mode (`--guide`) walks through the 8 angles of section 6: turn the board, press Enter,
wait for "Clap now", clap 3 times. Type `r` to redo an angle, `x` to stop labelling and `q`
to save and quit. `--min-db -60` ignores anything quieter than −60 dBFS, which suits claps at
1 m; leave it out for quieter or more distant sounds.

Each run is saved in `recordings\live_<date>_<time>\`: `audio.npy` (the whole run, about
23 MB a minute, kept out of git by the root `.gitignore`), `events.csv` (one row per sound),
`summary.txt`, `labels.csv`, `chip.csv` and `session.json`.

**Why the window ends 2 ms after the onset (5 Nov 2026).** In the live test
`recordings/live_20261105_2114` (24 claps, 8 directions, 1 m) the board stood about 30 cm in
front of a wall, on the far side from the clapper. The wall's echo arrived about
0.6 m ÷ 343 m/s ≈ 1.75 ms after the direct sound. With the old window (5 ms before to 30 ms
after the onset), GCC-PHAT was more than 10° off on 9 of 24 claps, the worst by 110°. With the
window ending 2 ms after the onset: median 1.0°, worst 4.4°, none over 10°. Across all three
clap sessions (72 claps), windows ending 1.5 to 2 ms after the onset did about equally well,
and 2.5 ms or more got worse next to the wall. On the 24 and 28 Oct sessions the change moved
the GCC-PHAT median by only 0.2°. The run's folder keeps every clap (`clips.npz`) and both
replays (`summary_replay.txt` with the old window, `summary_replay_5-2ms.txt` with the new
one); its full `audio.npy` is too big for GitHub.

## 9. Room simulation (`sim/`)

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

## 10. ROS 2 nodes (`audio_node`, `speech_node`)

ROS 2 is the software the robot's parts use to talk to each other. A **node** is one running
program, a **topic** is a named channel, and a **message** is one item sent on a topic with fixed
fields. This folder is a ROS 2 Humble package with two nodes:

- **`audio_node`** (Marco's ears) listens to the board and publishes one `SoundEvent` message per
  sound on `/marco/audio/sound_event`: the direction (`azimuth`, radians), its uncertainty, a
  confidence from 0 to 1 and the loudness. It imports the `Listener` class from
  `tools/live_doa.py`, so a sound gets the same answer here as in the live clap test (section 8).
  With `source:=replay` it plays a saved run through the same code instead of the microphone.
- **Keyword spotting.** If the Vosk speech recogniser and its model are installed (run sheet,
  section 10), `audio_node` also works out whether the sound was the word "marco" or "polo"
  and puts it in the message's `keyword` field (`type` becomes `keyword`). It hands
  `tools/keyword_spotter.py` the sound from 0.3 s before to 1.0 s after the start of each sound,
  taken from the channel the board's chip cleans up for speech. The recogniser may only answer
  "marco", "polo" or "something else", and a word counts only if its confidence is at least
  0.6 and it starts when the sound starts. Because the whole word has to be heard first, each
  message then goes out about a second after its sound started (the time stamp is still the
  start of the sound). Without Vosk the node works as before and `keyword` stays empty;
  `-p keywords:=false` switches it off. Not measured yet: how many real "Polo!"s it catches
  and how often ordinary talk is mistaken for one.
- **`speech_node`** (Marco's voice) plays `clips/<name>.wav` through the board when a name arrives
  on `/marco/speech/say`, and publishes `true` on `/marco/speech/active` while it plays.
  `audio_node` ignores sounds that start while that is true, so Marco does not report his own
  voice. The four clips are placeholder beeps until the real voice is recorded.
- The message type is a package of its own, `../marco_audio_msgs` (`msg/SoundEvent.msg`).

Build it on Ubuntu 22.04 with ROS 2 Humble (the robot's Jetson, or WSL2 on a Windows laptop):

```bash
sudo apt-get install -y --no-upgrade python3-numpy python3-scipy python3-matplotlib
mkdir -p ~/marco_ws/src && cd ~/marco_ws/src
git clone https://github.com/luqeei1/jaffa.git
cd ~/marco_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash          # again in every new terminal
```

`--symlink-install` makes the build point at these source files instead of copying them, so an
edit takes effect without rebuilding and `audio_node` can find a run in `recordings/` by its name.

Run them, each in its own terminal:

```bash
ros2 run marco_audio audio_node --ros-args -p source:=replay -p replay_path:=dirtest_20261028_1921
ros2 topic echo /marco/audio/sound_event            # every message as it is published
ros2 run marco_audio speech_node
ros2 topic pub --once /marco/speech/say std_msgs/msg/String "{data: marco}"
```

The replay prints one line per clap. The first three are always −3.8°, −3.9° and −2.1°.
Leave out everything after `audio_node` to use the live microphone. That also needs PortAudio
and the `sounddevice` library (run sheet, section 10), and it needs Linux to see the board: the
Jetson does, WSL does not, so on a laptop use replay. Each node lists its parameters at the top
of its file.

Checks that need no ROS 2 and no board (stand-in ROS modules in `test/`), from this folder:

```bash
python test/test_audio_node.py     # 28 checks: replay publishes exactly what live_doa.py finds
python test/test_speech_node.py    # 22 checks: queueing, the speaking flag, a player that fails
python test/test_launch_file.py    # 45 checks: what the launch file starts, and with what settings
python test/test_keyword_spotter.py  # 35 checks: which words count, and how audio_node uses the spotter
```

With Vosk and its model installed, `test_keyword_spotter.py` runs 4 more checks with the real
recogniser: silence and hiss are not words, and none of the 24 claps of the 28 Oct session is
called "marco" or "polo". To see what the spotter says for every sound of a saved run:

```bash
cd tools
python keyword_spotter.py ../recordings/dirtest_20261028_1921
```

## 11. Launch file

`launch/marco_audio.launch.py` starts both nodes with one command. It also publishes where the
board sits on the robot, as a static transform `base_link -> mic_array_link`: without it the
angle in a `SoundEvent` has nothing to be measured from.

```bash
ros2 launch marco_audio marco_audio.launch.py                    # the live board
ros2 launch marco_audio marco_audio.launch.py source:=replay     # saved clap test in a loop, no board
ros2 launch marco_audio marco_audio.launch.py mic_x:=0.30 mic_y:=0.0 mic_z:=0.29 mic_yaw_deg:=0
ros2 launch marco_audio marco_audio.launch.py keywords:=false    # directions only, no word spotting
ros2 launch marco_audio marco_audio.launch.py --show-args        # every setting and its default
```

`mic_x`, `mic_y` and `mic_z` are the board's position in metres from `base_link` (forward, left,
up). `mic_yaw_deg` is the way its USB edge points (0 = the robot's front, 90 = its left). Give
the board's rotation in one place only: `mic_yaw_deg` (in the transform) or `yaw_offset_deg`
(added inside `audio_node`). The launch file refuses both at once, because the rotation would be
counted twice.

## 12. On the robot's Jetson

[`docs/JETSON_RUNSHEET.md`](docs/JETSON_RUNSHEET.md) is the step-by-step sheet: connecting, which
terminal window each command goes in, what to expect, a fallback, troubleshooting, the first-time
set-up, and the interface (topics and message fields) for the other nodes.

Result (11 Nov 2026, the team's Jetson Orin Nano, Ubuntu 22.04, ROS 2 Humble): Linux offers the
board as one sound card with 6 inputs and 2 outputs. The replay gave the same 24 answers as on
the laptop. Live, finger clicks from in front of the USB edge were reported at +1.5° to +2.0°,
and from the opposite side at −173.6° to −174.5°, all with confidence 1.00. `speech_node` played
its clips through the board's 3.5 mm jack.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Only 2 channels after flashing | Device Manager → uninstall every reSpeaker entry → unplug/replug so Windows reloads the driver |
| Raw channels (ch2–ch5) silent | Same as above (known Windows issue with the 6-ch firmware) |
| No sound device after Zadig | Zadig replaced the audio driver: uninstall in Device Manager, replug |
| `CLIPPED` in record.py | Clap further away, or more softly |
| `Invalid number of channels` when a script opens the board | The device number changed: run `check_device.py` and use the WASAPI line marked OK |
| Many low-confidence answers near a wall, some pointing backwards | A strong echo inside the window: keep the window ending 2 ms after the onset (default in `live_doa.py`; `--post-ms 2` for `analyse_phat_session.py`), or move the board away from the wall |
| Angle wrong but `|u|` ≈ 1 | Mic order/orientation wrong: do the tap test |
| `|u|` far from 1, delays marked "impossible" | Bad recording (echo, clipping, wrong channels) |
| Anything while running the ROS 2 nodes or on the Jetson | See the table in section 8 of `docs/JETSON_RUNSHEET.md` |

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