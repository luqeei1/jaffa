# Marco audio on the Jetson: run sheet

The exact commands, in order, for running Marco's ears and voice on the team's Jetson, with what
you should see after each one. Follow it from the top. Nothing here needs remembering.

Written from the session on the team's Jetson Orin Nano on 11 Nov 2026 (Ubuntu 22.04, ROS 2 Humble).
The interface for the other nodes (topics, message, frame) is at the end.

## The two windows

You work in two terminal windows on your laptop (PowerShell on Windows). Every command below is
labelled with the window to type it in.

| Name | What it is for |
|---|---|
| **Window 1** | runs Marco's ears and voice. It stays busy while they are running |
| **Window 2** | checks on them while Window 1 is busy |

Both start as ordinary laptop windows: the line starts with `PS C:\`. After you log in, the line
starts with `jaffa@jetson`, and everything typed there runs on the Jetson. Look at the start of the
line whenever you are unsure which machine a window is talking to.

## Quick reference: the commands used most

These assume the window is already logged in and set up. If it is not, do this first (the full
version is in sections 1 and 2).

**Window 1 and Window 2**, once each:

```
ssh jaffa@<jetson-address>
```

**Window 1 and Window 2**, once the line starts with `jaffa@jetson`:

```
cd ~/marco_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Testing on your own, not with the team's nodes? Then also, in **Window 1 and Window 2**:

```
export ROS_DOMAIN_ID=42
```

**Start Marco listening** (this also gets his voice ready). **Window 1:**

```
ros2 launch marco_audio marco_audio.launch.py
```

Wait for the `Listening.` line. Stop with Ctrl+C in Window 1.

**Make Marco speak.** Window 1 must be running. The clips are placeholder beeps for now.

**Window 2**, say "Marco" (two notes going up):

```
ros2 topic pub --once /marco/speech/say std_msgs/msg/String "{data: marco}"
```

**Window 2**, say "Polo" (two notes going down):

```
ros2 topic pub --once /marco/speech/say std_msgs/msg/String "{data: polo}"
```

**Window 2**, say "Found you" (three quick notes going up):

```
ros2 topic pub --once /marco/speech/say std_msgs/msg/String "{data: found_you}"
```

**Window 2**, say "Sleepy" (one long note sliding down):

```
ros2 topic pub --once /marco/speech/say std_msgs/msg/String "{data: sleepy}"
```

**Watch what Marco hears**, as the messages the other nodes receive. **Window 2:**

```
ros2 topic echo /marco/audio/sound_event
```

Stop with Ctrl+C in Window 2.

**Fallback if the microphone misbehaves:** a saved clap test instead of the live board. **Window 1:**

```
ros2 launch marco_audio marco_audio.launch.py source:=replay
```

**Shut the Jetson down.** **Window 1:**

```
sudo shutdown now
```

## 0. Before you start

- The Jetson is powered (green light next to its USB-C port) and on the same network as your laptop.
- The mic board (ReSpeaker XVF3800) is in any of the Jetson's four USB sockets. It lies flat, logo
  up, with nothing covering it.
- A speaker or earphones are in the **board's** 3.5 mm jack, not the Jetson's.
- You have the password for the user `jaffa`. Ask the team: it is not written anywhere in this repo.

House rules on the shared Jetson: our code lives in `~/marco_ws` only. Never run `apt upgrade`.
Never edit `~/.bashrc`. Do not start anything heavy while someone is drive-testing the robot.

## 1. Connect

**Window 1:**

```
ssh jaffa@<jetson-address>
```

Type `yes` if it asks whether to continue, then the password (nothing shows while you type).
You are in when the line starts with `jaffa@jetson`.

Finding the address:

- On the same Wi-Fi as on 11 Nov, try the address it had then first: `172.26.172.192`.
- If that gives `Connection timed out`, the network has given the Jetson a new address. Any of
  these finds it:
  - someone who is already logged in runs `hostname -I` on the Jetson. The first number is the address
  - on a phone hotspot, look at its list of connected devices
  - plug a USB cable from the Jetson's USB-C port into your laptop and run this in a laptop window
    that is **not** logged in (the line starts with `PS C:\`):

    ```
    ssh jaffa@192.168.55.1 "hostname -I"
    ```

    That cable link keeps dropping out on Windows, so repeat it up to five times. It only has to
    hold for a few seconds. Then unplug the cable and use the address it printed.

## 2. Set up the window

**Window 1:**

```
cd ~/marco_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

These tell the window where ROS 2 and our two packages are. A window that has not run them does
not know the `ros2` command or the `marco_audio` package. Window 2 gets the same lines in section 5.

Then choose one:

- **Working with the team's nodes:** nothing more. You share their ROS domain.
- **Testing on your own:** one more line.

  **Window 1:**

  ```
  export ROS_DOMAIN_ID=42
  ```

  A domain is like a radio channel: nodes on different numbers cannot hear each other, so your test
  stays private. The setting lasts only until that window is closed. Window 2 needs the same line
  (section 5), or the two windows cannot hear each other.

## 3. Get the latest code and build

Once per session, and again after anyone pushes a change. Window 1 only: it changes files on the
Jetson's disk, which both windows share, so it never needs doing twice.

**Window 1:**

```
cd ~/marco_ws/src/jaffa
git pull
cd ~/marco_ws
colcon build --symlink-install
source install/setup.bash
```

You should see `Already up to date.` or a list of changed files, then `Summary: 2 packages finished`.

## 4. Start Marco's ears and voice

**Window 1:**

```
ros2 launch marco_audio marco_audio.launch.py
```

Window 1 should show, in the first few seconds:

1. `Marco audio: ears = live, voice = on, mic_array_link is ...`
2. `[mic_array_tf]: Spinning until stopped - publishing transform` and `from 'base_link' to 'mic_array_link'`
3. `[speech_node]: Ready. Player: aplay -q -D plughw:CARD=Array,DEV=0. Clips in ...: found_you, marco, polo, sleepy.`
4. `[audio_node]: Keyword spotting is on: listening for marco, polo on channel 1 ...`
   (or `Keyword spotting is off ...` if the recogniser is not installed: section 10)
5. `[audio_node]: Listening to device <number>: reSpeaker XVF3800 4-Mic Array ...` then `Background -9x dBFS ... Listening.`

Stay quiet until the `Listening.` line: the node is measuring the room's background noise.
After that, every sound gives one line in Window 1, for example:

```
#1:   +2.0 deg  (+-1.4)  confidence 1.00  level -52 dBFS
```

0 deg is straight out of the board's USB edge. Positive angles are anticlockwise seen from above
(to the left). Window 1 is now busy: leave it running and do the checks in Window 2.

With keyword spotting on, say "Polo!" clearly, about 1 m from the board. Its line ends with the
word and how sure the recogniser is:

```
#2:   +3.1 deg  (+-2.0)  confidence 0.93  level -48 dBFS  keyword POLO (0.97)
```

Each line now appears about a second after its sound, because the recogniser has to hear the
whole word first. A line ending in `(heard polo 0.41: not counted)` means the word was heard
but the recogniser was not sure enough.
To stop everything, press **Ctrl+C** in Window 1.

**Once the board is mounted on the robot**, say where it is. Measure from `base_link` (the centre
of the robot, at axle height) to the centre of the board, in metres, and start it with this
instead of the command above. The four numbers here are only an example: change them to your own
measurements.

**Window 1:**

```
ros2 launch marco_audio marco_audio.launch.py mic_x:=0.10 mic_y:=0.0 mic_z:=0.25 mic_yaw_deg:=0
```

`mic_x` = forward, `mic_y` = left, `mic_z` = up, `mic_yaw_deg` = which way the USB edge points
(0 = the robot's front, 90 = its left, 180 = its back, -90 = its right).

`mic_z` is measured from `base_link`, not from the floor. `base_link` is at axle height, about
0.11 m above the floor (confirm the figure with whoever set up navigation), so
`mic_z` = the board's height above the floor minus 0.11.

Worked example: the board is on top of the robot at the very front, 0.30 m ahead of the centre, in
the middle of the robot's width, 0.40 m above the floor, with its USB edge pointing forwards.
Then `mic_x` = 0.30, `mic_y` = 0, `mic_z` = 0.40 - 0.11 = 0.29 and `mic_yaw_deg` = 0.

**Window 1:**

```
ros2 launch marco_audio marco_audio.launch.py mic_x:=0.30 mic_y:=0.0 mic_z:=0.29 mic_yaw_deg:=0
```

`mic_yaw_deg` is the one that has to be right: an error there turns every direction by the same
amount. `mic_x` and `mic_y` to the nearest few centimetres are enough, and `mic_z` hardly changes
a direction at all.

## 5. Check it from Window 2

Leave Window 1 running. First log Window 2 in and set it up.

**Window 2:**

```
ssh jaffa@<jetson-address>
```

**Window 2**, once the line starts with `jaffa@jetson`:

```
cd ~/marco_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

**Window 2**, only if you ran it in Window 1 (testing on your own):

```
export ROS_DOMAIN_ID=42
```

Now the four checks.

**Window 2:**

```
ros2 topic list | grep marco
```

Window 2 shows `/marco/audio/sound_event`, `/marco/speech/active` and `/marco/speech/say`.

**Window 2:**

```
ros2 topic echo /marco/audio/sound_event
```

Clap once. Window 2 prints one message per sound: look for `frame_id: mic_array_link`, `azimuth`,
`confidence` and `method: gcc_phat_robust`. Window 1 prints its usual line for the same sound.
Press Ctrl+C in Window 2 to stop the echo (not in Window 1).

**Window 2:**

```
ros2 topic pub --once /marco/speech/say std_msgs/msg/String "{data: marco}"
```

You hear the clip called marco (also: `polo`, `found_you`, `sleepy`). Window 1 prints `Saying 'marco'`.

**Window 2:**

```
ros2 run tf2_ros tf2_echo base_link mic_array_link
```

Window 2 prints the board's position and rotation on the robot, once a second. Press Ctrl+C in
Window 2 to stop it.

## 6. Fallback: no board, or the live microphone misbehaves

Press Ctrl+C in Window 1 to stop the live version, then:

**Window 1:**

```
ros2 launch marco_audio marco_audio.launch.py source:=replay
```

This plays the saved clap test of 28 Oct through the same code, over and over. The other nodes
cannot tell the difference. The first three sounds are always `-3.8`, `-3.9` and `-2.1 deg`.
The checks in section 5 work the same way in Window 2.

## 7. Settings worth knowing

They go on the end of the launch command in Window 1, as `name:=value`. Adding `--show-args`
instead lists all of them.

| Setting | Default | What it does |
|---|---|---|
| `source` | `live` | `replay` = play a saved run instead of the microphone |
| `min_db` | `0` (= -70) | sounds quieter than this (dBFS) are ignored. `min_db:=-80` hears quieter sounds |
| `min_confidence` | `0.0` | `min_confidence:=0.5` stops doubtful directions being published |
| `method` | `gcc_phat_robust` | or `gcc_phat`, `srp_phat` |
| `speech` | `true` | `false` = ears only |
| `keywords` | `auto` | `auto` = spot "marco" / "polo" if the recogniser is installed. `false` = directions only. `true` = stop with an error if it is missing |
| `kws_min_conf` | `0.6` | a word the recogniser is less sure of than this is not counted. `kws_min_conf:=0.4` catches more words and makes more mistakes |
| `publish_tf` | `true` | `false` if another node already publishes `base_link -> mic_array_link` |
| `mic_x` `mic_y` `mic_z` `mic_yaw_deg` | `0` | where the board is on the robot (section 4) |
| `sound_topic` | `/marco/audio/sound_event` | publish the sounds somewhere else |

## 8. If something goes wrong

"Launch again" means: Ctrl+C in Window 1, then the launch command of section 4 in Window 1.
Any other command in this table goes in Window 2.

| What you see | Why | What to do |
|---|---|---|
| `ssh: ... Connection timed out` | wrong address, or laptop and Jetson are on different networks | find the address again (section 1) |
| `Connection reset` right after the password | you took more than about 2 minutes | run `ssh` again and type the password straight away |
| `ros2: command not found`, or `Package 'marco_audio' not found` | that window has not run section 2 | run the three lines of section 2 in that window |
| `No sound device with 6 or more inputs matches ...` and a list (Window 1) | the Jetson does not see the board | `lsusb` should show `2886:001a ... reSpeaker XVF3800`. Unplug and replug the board, launch again |
| `No audio has arrived from the board for 2 s` (Window 1) | the cable was knocked | replug the board, launch again |
| `No module named 'sounddevice'` (Window 1) | the first-time set-up was not done for this user | section 10 |
| claps give no line in Window 1 | quieter than -70 dBFS, or not 20 dB above the room's noise | clap closer, or launch again with `min_db:=-80` on the end |
| many `[inconsistent]` lines or low confidence | echoes: board covered, or right next to a wall | keep the board in the open, 30 cm or more from walls; launch again with `min_confidence:=0.5` on the end |
| `Keyword spotting is off ...` (Window 1) | the recogniser or its model is not installed for this user | section 10, the keyword spotter part. The directions work without it |
| "Polo!" gives a line with no keyword | too quiet, too far, or said while someone else was talking | say it louder and alone; look for `(heard polo ...: not counted)` and, if it shows, launch again with `kws_min_conf:=0.4` on the end |
| other words are reported as `POLO` or `MARCO` | words that sound alike ("hello", "marker") can fool it | launch again with `kws_min_conf:=0.8` on the end, or `keywords:=false` to switch it off |
| the voice is too quiet or too loud | the board's playback volume | `amixer -c Array sset 'PCM',0 55` then `amixer -c Array sset 'PCM',1 55` (scale 0 to 60) |
| no voice at all | plugged into the wrong socket | the plug goes in the board's jack; the `Ready. Player:` line in Window 1 must say `CARD=Array` |
| `ros2 topic echo` prints nothing in Window 2 | the two windows are on different ROS domains | the same `export ROS_DOMAIN_ID=...` line in both windows, or in neither |
| `git pull` or `git clone` stops moving | the network stalled | Ctrl+C in that window and run it again |

## 9. Shut down

Press Ctrl+C in Window 1 if Marco's audio is still running, then:

**Window 1:**

```
sudo shutdown now
```

Both windows lose their connection, which is expected. Wait for the green light to go out (about
15 seconds), then switch off at the wall. Never pull the plug while it is running: that can damage
the storage.

## 10. First-time set-up (already done on the team's Jetson, 11 Nov 2026)

Only needed on a new Jetson or after it has been wiped. It adds five small packages and one Python
library, and upgrades nothing. Do section 1 first.

**Window 1:**

```
sudo apt-get install -y --no-upgrade --no-install-recommends libportaudio2 python3-cffi python3-pip
python3 -m pip install --user --no-deps sounddevice
mkdir -p ~/marco_ws/src
cd ~/marco_ws/src
git clone https://github.com/luqeei1/jaffa.git
cd ~/marco_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

`numpy`, `scipy`, `matplotlib` and `alsa-utils` were already on the Jetson. If they are missing:

**Window 1:**

```
sudo apt-get install -y --no-upgrade python3-numpy python3-scipy python3-matplotlib alsa-utils
```

Then check that Linux sees the board with all six channels.

**Window 1:**

```
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

One line must read `reSpeaker XVF3800 4-Mic Array: USB Audio (hw:...), ALSA (6 in, 2 out)`.

**The keyword spotter** (the words "marco" and "polo") needs the Vosk speech recogniser and its
model: three small Python libraries for this user and one 40 MB folder in `~/marco_ws/models`.
Without them everything else still works and `keyword` stays empty.

**Window 1:**

```
python3 -c "import requests, cffi; print('ok')"
python3 -m pip install --user --no-deps vosk srt tqdm
mkdir -p ~/marco_ws/models
cd ~/marco_ws/models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
python3 -m zipfile -e vosk-model-small-en-us-0.15.zip .
ls vosk-model-small-en-us-0.15
```

The first line must print `ok` and the last one `README  am  conf  graph  ivector`.
Then carry on from section 2.

---

# Interface: what the audio nodes give the rest of Marco

| Topic | Type | Direction | Meaning |
|---|---|---|---|
| `/marco/audio/sound_event` | `marco_audio_msgs/msg/SoundEvent` | audio_node publishes | one message per sound heard |
| `/marco/speech/say` | `std_msgs/msg/String` | speech_node listens | the name of a clip to play: `marco`, `polo`, `found_you`, `sleepy` |
| `/marco/speech/active` | `std_msgs/msg/Bool` | speech_node publishes | `true` from just before a clip starts until it ends; repeated every 0.5 s |
| transform `base_link -> mic_array_link` | static | the launch file publishes | where the board is on the robot |

**SoundEvent**

| Field | Type | Meaning |
|---|---|---|
| `header.stamp` | time | when the sound started |
| `header.frame_id` | string | the frame the angle is measured in: `mic_array_link` |
| `azimuth` | float32 | direction the sound came from, radians, -pi to pi. 0 = straight ahead (+x), positive = anticlockwise seen from above, so +pi/2 is to the left (REP 103) |
| `azimuth_std` | float32 | one-sigma uncertainty of `azimuth`, radians. pi = direction unknown |
| `confidence` | float32 | 0 (do not trust) to 1 (trust). Below 0.5 is doubtful |
| `level_db` | float32 | loudness in dBFS: 0 is the loudest the board can measure, a clap at 1 m is about -40 |
| `method` | string | `gcc_phat`, `gcc_phat_robust` or `srp_phat` |
| `type` | string | `keyword` if the sound was one of the game words, otherwise `sound` |
| `keyword` | string | `marco` or `polo`, or empty if it was neither (always empty when the keyword spotter is off) |

**Using it in another node (Python)**

```python
from marco_audio_msgs.msg import SoundEvent

self.create_subscription(SoundEvent, "/marco/audio/sound_event", self.on_sound, 10)

def on_sound(self, msg):
    if msg.keyword == "polo":
        ...                         # somebody answered "Polo!"
    if msg.confidence < 0.5:
        return                      # doubtful direction: ignore it
    bearing = msg.azimuth           # radians, in the frame msg.header.frame_id
    spread = msg.azimuth_std        # radians, one sigma
```

The message type lives in the package `marco_audio_msgs`. A terminal can use it after
`source ~/marco_ws/install/setup.bash` (run it after sourcing your own workspace).

**Things to know**

- The angle is measured from the board, not from the map. Use the transform to turn it into a
  direction in `base_link` or `map`. If the board is mounted with its USB edge to the robot's
  front, the angle is already the angle from the robot's front.
- audio_node drops any sound that starts while `/marco/speech/active` is true (and for 0.3 s
  after), so Marco does not report his own voice.
- Sounds with a doubtful direction are still published, with a low `confidence`. Filter on it.
- With the keyword spotter on, a message goes out about 1 s after its sound started, because
  the whole word has to be heard first. `header.stamp` is still the moment the sound started,
  so use the stamp, not the arrival time.
- `keyword` says what the recogniser heard, `confidence` says how good the direction is. They
  are separate: a clear "polo" can come with a doubtful direction. A "marco" or "polo" is
  published even when its direction is below `min_confidence`.
- The recogniser only knows two words, so a word that sounds like one of them ("hello",
  "marker") can be reported as it. Treat `polo` as "probably polo", most of all outside the
  seconds after Marco has called out.
- Marco's voice must come out through the mic board (its speaker socket or 3.5 mm jack). The
  board can only cancel the echo of sound that it played itself.
