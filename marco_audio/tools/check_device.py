"""
Step 1: find the ReSpeaker and check it can deliver 6 channels at 16 kHz.

Run:   python check_device.py

What it prints:
  * every audio input device Windows knows about, grouped by "host API"
  * for each ReSpeaker entry, whether 6 channels @ 16 kHz can be opened,
    in normal (shared) mode and in WASAPI exclusive mode

Copy the index number of a line marked OK - you pass it to record.py.
"""

import sounddevice as sd

from mic_array import FS, N_CHANNELS

NAME_HINTS = ("respeaker", "xvf", "3800")


def looks_like_respeaker(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in NAME_HINTS)


def try_open(index: int, exclusive: bool) -> str:
    """Ask PortAudio whether this device accepts our format, without recording."""
    extra = None
    if exclusive:
        extra = sd.WasapiSettings(exclusive=True)
    try:
        sd.check_input_settings(device=index, channels=N_CHANNELS,
                                samplerate=FS, dtype="float32",
                                extra_settings=extra)
        return "OK"
    except Exception as e:  # PortAudio gives fairly cryptic messages
        return f"no ({e})"


def main() -> None:
    hostapis = sd.query_hostapis()
    devices = sd.query_devices()

    print(f"sounddevice {sd.__version__}, PortAudio: {sd.get_portaudio_version()[1]}\n")
    print("All input devices:")
    print(f"{'idx':>4}  {'host API':<22} {'in ch':>5} {'def. rate':>9}  name")
    for i, d in enumerate(devices):
        if d["max_input_channels"] == 0:
            continue
        api = hostapis[d["hostapi"]]["name"]
        star = "  <-- ReSpeaker?" if looks_like_respeaker(d["name"]) else ""
        print(f"{i:>4}  {api:<22} {d['max_input_channels']:>5} "
              f"{d['default_samplerate']:>9.0f}  {d['name']}{star}")

    candidates = [i for i, d in enumerate(devices)
                  if d["max_input_channels"] > 0 and looks_like_respeaker(d["name"])]
    if not candidates:
        print("\nNo ReSpeaker found. Is it plugged in, and does it show under "
              "Sound settings > Input? If you just ran Zadig, see the README "
              "troubleshooting section.")
        return

    print(f"\nCan each ReSpeaker entry open {N_CHANNELS} channels @ {FS} Hz?")
    for i in candidates:
        d = devices[i]
        api = hostapis[d["hostapi"]]["name"]
        print(f"  [{i}] {api}: shared -> {try_open(i, exclusive=False)}")
        if "WASAPI" in api:
            print(f"  [{i}] {api}: exclusive -> {try_open(i, exclusive=True)}")

    print("\nIf every line says 'no' and 'in ch' shows 2, the board is still on the "
          "2-channel firmware (or Windows cached the old driver) - see README.")


if __name__ == "__main__":
    main()
