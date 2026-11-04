"""
Read the XVF3800 chip's OWN direction estimates over USB.

The chip runs its own direction finding inside its DSP. These are the
"onboard" method we compare against our own code. Two readings:

  DOA_VALUE           degrees 0-359 + a 'speech detected' flag. Tuned for voices.
  AEC_AZIMUTH_VALUES  4 angles in radians: [beam 1, beam 2, free-running beam,
                      auto-selected beam]. The free-running beam follows the
                      loudest sound, speech or not, so it should react to claps.

How it talks to the chip: USB "vendor control transfers". Every USB device
has endpoint 0 for control messages; Seeed's firmware accepts custom requests
there, addressed by a resource id (which part of the firmware) and a command
id (which setting). Reads set bit 7 of the command id. The reply starts with
one status byte (0 = success, 64 = busy, ask again), then the data.
Protocol taken from Seeed's python_control/xvf_host.py.

Run on its own to watch the readings live (Ctrl+C to stop):
    python chip_doa.py
"""

import math
import struct
import sys
import time

import usb.core
import usb.util

VID, PID = 0x2886, 0x001A

# name: (resource id, command id, count, struct format of one value)
COMMANDS = {
    "VERSION": (48, 0, 3, "B"),
    "DOA_VALUE": (20, 18, 2, "H"),
    "AEC_AZIMUTH_VALUES": (33, 75, 4, "f"),
}
STATUS_OK, STATUS_RETRY = 0, 64


class ChipDOA:
    def __init__(self):
        if sys.platform.startswith("win"):
            import libusb_package       # ships the libusb DLL that pyusb needs on Windows
            dev = libusb_package.find(idVendor=VID, idProduct=PID)
        else:
            dev = usb.core.find(idVendor=VID, idProduct=PID)
        if dev is None:
            raise RuntimeError("XVF3800 not found on USB")
        self.dev = dev

    def read(self, name):
        resid, cmdid, count, fmt = COMMANDS[name]
        length = 1 + count * struct.calcsize(fmt)
        req_type = usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE
        for _ in range(100):
            resp = self.dev.ctrl_transfer(req_type, 0, 0x80 | cmdid, resid, length, 1000)
            if resp[0] == STATUS_OK:
                return struct.unpack("<" + fmt * count, bytes(resp[1:length]))
            if resp[0] != STATUS_RETRY:
                raise RuntimeError(f"{name}: status {resp[0]}")
            time.sleep(0.002)
        raise RuntimeError(f"{name}: chip stayed busy")

    def snapshot(self):
        """One reading of everything we log: (doa_deg, speech_flag, az0..az3 in degrees)."""
        doa, speech = self.read("DOA_VALUE")
        az = [math.degrees(a) % 360 if not math.isnan(a) else float("nan")
              for a in self.read("AEC_AZIMUTH_VALUES")]
        return doa, speech, az

    def close(self):
        usb.util.dispose_resources(self.dev)


if __name__ == "__main__":
    chip = ChipDOA()
    print("Firmware version:", ".".join(map(str, chip.read("VERSION"))))
    print("Make noises around the board. Ctrl+C to stop.\n")
    print(f"{'DOA_VALUE':>9} {'speech':>6} {'beam1':>7} {'beam2':>7} {'free':>7} {'auto':>7}")
    try:
        while True:
            doa, speech, az = chip.snapshot()
            print(f"{doa:>9} {speech:>6} " + " ".join(f"{a:>7.1f}" for a in az))
            time.sleep(0.2)
    except KeyboardInterrupt:
        chip.close()
