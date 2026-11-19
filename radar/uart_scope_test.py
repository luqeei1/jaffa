#!/usr/bin/env python3
"""
uart_scope_test.py - sends one known byte out of the Jetson's header UART so the signal can
be looked at on an oscilloscope. Not part of the ROS 2 package.

What it does
    Every --interval seconds it sends the single byte 0xFA on Jetson pin 8 (TX), and nothing
    else. If something is wired to pin 10 (RX), any bytes that come back are printed in hex.

Run
    python3 radar/uart_scope_test.py
    python3 radar/uart_scope_test.py --interval 0.2 --port /dev/ttyTHS1
    Ctrl+C to stop.

What to expect on the scope (115200 baud, 8N1)
    idle line high (~3.3 V); one bit = 8.68 us; the whole byte (start + 8 data + stop) = 86.8 us.
    0xFA is sent least significant bit first, so after the falling edge it looks like:
        start(0)  0 1 0 1 1 1 1 1  stop(1)
    i.e. low for 2 bits, high 1, low 1, then high for 6 bits (5 data + stop) back into idle.
"""
import argparse
import time

import serial

BYTE = bytes([0xFA])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyTHS1", help="serial port (default /dev/ttyTHS1, header pins 8/10)")
    parser.add_argument("--baud", type=int, default=115200, help="baud rate (default 115200)")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between bytes (default 0.5)")
    args = parser.parse_args()

    port = serial.Serial(args.port, args.baud, timeout=0)
    port.reset_input_buffer()
    print(f"Sending {BYTE.hex()} on {args.port} @ {args.baud} every {args.interval} s. Ctrl+C to stop.")

    sent = 0
    try:
        while True:
            port.write(BYTE)
            port.flush()
            sent += 1
            time.sleep(args.interval)
            received = port.read(4096)
            note = f"  got {len(received)} B: {received[:40].hex(' ')}" if received else ""
            print(f"\rbytes sent: {sent}{note}", end="\n" if received else "", flush=True)
    except KeyboardInterrupt:
        print(f"\nStopped after {sent} bytes.")
    finally:
        port.close()


if __name__ == "__main__":
    main()
