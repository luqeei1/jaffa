"""
Copy the OAK Thermal's FACTORY lens calibration into calibration/thermal_to_color.yaml.

Luxonis calibrates the thermal sensor (socket CAM_E, 256 x 192) at the factory and stores it
on the device: fx, fy, cx, cy and an 8-coefficient OpenCV "rational" distortion model. It was
checked on 2026-11-18: undistorting a frame with it makes the curved ceiling lights straight.
This replaces the lens part of the checkerboard calibration (calib_capture.py --mode lens).

The factory data has NO pose between the thermal sensor and anything else (not even the
OAK's own colour camera), so the thermal <-> RealSense pose still has to be measured.

Only the intrinsics and distortion are written; the pose in the file is kept.

Usage (OAK connected, from the fusion folder):  python read_oak_factory.py
"""

import dataclasses

import calibration as calib


def read_factory():
    import depthai as dai
    with dai.Device() as dev:
        cal = dev.readCalibration()
        sockets = {f.socket: f for f in dev.getConnectedCameraFeatures()}
        sock = dai.CameraBoardSocket.CAM_E
        if sock not in sockets:
            raise SystemExit(f"No thermal sensor on CAM_E; cameras found: {list(sockets)}")
        w, h = sockets[sock].width, sockets[sock].height
        k = cal.getCameraIntrinsics(sock, w, h)
        dist = [float(v) for v in cal.getDistortionCoefficients(sock)[:8]]
        model = str(cal.getDistortionModel(sock))
    if "Perspective" not in model:
        raise SystemExit(f"Unexpected factory distortion model {model}; expected Perspective (OpenCV rational)")
    return k, dist, w, h


def main():
    k, dist, w, h = read_factory()
    old = calib.load()
    cal = dataclasses.replace(
        old, fx=float(k[0][0]), fy=float(k[1][1]), cx=float(k[0][2]), cy=float(k[1][2]), width=w, height=h,
        model="rational", dist=dist,
        source=f"{old.source}; lens = OAK Thermal factory calibration (read_oak_factory.py)")
    print(f"factory thermal lens: fx {cal.fx:.2f} fy {cal.fy:.2f} cx {cal.cx:.2f} cy {cal.cy:.2f}  {w}x{h}")
    print(f"  distortion (rational k1 k2 p1 p2 k3 k4 k5 k6): {[round(v, 4) for v in dist]}")
    print(f"Wrote {calib.save(cal)} (pose unchanged)")


if __name__ == "__main__":
    main()
