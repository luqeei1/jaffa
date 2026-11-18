"""
Synthetic test scene for align_view.py: test and practise the alignment without cameras.

A simple room is described as boxes in 3D (walls, a doorway, floor, a hot radiator, a warm
monitor, a chair and a standing person). Both cameras are ray-traced into it:
  - the D435i (848 x 480 depth, metres along its optical axis, with small noise),
  - the OAK Thermal (256 x 192, degrees C), placed with a HIDDEN "true" calibration that
    differs from the hand-measured defaults.
The result is saved as a normal snapshot. Load it in the viewer, tune the sliders until the
outlines line up, and compare with the true values stored in the snapshot's _meta.yaml
(key "true_calibration_answer_key"; look only after tuning!).

Usage:
  python make_test_scene.py                   # hidden calibration (practice)
  python make_test_scene.py --true-defaults   # thermal exactly at the default calibration
  python align_view.py --load captures/test_scene
"""

import argparse
import dataclasses
import os

import numpy as np
import yaml

import calibration as calib
from geometry import PinholeIntrinsics

HERE = os.path.dirname(os.path.abspath(__file__))
D435I = PinholeIntrinsics(fx=421.0, fy=421.0, cx=423.5, cy=239.5, width=848, height=480)

# Hidden "true" mounting for the practice scene: a few cm and degrees away from the defaults.
HIDDEN = dataclasses.replace(calib.ThermalCalibration(), y_left_m=-0.095, z_up_m=-0.26, roll_deg=0.75,
                             pitch_deg=-1.5, yaw_deg=2.5, fx=131.0, fy=146.5, cx=129.0, cy=94.0,
                             source="synthetic test scene (answer key)")

# Scene in the D435i OPTICAL frame (x right, y down, z forward, metres); the D435i is 0.6 m
# above the floor. Each box: (x_min, x_max, y_min, y_max, z_min, z_max, temperature C).
FLOOR_Y = 0.6
SCENE = [
    (-6.0, 0.9, -3.0, FLOOR_Y, 5.0, 5.1, 21.0),    # back wall, left of the doorway
    (1.8, 6.0, -3.0, FLOOR_Y, 5.0, 5.1, 21.0),     # back wall, right of the doorway
    (0.9, 1.8, -3.0, -1.5, 5.0, 5.1, 21.0),        # wall above the doorway
    (-6.0, 6.0, -3.0, FLOOR_Y, 8.0, 8.1, 19.5),    # room behind the doorway
    (0.8, 0.9, -1.55, FLOOR_Y, 4.95, 5.0, 21.6),   # door frame, left (protrudes 5 cm)
    (1.8, 1.9, -1.55, FLOOR_Y, 4.95, 5.0, 21.6),   # door frame, right
    (-6.0, 6.0, FLOOR_Y, FLOOR_Y + 0.1, 0.0, 8.1, 19.0),  # floor
    (-2.3, -2.2, -3.0, FLOOR_Y, 0.0, 5.1, 21.0),   # left wall
    (-2.2, -2.1, -0.05, 0.45, 2.6, 3.6, 48.0),     # radiator on the left wall (hot)
    (1.2, 1.75, -0.35, -0.02, 3.4, 3.45, 31.0),    # monitor (warm)
    (0.9, 1.75, -0.02, 0.02, 3.2, 3.6, 21.5),      # desk top
    (0.5, 1.0, 0.05, FLOOR_Y, 1.5, 1.9, 22.5),     # chair, close to the robot
    (-0.55, -0.05, -0.85, 0.1, 2.4, 2.65, 30.5),   # person: torso (clothing)
    (-0.5, -0.1, 0.1, FLOOR_Y, 2.45, 2.6, 28.5),   # person: legs
    (-0.4, -0.2, -1.15, -0.85, 2.45, 2.65, 34.5),  # person: head (skin)
]
EMPTY_TEMP_C = 20.0


def ray_trace(origin, dirs):
    """Nearest box hit for each ray. Returns (distance along the ray, temperature); inf/EMPTY if none."""
    boxes = np.array(SCENE, dtype=np.float64)
    lo = boxes[:, [0, 2, 4]]
    hi = boxes[:, [1, 3, 5]]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs                                       # (N, 3)
        t1 = (lo[None] - origin) * inv[:, None]                # (N, B, 3)
        t2 = (hi[None] - origin) * inv[:, None]
    t_near = np.nanmax(np.minimum(t1, t2), axis=2)             # slab method
    t_far = np.nanmin(np.maximum(t1, t2), axis=2)
    hit = (t_far >= np.maximum(t_near, 0.0))
    t = np.where(hit, np.where(t_near > 0, t_near, t_far), np.inf)
    best = np.argmin(t, axis=1)
    dist = t[np.arange(len(t)), best]
    temp = np.where(np.isfinite(dist), boxes[best, 6], EMPTY_TEMP_C)
    return dist, temp


def pixel_rays(intr):
    """Unit-z ray directions (x/z, y/z, 1) for every pixel of a pinhole camera."""
    v, u = np.mgrid[0:intr.height, 0:intr.width]
    return np.column_stack([((u - intr.cx) / intr.fx).ravel(), ((v - intr.cy) / intr.fy).ravel(),
                            np.ones(u.size)])


def render_depth(rng):
    dirs = pixel_rays(D435I)
    dist, _ = ray_trace(np.zeros(3), dirs)
    z = dist  # directions have z = 1, so the ray parameter IS the optical-axis depth
    z = np.where(np.isfinite(z) & (z <= 10.0), z, 0.0)
    # D435i-like noise: depth error grows with distance squared (~2 mm at 1 m).
    z = np.where(z > 0, z + rng.normal(0, 0.002, z.shape) * z ** 2, 0.0)
    return z.reshape(D435I.height, D435I.width).astype(np.float32)


def render_thermal(cal, rng):
    """Thermal image: rays from the thermal camera, expressed in the D435i optical frame."""
    to_thermal = cal.depth_to_thermal()
    to_depth = np.linalg.inv(to_thermal)          # thermal optical -> depth optical
    origin = to_depth[:3, 3]                      # thermal lens position in the depth frame
    dirs = pixel_rays(cal.intrinsics()) @ to_depth[:3, :3].T
    _, temp = ray_trace(origin, dirs)
    temp = temp + rng.normal(0, 0.1, temp.shape)  # sensor noise
    return temp.reshape(cal.height, cal.width).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--true-defaults", action="store_true",
                        help="place the thermal camera exactly at the default calibration")
    parser.add_argument("--out", default=os.path.join(HERE, "captures", "test_scene"), help="snapshot prefix")
    args = parser.parse_args()

    truth = calib.ThermalCalibration() if args.true_defaults else HIDDEN
    rng = np.random.default_rng(0)
    depth = render_depth(rng)
    temp = render_thermal(truth, rng)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out + "_thermal_C.npy", temp)
    np.save(args.out + "_depth_m.npy", depth)
    with open(args.out + "_meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"depth_intrinsics": dataclasses.asdict(D435I),
                        "dt_thermal_minus_depth_s": 0.0,
                        "true_calibration_answer_key": dataclasses.asdict(truth)}, f, sort_keys=False)
    print(f"Wrote {args.out}_*  (thermal {temp.min():.1f}-{temp.max():.1f} C, "
          f"depth on {100 * (depth > 0).mean():.0f}% of pixels)")
    print(f"Now run:  python align_view.py --load {os.path.relpath(args.out, HERE)}")


if __name__ == "__main__":
    main()
