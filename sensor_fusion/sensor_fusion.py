"""
Marco sensor-fusion belief-map prototype.

Dependencies:
    pip install numpy matplotlib

This is a simulation using dummy observations, not a hardware driver.
All positions and bearings are assumed to be expressed in the map frame.
"""

import math
from dataclasses import dataclass
import numpy as np


# ----------------------------- Configuration -----------------------------

MAP_MIN_X, MAP_MAX_X = 0.0, 8.0
MAP_MIN_Y, MAP_MAX_Y = 0.0, 6.0
RESOLUTION = 0.05                 # metres per grid cell
CHI2_95_DOF3 = 7.815               # 95% chi-squared threshold, 3D
ROBOT_XY = np.array([1.0, 1.0])
ROBOT_YAW = np.deg2rad(0.0)        # robot heading in map frame


@dataclass
class PositionObservation:
    position: np.ndarray           # [x, y, z] in map frame, metres
    covariance: np.ndarray         # 3x3 covariance, m^2
    confidence: float              # detector confidence in [0, 1]


def rigid_transform(matrix):
    """Validate and return a homogeneous rigid transform."""
    transform = np.asarray(matrix, dtype=float)
    if (transform.shape != (4, 4) or not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], [0, 0, 0, 1])):
        raise ValueError("camera transform must be a finite 4x4 rigid transform")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0)):
        raise ValueError("camera transform must contain a proper rotation")
    return transform


def rgbd_position_observation(frame, camera_to_map):
    """Adapt a single-person RGB-D frame using its calibrated optical-to-map pose."""
    transform = rigid_transform(camera_to_map)
    rotation = transform[:3, :3]
    # ponytail: skip ambiguous frames until a target association is available.
    if len(frame.detections) != 1:
        return None
    detection = frame.detections[0]
    if detection.type != "POSITION":
        return None
    return PositionObservation(
        position=rotation @ detection.position + transform[:3, 3],
        covariance=rotation @ detection.covariance @ rotation.T,
        confidence=detection.confidence,
    )


@dataclass
class AudioObservation:
    bearing_rad: float             # relative to the observation frame's forward axis
    std_rad: float                 # angular standard deviation
    confidence: float
    keyword_detected: bool         # e.g. recognised "Polo"
    unclassified: bool = False     # caller explicitly accepts an unclassified sound


@dataclass
class RadarObservation:
    range_m: float                 # measured target range, metres
    range_std_m: float
    confidence: float
    human_present: bool
    # Field of view is represented by half-angle around robot heading.
    half_fov_rad: float = np.deg2rad(60.0)


@dataclass
class RobotPose:
    """Robot pose in the fixed map frame at the sensor timestamp."""
    xy: np.ndarray
    yaw_rad: float
    timestamp_s: float

    def __post_init__(self):
        self.xy = np.asarray(self.xy, dtype=float)
        if self.xy.shape != (2,) or not np.all(np.isfinite(self.xy)):
            raise ValueError("RobotPose.xy must be two finite values")
        if not np.isfinite(self.yaw_rad) or not np.isfinite(self.timestamp_s):
            raise ValueError("RobotPose values must be finite")


# ------------------------------ Map creation ------------------------------

xs = np.arange(MAP_MIN_X, MAP_MAX_X + RESOLUTION, RESOLUTION)
ys = np.arange(MAP_MIN_Y, MAP_MAX_Y + RESOLUTION, RESOLUTION)
XX, YY = np.meshgrid(xs, ys)

def normalise(grid):
    """Normalise a non-negative grid so that its sum is one."""
    total = np.sum(grid)
    if not np.isfinite(total) or total <= 0:
        return np.full_like(grid, 1.0 / grid.size)
    return grid / total


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def diffuse_belief(belief, amount=0.08):
    """
    Simple motion prediction: mix the current belief with its four
    neighbouring cells. This allows uncertainty to grow over time.
    """
    neighbours = (
        np.roll(belief, 1, axis=0)
        + np.roll(belief, -1, axis=0)
        + np.roll(belief, 1, axis=1)
        + np.roll(belief, -1, axis=1)
    ) / 4.0
    predicted = (1.0 - amount) * belief + amount * neighbours
    # Remove wrap-around probability introduced by np.roll at map edges.
    predicted[0, :] = np.minimum(predicted[0, :], belief[0, :] + 1e-15)
    predicted[-1, :] = np.minimum(predicted[-1, :], belief[-1, :] + 1e-15)
    predicted[:, 0] = np.minimum(predicted[:, 0], belief[:, 0] + 1e-15)
    predicted[:, -1] = np.minimum(predicted[:, -1], belief[:, -1] + 1e-15)
    return normalise(predicted)


def translate_belief(belief, delta_xy):
    """Translate a map-frame belief by a target displacement in metres."""
    delta_x, delta_y = np.asarray(delta_xy, dtype=float)
    if not np.all(np.isfinite([delta_x, delta_y])):
        raise ValueError("delta_xy must contain finite values")

    shifted_x = np.zeros_like(belief)
    source_x = xs - delta_x
    for row in range(belief.shape[0]):
        shifted_x[row, :] = np.interp(
            source_x, xs, belief[row, :], left=0.0, right=0.0
        )

    shifted = np.zeros_like(belief)
    source_y = ys - delta_y
    for column in range(belief.shape[1]):
        shifted[:, column] = np.interp(
            source_y, ys, shifted_x[:, column], left=0.0, right=0.0
        )
    return normalise(shifted)


def fuse_position_observations(visible, thermal):
    """Gate and fuse position observations, returning the accepted estimate."""
    if visible is None and thermal is None:
        return None, None, False
    if visible is None:
        return thermal.position.copy(), thermal.covariance.copy(), True
    if thermal is None:
        return visible.position.copy(), visible.covariance.copy(), True

    _, accepted, _ = mahalanobis_gate(visible, thermal)
    if accepted:
        position, covariance = covariance_intersection(
            visible, thermal, omega=0.5
        )
        return position, covariance, True
    return visible.position.copy(), visible.covariance.copy(), False


# ------------------- RGB-D / thermal association and fusion ----------------

def mahalanobis_gate(visible, thermal, cross_covariance=None):
    """
    Test whether visible and thermal 3D positions are statistically
    consistent. cross_covariance is Cov(error_visible, error_thermal).

    If omitted, the code assumes independent errors for this gate.
    That assumption is only an approximation because the thermal
    algorithm uses the RGB-D depth image.
    """
    residual = np.asarray(visible.position) - np.asarray(thermal.position)

    if cross_covariance is None:
        S = visible.covariance + thermal.covariance
    else:
        S = (visible.covariance + thermal.covariance
             - cross_covariance - cross_covariance.T)

    # Small diagonal regularisation helps if S is nearly singular.
    S = S + np.eye(3) * 1e-9
    d2 = float(residual.T @ np.linalg.solve(S, residual))
    return d2, d2 <= CHI2_95_DOF3, S


def covariance_intersection(obs_a, obs_b, omega=0.5):
    """
    Conservative fusion for estimates with unknown cross-correlation.
    omega=0.5 is used here as a simple demonstration setting.
    """
    cov_a = obs_a.covariance
    cov_b = obs_b.covariance
    info_a = np.linalg.inv(cov_a)
    info_b = np.linalg.inv(cov_b)

    fused_info = omega * info_a + (1.0 - omega) * info_b
    fused_cov = np.linalg.inv(fused_info)
    fused_pos = fused_cov @ (
        omega * info_a @ obs_a.position
        + (1.0 - omega) * info_b @ obs_b.position
    )
    return fused_pos, fused_cov


def visual_likelihood(position_xyz, covariance_xyz):
    """
    Convert a fused 3D map-frame position into a 2D Gaussian likelihood.
    The x-y block is used because the belief map is horizontal.
    """
    mean_xy = np.asarray(position_xyz[:2])
    cov_xy = np.asarray(covariance_xyz[:2, :2])
    cov_xy = cov_xy + np.eye(2) * 1e-8

    dx = XX - mean_xy[0]
    dy = YY - mean_xy[1]
    inv_cov = np.linalg.inv(cov_xy)

    exponent = (
        inv_cov[0, 0] * dx**2
        + 2.0 * inv_cov[0, 1] * dx * dy
        + inv_cov[1, 1] * dy**2
    )
    likelihood = np.exp(-0.5 * exponent)

    # A small floor prevents a single imperfect model from making all
    # other cells have exactly zero probability.
    return np.maximum(likelihood, 1e-6)


# ------------------------------ Audio model -------------------------------

def audio_likelihood(obs, robot_xy=ROBOT_XY, robot_yaw=ROBOT_YAW, *, reliability=None, min_confidence=0.4):
    """Bearing-only likelihood: high probability along a directional wedge."""
    if obs is None or not (obs.keyword_detected or obs.unclassified) or obs.confidence < min_confidence:
        return np.ones_like(XX), False

    candidate_bearings = np.arctan2(YY - robot_xy[1], XX - robot_xy[0])
    measured_bearing = wrap_angle(robot_yaw + obs.bearing_rad)
    angular_error = wrap_angle(candidate_bearings - measured_bearing)

    sigma = max(obs.std_rad, np.deg2rad(2.0))
    likelihood = np.exp(-0.5 * (angular_error / sigma)**2)

    # At the microphone the bearing is undefined. Marginalise over all angles.
    angular_mean = sigma / math.sqrt(2 * math.pi) * math.erf(math.pi / (math.sqrt(2) * sigma))
    likelihood[np.hypot(XX - robot_xy[0], YY - robot_xy[1]) < 0.15] = angular_mean
    if reliability is None:
        reliability = obs.confidence if obs.unclassified else 1.0
    likelihood = (1.0 - reliability) + reliability * likelihood
    return np.maximum(likelihood, 1e-6), True


# ------------------------------ Radar model -------------------------------

def radar_likelihood(obs, robot_xy=ROBOT_XY, robot_yaw=ROBOT_YAW):
    """Range-ring likelihood, restricted to the radar field of view."""
    if obs is None or not obs.human_present or obs.confidence < 0.4:
        return np.ones_like(XX), False

    dx = XX - robot_xy[0]
    dy = YY - robot_xy[1]
    candidate_range = np.hypot(dx, dy)
    candidate_bearing = np.arctan2(dy, dx)
    relative_bearing = wrap_angle(candidate_bearing - robot_yaw)

    sigma = max(obs.range_std_m, 0.05)
    likelihood = np.exp(
        -0.5 * ((candidate_range - obs.range_m) / sigma)**2
    )

    in_fov = np.abs(relative_bearing) <= obs.half_fov_rad
    likelihood = np.where(in_fov, likelihood, 1e-6)
    return np.maximum(likelihood, 1e-6), True


# ----------------------------- Belief update ------------------------------

def update_belief(prior, visual=None, audio=None, radar=None):
    """Bayesian grid update using only sensors with valid observations."""
    log_belief = np.log(np.maximum(prior, 1e-300))
    used = []

    if visual is not None:
        log_belief += np.log(visual)
        used.append("visual")

    if audio is not None:
        log_belief += np.log(audio)
        used.append("audio")

    if radar is not None:
        log_belief += np.log(radar)
        used.append("radar")

    # Stable normalisation in log space.
    log_belief -= np.max(log_belief)
    posterior = np.exp(log_belief)
    return normalise(posterior), used


class BeliefMapFilter:
    """Stateful predict/update filter for repeated sensor-fusion cycles."""

    def __init__(self, initial_belief=None, diffusion_amount=0.02):
        if initial_belief is None:
            initial_belief = np.full(XX.shape, 1.0 / XX.size)
        initial_belief = np.asarray(initial_belief, dtype=float)
        if initial_belief.shape != XX.shape:
            raise ValueError(
                f"initial_belief must have shape {XX.shape}, "
                f"got {initial_belief.shape}"
            )
        self.belief = normalise(np.maximum(initial_belief, 0.0))
        self.diffusion_amount = diffusion_amount
        self._last_timestamp_s = None

    def reset(self, belief=None):
        """Replace the current map, useful when tracking a new target."""
        if belief is None:
            belief = np.full(XX.shape, 1.0 / XX.size)
        belief = np.asarray(belief, dtype=float)
        if belief.shape != XX.shape:
            raise ValueError(
                f"belief must have shape {XX.shape}, got {belief.shape}"
            )
        self.belief = normalise(np.maximum(belief, 0.0))
        self._last_timestamp_s = None

    def age(self, dt_s, half_life_s=5.0):
        """Forget old evidence in elapsed time, including periods without detections.

        This uniform mixture models loss of target location, not a known velocity.
        """
        if not np.isfinite(dt_s) or dt_s < 0 or not np.isfinite(half_life_s) or half_life_s <= 0:
            raise ValueError('age needs nonnegative elapsed time and a positive half-life')
        retained = 2.0 ** (-dt_s / half_life_s)
        self.belief = normalise(retained * self.belief + (1 - retained) / self.belief.size)
        return self.belief.copy()

    def update(
        self,
        visual=None,
        audio=None,
        radar=None,
        target_velocity_xy=None,
        dt_s=None,
    ):
        """
        Predict from the current map, then retain and return the posterior.

        target_velocity_xy is an estimated target velocity in map-frame
        metres/second. If unavailable, the filter uses diffusion only.
        """
        prior = self.belief
        if target_velocity_xy is not None:
            if dt_s is None or dt_s <= 0 or not np.isfinite(dt_s):
                raise ValueError(
                    "positive finite dt_s is required with target velocity"
                )
            target_velocity_xy = np.asarray(target_velocity_xy, dtype=float)
            if target_velocity_xy.shape != (2,) or not np.all(
                np.isfinite(target_velocity_xy)
            ):
                raise ValueError(
                    "target_velocity_xy must contain two finite values"
                )
            prior = translate_belief(
                prior, target_velocity_xy * float(dt_s)
            )
        prior = diffuse_belief(
            prior,
            amount=self.diffusion_amount,
        )
        self.belief, sensors_used = update_belief(
            prior,
            visual=visual,
            audio=audio,
            radar=radar,
        )
        return self.belief.copy(), sensors_used

    def update_from_observations(
        self,
        robot_pose,
        visible=None,
        thermal=None,
        audio=None,
        radar=None,
        target_velocity_xy=None,
    ):
        """
        Hardware-facing update entry point.

        Sensor adapters must provide observations transformed into the map
        frame, with covariances in matching units. Audio bearing is relative
        to the robot forward axis. Radar range is in metres.
        """
        if not isinstance(robot_pose, RobotPose):
            raise TypeError("robot_pose must be a RobotPose")

        position, covariance, position_accepted = fuse_position_observations(
            visible, thermal
        )
        visual = None
        if position is not None:
            visual = visual_likelihood(position, covariance)

        audio_map, audio_used = audio_likelihood(
            audio,
            robot_xy=robot_pose.xy,
            robot_yaw=robot_pose.yaw_rad,
        )
        radar_map, radar_used = radar_likelihood(
            radar,
            robot_xy=robot_pose.xy,
            robot_yaw=robot_pose.yaw_rad,
        )

        if self._last_timestamp_s is None:
            dt_s = None
        else:
            dt_s = robot_pose.timestamp_s - self._last_timestamp_s
            if dt_s <= 0:
                raise ValueError("robot pose timestamps must increase")

        belief, sensors_used = self.update(
            visual=visual,
            audio=audio_map if audio_used else None,
            radar=radar_map if radar_used else None,
            target_velocity_xy=target_velocity_xy,
            dt_s=dt_s,
        )
        self._last_timestamp_s = robot_pose.timestamp_s
        return belief, sensors_used, position_accepted

    def estimate(self):
        """Return the maximum-probability map cell and its probability."""
        max_index = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        return XX[max_index], YY[max_index], self.belief[max_index]


# ---------------------------- Dummy sensor data ---------------------------

def make_dummy_observations():
    """
    Simulated child location is approximately (5.0, 3.8) m.
    Small differences are added to visible and thermal positions.
    """
    visible = PositionObservation(
        position=np.array([4.95, 3.78, 0.85]),
        covariance=np.diag([0.18**2, 0.18**2, 0.25**2]),
        confidence=0.92,
    )

    thermal = PositionObservation(
        position=np.array([5.08, 3.86, 0.82]),
        covariance=np.diag([0.25**2, 0.25**2, 0.30**2]),
        confidence=0.84,
    )

    # Direction from robot (1,1) to child (5,3.8), plus small noise.
    true_bearing = np.arctan2(3.8 - 1.0, 5.0 - 1.0)
    audio = AudioObservation(
        bearing_rad=true_bearing + np.deg2rad(3.0),
        std_rad=np.deg2rad(8.0),
        confidence=0.95,
        keyword_detected=True,     # recognised "Polo"
    )

    radar = RadarObservation(
        range_m=np.hypot(5.0 - 1.0, 3.8 - 1.0) + 0.12,
        range_std_m=0.35,
        confidence=0.82,
        human_present=True,
        half_fov_rad=np.deg2rad(60.0),
    )
    return visible, thermal, audio, radar


def make_moving_observations(target_xy, robot_xy, robot_yaw, rng):
    """Create noisy observations for one step of the moving simulation."""
    target_xy = np.asarray(target_xy, dtype=float)
    delta = target_xy - robot_xy
    true_bearing = np.arctan2(delta[1], delta[0])
    true_range = np.linalg.norm(delta)

    visible = PositionObservation(
        position=np.array([
            target_xy[0] + rng.normal(0.0, 0.12),
            target_xy[1] + rng.normal(0.0, 0.12),
            0.85,
        ]),
        covariance=np.diag([0.16**2, 0.16**2, 0.25**2]),
        confidence=0.92,
    )
    thermal = PositionObservation(
        position=np.array([
            target_xy[0] + rng.normal(0.0, 0.20),
            target_xy[1] + rng.normal(0.0, 0.20),
            0.82,
        ]),
        covariance=np.diag([0.24**2, 0.24**2, 0.30**2]),
        confidence=0.84,
    )
    audio = AudioObservation(
        bearing_rad=wrap_angle(
            true_bearing - robot_yaw + rng.normal(0.0, np.deg2rad(3.0))
        ),
        std_rad=np.deg2rad(9.0),
        confidence=0.90,
        keyword_detected=True,
    )
    radar = RadarObservation(
        range_m=true_range + rng.normal(0.0, 0.20),
        range_std_m=0.35,
        confidence=0.85,
        human_present=True,
    )
    return visible, thermal, audio, radar


def run_movement_simulation(steps=6, seed=7):
    """Run a repeatable robot/target movement scenario and retain map history."""
    rng = np.random.default_rng(seed)
    belief_filter = BeliefMapFilter(diffusion_amount=0.04)
    robot_positions = [
        np.array([1.0 + 0.35 * step, 1.0 + 0.15 * step])
        for step in range(steps)
    ]
    target_positions = [
        np.array([4.3 + 0.18 * step, 3.2 + 0.12 * step])
        for step in range(steps)
    ]
    history = []
    for step, (robot_xy, target_xy) in enumerate(
        zip(robot_positions, target_positions)
    ):
        robot_yaw = np.arctan2(0.15, 0.35)
        visible, thermal, audio, radar = make_moving_observations(
            target_xy, robot_xy, robot_yaw, rng
        )
        pose = RobotPose(
            xy=robot_xy,
            yaw_rad=robot_yaw,
            timestamp_s=float(step),
        )
        target_velocity = np.array([0.18, 0.12]) if step else None
        belief, sensors_used, _ = belief_filter.update_from_observations(
            robot_pose=pose,
            visible=visible,
            thermal=thermal,
            audio=audio,
            radar=radar,
            target_velocity_xy=target_velocity,
        )
        estimate = belief_filter.estimate()
        history.append({
            "step": step,
            "robot_xy": robot_xy.copy(),
            "target_xy": target_xy.copy(),
            "belief": belief,
            "estimate": estimate,
            "sensors_used": sensors_used,
        })
    return history


def main():
    import matplotlib.pyplot as plt

    history = run_movement_simulation()
    print("Movement simulation:")
    for item in history:
        x, y, probability = item["estimate"]
        print(
            f"  step {item['step']}: robot={np.round(item['robot_xy'], 2)}, "
            f"target={np.round(item['target_xy'], 2)}, "
            f"estimate=({x:.2f}, {y:.2f}), p={probability:.6f}"
        )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, item in zip(axes.flat, history):
        estimate_x, estimate_y, _ = item["estimate"]
        im = ax.imshow(
            item["belief"],
            origin="lower",
            extent=[MAP_MIN_X, MAP_MAX_X, MAP_MIN_Y, MAP_MAX_Y],
            aspect="equal",
        )
        ax.scatter(*item["robot_xy"], marker="s", label="Robot")
        ax.scatter(*item["target_xy"], marker="o", label="True target")
        ax.scatter(estimate_x, estimate_y, marker="x",
                   label="Belief maximum")
        ax.set_title(f"Step {item['step']}")
        ax.set_xlabel("Map x (m)")
        ax.set_ylabel("Map y (m)")
        ax.legend(loc="upper right", fontsize="small")
        fig.colorbar(im, ax=ax)

    fig.suptitle("Belief-map evolution while robot and target move")
    plt.show()


if __name__ == "__main__":
    main()
