"""One Polo, three minutes of silence, then a simulated RGB-D position.

Run from repo root: python marco_audio/tools/simulate_sparse_polo.py
Optional --speech-wav: mono int16 recording of exactly one Polo (16 kHz or resampled).
The microphone delays and RGB-D observation are simulated, not hardware measurements.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sensor_fusion.sensor_fusion import (
    AudioObservation, BeliefMapFilter, audio_likelihood, audio_sector_probability,
    normalise, visual_likelihood, wrap_angle, XX, YY, xs, ys,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speech-wav', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'marco_audio/sim/out/sparse_polo.png')
    args = parser.parse_args()
    mic = (4., 3.)
    bearing = math.radians(35)
    std = math.radians(10)
    speech_score, direction_score = .9, .8
    events = []
    if args.speech_wav:
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        from vosk import Model, SetLogLevel
        from polo_detection import PoloDetector
        from test_polo_detection import array_audio
        rate, speech = wavfile.read(args.speech_wav)
        if speech.ndim != 1 or speech.dtype != np.int16:
            raise ValueError('Expected mono int16 speech WAV')
        speech = resample_poly(speech.astype(float) / 32768, 16000, rate)
        audio = array_audio(np.pad(speech, (8000, 32000)), bearing)
        SetLogLevel(-1)
        detector = PoloDetector(Model(str(ROOT / 'vosk-model-small-en-us-0.15')))
        for start in range(0, len(audio), 320):
            events.extend(detector.feed(audio[start:start + 320]))
        assert len(events) == 1, f'Expected one recognized Polo, got {len(events)}'
        event = events[0]
        bearing, std = event['azimuth'], event['std']
        speech_score, direction_score = event['speech_confidence'], event['confidence']
    observation = AudioObservation(bearing, std, direction_score, True)
    q = min(.8, speech_score * direction_score)  # identical to live viewer
    fusion = BeliefMapFilter(diffusion_amount=0)
    angles = wrap_angle(np.arctan2(YY - mic[1], XX - mic[0]) - bearing)
    gaussian = np.exp(-.5 * (angles / std)**2)
    angular_mean = std / math.sqrt(2 * math.pi) * math.erf(math.pi / (math.sqrt(2) * std))
    gaussian[np.hypot(XX - mic[0], YY - mic[1]) < .15] = angular_mean
    old = normalise((1 - q) + q * gaussian)
    likelihood, used = audio_likelihood(observation, robot_xy=mic, reliability=q, min_confidence=.15)
    assert used
    single, _ = fusion.update(audio=likelihood)
    # No observation means no update, not multiplication by the same likelihood again.
    if args.speech_wav:
        silence_events = []
        for _ in range(180 * 5):
            silence_events.extend(detector.feed(np.zeros((3200, 6), np.float32)))
        silence_events.extend(detector.finish())
        assert not silence_events, 'Silence must not create additional Polo updates'
    retained = fusion.belief.copy()
    np.testing.assert_array_equal(retained, single)
    target = np.array([mic[0] + 2.5 * math.cos(bearing), mic[1] + 2.5 * math.sin(bearing), 1.0])
    visual = visual_likelihood(target, np.diag([.25**2, .25**2, .3**2]))
    combined, _ = fusion.update(visual=visual)  # audio is already in the prior: do not reapply it
    np.testing.assert_allclose(combined, normalise(retained * visual))
    peak = np.unravel_index(np.argmax(combined), combined.shape)
    assert math.hypot(XX[peak] - target[0], YY[peak] - target[1]) < .15
    stats = dict(simulation=True, recognized_polos=len(events) if args.speech_wav else None,
                 silence_seconds=180, map_updates_from_audio=1, weight=q,
                 old_sector_mass=audio_sector_probability(old, observation, mic),
                 single_sector_mass=audio_sector_probability(single, observation, mic),
                 after_silence_sector_mass=audio_sector_probability(retained, observation, mic),
                 simulated_rgbd_peak_xy=[float(XX[peak]), float(YY[peak])])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    fig.get_layout_engine().set(h_pad=.12, hspace=.12)
    for ax, grid, title in zip(axes.flat, (old, single, retained, combined),
                               ('Previous weighting: one Polo', 'New weighting: one Polo',
                                'Same belief after 3 minutes of silence', 'Later simulated RGB-D update')):
        im = ax.imshow(grid * grid.size, origin='lower', extent=(xs[0], xs[-1], ys[0], ys[-1]), vmin=0, vmax=20)
        ax.plot(*mic, 'w+', markersize=12)
        ax.plot([mic[0], mic[0] + 3 * math.cos(bearing)], [mic[1], mic[1] + 3 * math.sin(bearing)], 'w--', lw=1)
        mass = audio_sector_probability(grid, observation, mic)
        ax.set(title=f'{title}\nModel mass in ±{math.degrees(2*std):.0f}° sector: {mass:.1%}', xlabel='Map X (m)', ylabel='Map Y (m)')
    fig.colorbar(im, ax=axes, extend='max', label='Belief / uniform prior (fixed scale)')
    fig.suptitle('Simulation: one accepted Polo, no repetition\nAudio alone gives direction; RGB-D supplies position')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))
    print('Saved', args.output)


if __name__ == '__main__':
    main()
