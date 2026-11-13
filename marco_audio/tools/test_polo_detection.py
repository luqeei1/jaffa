"""Run from repo root: python marco_audio/tools/test_polo_detection.py [--speech-wav PATH]."""
import argparse
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import doa_phat as dp
from polo_detection import PoloDetector, speech_bearing

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sensor_fusion.sensor_fusion import AudioObservation, BeliefMapFilter, audio_likelihood, normalise


def array_audio(signal, angle):
    """Ideal plane wave; processed channels keep the undelayed speech."""
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(n, 1 / dp.FS)
    direction = np.array([math.cos(angle), math.sin(angle)])
    delays = -dp.MIC_XY @ direction / dp.SPEED_OF_SOUND
    raw = [np.fft.irfft(spectrum * np.exp(-2j * np.pi * freq * t), n) for t in delays]
    return np.column_stack([signal, signal, *raw]).astype(np.float32)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.signal = np.random.default_rng(8).normal(0, 0.05, 16000)

    def test_stale_future_reset_and_out_of_order_rejected(self):
        from audio_belief_live import fresh_capture
        self.assertTrue(fresh_capture(99, 100, 90, 3, 98))
        for captured in (89, 96, 98, 101, float('nan')):
            self.assertFalse(fresh_capture(captured, 100, 90, 3, 98))

    def test_capture_pose_interpolation_and_missing_coverage(self):
        import tempfile
        from sensor_fusion.pose_replay import PoseReplay
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'poses.csv'
            path.write_text('stamp_ms,x_m,y_m,yaw_rad\n1000,1,2,0\n1200,3,2,1.5707963267948966\n')
            poses = PoseReplay(path, 250)
            pose = poses.base_to_map(1100)
            np.testing.assert_allclose(pose[:2, 3], [2, 2])
            self.assertAlmostEqual(math.atan2(pose[1, 0], pose[0, 0]), math.pi / 4)
            self.assertIsNone(poses.base_to_map(1300))
            self.assertIsNone(PoseReplay(path, 100).base_to_map(1100))

    def test_bearings_all_around_board(self):
        for degrees in range(-180, 180, 30):
            expected = math.radians(degrees)
            result = speech_bearing(array_audio(self.signal, expected)[:, 2:])
            self.assertIsNotNone(result)
            self.assertLess(dp.angle_error_deg(result['azimuth'], expected), 2)

    def test_silence_and_unrelated_microphones_rejected(self):
        self.assertIsNone(speech_bearing(np.zeros((16000, 4))))
        self.assertIsNone(speech_bearing(np.random.default_rng(3).normal(0, .05, (16000, 4))))

    def test_separated_bearings_are_not_averaged(self):
        raw = np.concatenate([array_audio(self.signal, math.radians(a))[:, 2:] for a in (-35, 35)])
        self.assertIsNone(speech_bearing(raw))
        raw = np.concatenate([array_audio(self.signal, math.radians(a))[:, 2:] for a in (-179, 179)])
        result = speech_bearing(raw)
        self.assertIsNotNone(result)
        self.assertLess(dp.angle_error_deg(result['azimuth'], math.pi), 3)

    def test_age_during_silence_and_recover_new_direction(self):
        f = BeliefMapFilter(diffusion_amount=0)
        def likelihood(degrees):
            return audio_likelihood(AudioObservation(math.radians(degrees), math.radians(10), .8, True),
                                    robot_xy=(4, 3), reliability=.8)[0]
        for _ in range(20):
            f.update(audio=likelihood(90))
        prior = f.belief.copy()
        f.age(5, half_life_s=5)
        np.testing.assert_allclose(f.belief, .5 * prior + .5 / prior.size)
        for _ in range(5):
            f.age(1, half_life_s=5)
            f.update(audio=likelihood(-90))
        self.assertGreater(f.belief[:60].sum(), f.belief[61:].sum())
        self.assertAlmostEqual(f.belief.sum(), 1)

    def test_word_timestamp_duplicate_unknown_and_stale(self):
        detector = PoloDetector.__new__(PoloDetector)
        detector.buffer = array_audio(self.signal, math.pi / 2)
        detector.n = len(self.signal)
        detector.seen = []
        detector.debug = False
        detector.speech_threshold = .35
        detector.bearing_threshold = .15
        word = dict(word='polo', start=.1, end=.7, conf=.47)
        result = dict(result=[word])
        self.assertEqual(len(detector.process_result(result, 0)), 1)
        self.assertEqual(detector.process_result(result, 1), [])
        detector.seen = []
        self.assertEqual(detector.process_result(dict(result=[dict(word, word='[unk]')]), 0), [])
        self.assertEqual(detector.process_result(dict(result=[dict(word, conf=.1)]), 0), [])
        detector.n += 2 * dp.FS
        self.assertEqual(detector.process_result(result, 0), [])

    def test_processed_channel_delay_shifts_raw_window(self):
        from unittest.mock import patch
        detector = PoloDetector.__new__(PoloDetector)
        detector.buffer = array_audio(self.signal, 0)
        detector.n = len(self.signal)
        detector.seen = []
        detector.debug = False
        detector.speech_threshold, detector.bearing_threshold = .35, .15
        detector.processed_delay_samples = 160
        with patch('polo_detection.speech_bearing', return_value=dict(azimuth=0, std=.2, confidence=.8)) as bearing:
            events = detector.process_result(dict(result=[dict(word='polo', start=.1, end=.7, conf=.8)]), 0)
            np.testing.assert_array_equal(bearing.call_args.args[0], detector.buffer[1440:11040, 2:6])
            self.assertAlmostEqual(events[0]['start'], .09)

    def test_low_confidence_updates_accumulate_and_rotate(self):
        obs = AudioObservation(0, math.radians(15), .2, True)
        likelihood, used = audio_likelihood(obs, robot_xy=(4, 3), robot_yaw=math.pi / 2,
                                            reliability=.1, min_confidence=.15)
        self.assertTrue(used)
        fusion = BeliefMapFilter(diffusion_amount=0)
        first, _ = fusion.update(audio=likelihood)
        second, _ = fusion.update(audio=likelihood)
        np.testing.assert_allclose(second, normalise(first * likelihood))
        self.assertFalse(np.allclose(first, second))
        self.assertAlmostEqual(second.sum(), 1)
        self.assertGreater(second[80, 80], second[40, 80])


def check_speech(path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from vosk import Model, SetLogLevel
    fs, signal = wavfile.read(path)
    if signal.ndim != 1 or signal.dtype != np.int16:
        raise ValueError('Speech check expects a mono int16 WAV')
    signal = resample_poly(signal.astype(float) / 32768, dp.FS, fs)
    audio = array_audio(np.pad(signal, (8000, 32000)), math.pi / 2)
    SetLogLevel(-1)
    detector = PoloDetector(Model(str(ROOT / 'vosk-model-small-en-us-0.15')), debug=True)
    events = []
    for start in range(0, len(audio), 320):
        events.extend(detector.feed(audio[start:start + 320]))
    events.extend(detector.finish())
    assert len(events) == 3, f'Expected 3 Polos in test phrase; got {events}'
    for event in events:
        assert dp.angle_error_deg(event['azimuth'], math.pi / 2) < 3, event
    print('PASS: real speech recognizer, three Polos, no duplicate updates, +90 degree bearings')
    check_viewer(audio)


def check_viewer(audio):
    """Exercise the actual worker and Vosk while plot processing sleeps 100 ms."""
    import contextlib
    import io
    import threading
    import time
    from types import SimpleNamespace
    from unittest.mock import patch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import audio_belief_live as viewer

    done = threading.Event()
    class Stream:
        def __init__(self, **kwargs):
            self.callback = kwargs['callback']

        def __enter__(self):
            def feed():
                epoch = time.time()
                pacing_start = time.monotonic()
                for start in range(0, len(audio), 320):
                    time.sleep(max(0, pacing_start + start / dp.FS - time.monotonic()))
                    block = audio[start:start + 320]
                    self.callback(block, len(block), SimpleNamespace(currentTime=time.time(),
                                  inputBufferAdcTime=epoch + start / dp.FS), SimpleNamespace(input_overflow=False))
                time.sleep(2)
                done.set()
            self.thread = threading.Thread(target=feed)
            self.thread.start()
            return self

        def __exit__(self, *args):
            self.thread.join()

    def pause(_):
        time.sleep(.1)
        if done.is_set():
            fig = plt.gcf()
            ax = fig.axes[0]
            belief = np.asarray(ax.images[0].get_array())
            assert 'Polo' in ax.get_title() and '+90' in ax.get_title()
            assert abs(belief.mean() - 1) < 1e-8 and belief.max() > belief.min()
            assert ax.images[0].get_clim() == (0, 20)
            fig.canvas.draw()
            plt.close(fig)

    output = io.StringIO()
    with contextlib.redirect_stdout(output), patch('sys.argv', ['viewer', '--device', '14']), \
            patch('sounddevice.InputStream', Stream), patch('matplotlib.pyplot.pause', pause), \
            patch('matplotlib.pyplot.show'):
        viewer.main()
    log = output.getvalue()
    assert log.count('raw board bearing') == 3, log
    assert 'Audio gap' not in log, log
    print('PASS: Vosk worker + slow plot: three belief updates, +90 degrees, no audio gaps')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speech-wav', type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PipelineTests))
    if not result.wasSuccessful():
        sys.exit(1)
    if args.speech_wav:
        check_speech(args.speech_wav)
