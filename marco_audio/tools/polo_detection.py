"""Timestamp-matched Polo recognition and Ahmad's raw-microphone direction fit."""
import json
import math

import numpy as np
import doa_phat as dp

FS = dp.FS


def speech_bearing(raw, min_confidence=0.15):
    """Combine consistent 40 ms speech windows, not the 7 ms clap-onset window."""
    estimates = []
    for start in range(0, len(raw) - 639, 320):
        window = raw[start:start + 640]
        if not np.all(np.isfinite(window)) or dp.level_dbfs(window) < -85:
            continue
        pairs = dp.gcc_phat_pairs(window)
        fit = dp.robust_fit(pairs.tau, pairs.peak)
        result = dp.gcc_event_fields(fit, pairs)
        if result['confidence'] > 0 and result['confidence'] >= min_confidence:
            estimates.append(result)
    if len(estimates) < 2:
        return None
    angles = np.array([e['azimuth_rad'] for e in estimates])
    weights = np.array([e['confidence'] for e in estimates])
    # Require one compact group containing 80% of accepted directional evidence.
    # A mean between two separated directions is not a measured source direction.
    separation = np.abs(np.angle(np.exp(1j * (angles[:, None] - angles[None, :]))))
    groups = separation <= math.radians(20)
    support = groups @ weights
    selected = groups[np.argmax(support)]
    fraction = float(support.max() / weights.sum())
    if fraction < 0.8:
        return None
    estimates = [e for e, keep in zip(estimates, selected) if keep]
    angles, weights = angles[selected], weights[selected]
    if len(angles) < 2:
        return None
    vector = np.sum(weights * np.exp(1j * angles)) / weights.sum()
    agreement = abs(vector)
    # Opposing sources/reflections must not turn into an invented middle direction.
    if agreement < 0.65:
        return None
    spread = math.sqrt(max(0, -2 * math.log(min(1, agreement))))
    std = max(math.radians(10), spread,
              float(np.average([e['azimuth_std_rad'] for e in estimates], weights=weights)))
    confidence = float(weights.mean() * agreement * fraction)
    if confidence < min_confidence:
        return None
    return dict(azimuth=float(np.angle(vector)), std=std, confidence=confidence)


class PoloDetector:
    """Recognise processed ch0/ch1; localise exactly that word on raw ch2..5.

    All timestamps use one audio sample clock. Final results only: a partial
    hypothesis and the second processed channel cannot update the map twice.
    """
    def __init__(self, model, speech_threshold=0.35, bearing_threshold=0.15, debug=False,
                 processed_delay_ms=0.0):
        from vosk import KaldiRecognizer
        self.recognizers = [KaldiRecognizer(model, FS, json.dumps(['polo', '[unk]']))
                            for _ in range(2)]
        for recognizer in self.recognizers:
            recognizer.SetWords(True)
        self.buffer = np.zeros((20 * FS, 6), dtype=np.float32)
        self.n = 0
        self.seen = []
        self.speech_threshold = speech_threshold
        self.bearing_threshold = bearing_threshold
        self.debug = debug
        self.processed_delay_samples = round(processed_delay_ms * FS / 1000)

    def process_result(self, result, channel):
        if self.debug:
            print(f"Speech ch{channel}: {result.get('text', '') or '[no words]'}", flush=True)
        events = []
        for word in result.get('result', []):
            if word.get('word', '').lower() != 'polo':
                continue
            confidence = float(word.get('conf', 0))
            if confidence < self.speech_threshold:
                print(f"Polo below speech threshold ({confidence:.2f}).", flush=True)
                continue
            delay = getattr(self, 'processed_delay_samples', 0)
            a, b = int(word['start'] * FS) - delay, int(word['end'] * FS) - delay
            if any(a < end + 0.15 * FS and b > start - 0.15 * FS
                   for start, end in self.seen):
                continue
            if a < max(0, self.n - len(self.buffer)) or b > self.n or b <= a:
                print('Polo audio expired or incomplete; skipping its bearing.', flush=True)
                continue
            raw = self.buffer[np.arange(a, b) % len(self.buffer), 2:6]
            bearing = speech_bearing(raw, self.bearing_threshold)
            if bearing is None:
                print(f'Heard Polo ({confidence:.2f}); raw microphones disagree on direction.', flush=True)
                continue
            self.seen.append((a, b))
            self.seen = [(s, e) for s, e in self.seen if e > self.n - len(self.buffer)]
            events.append({**bearing, 'speech_confidence': confidence, 'start': a / FS,
                           'end': b / FS, 'keyword': 'polo'})
        return events

    def feed(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != 6 or not np.all(np.isfinite(samples)):
            raise ValueError('Expected finite six-channel audio')
        if len(samples) > len(self.buffer):
            raise ValueError('Feed audio in blocks shorter than 20 seconds')
        self.buffer[(self.n + np.arange(len(samples))) % len(self.buffer)] = samples
        self.n += len(samples)
        events = []
        for channel, recognizer in enumerate(self.recognizers):
            pcm = (np.clip(samples[:, channel], -1, 1) * 32767).astype('<i2').tobytes()
            if recognizer.AcceptWaveform(pcm):
                events.extend(self.process_result(json.loads(recognizer.Result()), channel))
        return events

    def finish(self):
        events = []
        for channel, recognizer in enumerate(self.recognizers):
            events.extend(self.process_result(json.loads(recognizer.FinalResult()), channel))
        return events
