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
                 processed_delay_ms=0.0, spotter=None):
        self.spotter = spotter
        self.pending = []
        if spotter is None:
            from vosk import KaldiRecognizer
            self.recognizers = [KaldiRecognizer(model, FS, json.dumps(['polo', '[unk]']))
                                for _ in range(2)]
        else:
            from live_doa import Listener, Labels
            self.listener = Listener(Labels())
            self.recognizers = []
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
        if self.spotter is not None:
            self.pending.extend(self.listener.feed(samples))
            return self._release_keywords()
        events = []
        for channel, recognizer in enumerate(self.recognizers):
            pcm = (np.clip(samples[:, channel], -1, 1) * 32767).astype('<i2').tobytes()
            if recognizer.AcceptWaveform(pcm):
                events.extend(self.process_result(json.loads(recognizer.Result()), channel))
        return events

    def _release_keywords(self):
        """Use Ahmad's onset gating, normalization, and Marco/Polo vocabulary.

        Accepted word timestamps still select the raw samples for the speech bearing.
        """
        from keyword_spotter import PRE_S, POST_S, EARLY_S, LATE_S
        events = []
        while self.pending and self.n >= self.pending[0]['sample'] + int(POST_S * FS):
            onset = self.pending.pop(0)['sample']
            a, b = max(0, onset - int(PRE_S * FS)), onset + int(POST_S * FS)
            if a < self.n - len(self.buffer):
                continue
            block = self.buffer[np.arange(a, b) % len(self.buffer)]
            answers = [(channel, self.spotter.spot(block[:, channel], (onset - a) / FS))
                       for channel in (1, 0)]
            # Prefer the best accepted Polo, rather than whichever channel is evaluated first.
            answers.sort(key=lambda entry: entry[1]['conf'], reverse=True)
            for channel, answer in answers:
                if self.debug:
                    print(f"Ahmad spotter ch{channel}: {answer['heard'] or '[no words]'}", flush=True)
                if answer['keyword'] != 'polo':
                    continue
                matches = [w for w in answer['found'] if w['word'] == 'polo'
                           and w['conf'] >= self.speech_threshold
                           and -EARLY_S <= w['start'] - (onset - a) / FS <= LATE_S]
                if not matches:
                    continue
                word = min(matches, key=lambda w: abs(w['start'] - (onset - a) / FS))
                event = self.process_result({'result': [dict(word, start=word['start'] + a / FS,
                                                            end=word['end'] + a / FS)]}, channel)
                events.extend(event)
                if event:
                    break
        return events

    def finish(self):
        events = []
        for channel, recognizer in enumerate(self.recognizers):
            events.extend(self.process_result(json.loads(recognizer.FinalResult()), channel))
        return events
