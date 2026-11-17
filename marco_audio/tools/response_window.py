"""One Marco playback -> one timed Polo reply. No belief is reset here."""
import threading


class ResponseWindow:
    def __init__(self, seconds=10.0, hold=0.3, tail=1.1, grace=3.0):
        self.seconds, self.hold, self.tail, self.grace = seconds, hold, tail, grace
        self.lock = threading.Lock()
        self.generation = 0
        self.state = 'idle'
        self.message = 'Press M to call Marco'
        self.start = self.end = None

    def request(self):
        with self.lock:
            if self.state != 'idle':
                return None
            self.generation += 1
            self.state, self.message = 'speaking', 'Playing Marco; recognition muted'
            self.start = self.end = None
            return self.generation

    def current(self, generation):
        with self.lock:
            return self.generation == generation and self.state != 'idle'

    def is_idle(self):
        with self.lock:
            return self.state == 'idle'

    def played(self, generation, now):
        with self.lock:
            if generation != self.generation or self.state != 'speaking':
                return
            self.start = now + self.hold
            self.end = self.start + self.seconds
            self.state = 'listening'
            self.message = 'Waiting for one Polo'

    def capture_id(self, stamp):
        with self.lock:
            if self.state == 'idle':
                return -1  # noise-floor calibration only; no word recognition
            if self.state == 'listening' and self.start <= stamp <= self.end + self.tail:
                return self.generation
            return None

    def accept(self, generation, start, end, now):
        with self.lock:
            if (generation != self.generation or self.state != 'listening'
                    or not self.start <= start < end <= self.end
                    or now > self.end + self.tail + self.grace):
                return False
            self.state, self.message = 'idle', 'Polo accepted; belief retained. M = next Marco'
            return True

    def cancel(self, message='Round cancelled; M = next Marco'):
        with self.lock:
            self.generation += 1
            self.state, self.message = 'idle', message

    def failed(self, generation, message):
        with self.lock:
            if generation == self.generation:
                self.state, self.message = 'idle', message

    def status(self, now):
        with self.lock:
            if self.state == 'listening':
                if now < self.start:
                    return 'Playback finished; waiting for echoes to settle'
                if now <= self.end:
                    return f'Listening for one Polo: {self.end - now:.1f}s'
                if now <= self.end + self.tail + self.grace:
                    return 'Finishing recognition of the reply'
                self.state, self.message = 'idle', 'No usable Polo; belief retained. M = next Marco'
            return self.message


def play_clip(path, output_device, cancelled):
    """Portable WAV playback on the selected sounddevice output; returns after drain.

    Uses the same named-clip idea as speech_node; native Windows does not need ROS/aplay.
    """
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd
    import sounddevice as sd
    rate, samples = wavfile.read(path)
    if samples.dtype.kind in 'iu':
        if samples.dtype == np.uint8:
            samples = (samples.astype(np.float32) - 128) / 128
        else:
            samples = samples.astype(np.float32) / (np.iinfo(samples.dtype).max + 1)
    else:
        samples = samples.astype(np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Marco clip must contain finite mono/stereo audio')
    if len(samples) / rate > 30:
        raise ValueError('Marco clip must be at most 30 seconds')
    device = sd.query_devices(output_device, 'output')
    output_rate = int(device['default_samplerate'])
    if rate != output_rate:
        divisor = gcd(rate, output_rate)
        samples = resample_poly(samples, output_rate // divisor, rate // divisor, axis=0)
    samples = np.clip(samples, -1, 1).astype(np.float32)
    with sd.OutputStream(device=output_device, samplerate=output_rate,
                         channels=samples.shape[1], dtype='float32') as stream:
        for start in range(0, len(samples), 1024):
            if cancelled():
                stream.abort()
                return
            if stream.write(samples[start:start + 1024]):
                raise RuntimeError('Marco playback underflow; retry the round')


def find_output(sd, input_device):
    """Prefer the same ReSpeaker/host API; never silently play on laptop speakers."""
    source = sd.query_devices(input_device)
    for index, device in enumerate(sd.query_devices()):
        if (device['max_output_channels'] > 0 and device['hostapi'] == source['hostapi']
                and any(word in device['name'].lower() for word in ('respeaker', 'xvf3800'))):
            return index
    raise ValueError('No ReSpeaker output found; specify --output-device INDEX')
