"""Stand-in for the generated SoundEvent class, with the same strict type checks rclpy applies."""
class _Stamp:
    sec = 0; nanosec = 0
class _Header:
    def __init__(self): self.stamp, self.frame_id = _Stamp(), ""
class SoundEvent:
    FLOATS = ("azimuth", "azimuth_std", "confidence", "level_db")
    STRS = ("method", "type", "keyword")
    def __init__(self):
        self.header = _Header()
        for f in self.FLOATS: setattr(self, f, 0.0)
        for s in self.STRS: setattr(self, s, "")
    def check(self):
        for f in self.FLOATS:
            assert isinstance(getattr(self, f), float), (f, type(getattr(self, f)))
            assert abs(getattr(self, f)) < 3.4e38
        for s in self.STRS: assert isinstance(getattr(self, s), str), s
        assert type(self.header.stamp.sec) is int and -2**31 <= self.header.stamp.sec < 2**31
        assert type(self.header.stamp.nanosec) is int and 0 <= self.header.stamp.nanosec < 10**9
        assert isinstance(self.header.frame_id, str)
