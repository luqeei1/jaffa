import rclpy
class _P:
    def __init__(self, v): self.value = v
class _Now:
    @property
    def nanoseconds(self): return rclpy.CLOCK["ns"]
class _Clock:
    def now(self): return _Now()
class _Log:
    def __init__(self): self.lines = []
    def info(self, m): self.lines.append(("info", m))
    def warning(self, m): self.lines.append(("warn", m))
    def error(self, m): self.lines.append(("error", m))
class _Pub:
    def __init__(self, typ, topic): self.typ, self.topic, self.sent = typ, topic, []
    def publish(self, m):
        assert isinstance(m, self.typ); m.check(); self.sent.append(m)
class _Timer:
    def __init__(self, period, cb): self.period, self.cb, self.cancelled = period, cb, False
    def cancel(self): self.cancelled = True
class Node:
    def __init__(self, name):
        self.name, self._log, self.pubs, self.subs, self.timers, self.declared = name, _Log(), [], {}, [], {}
    def declare_parameter(self, name, value=None, descriptor=None, ignore_override=False):
        assert descriptor is not None and descriptor.dynamic_typing
        v = rclpy.OVERRIDES.get(name, value); self.declared[name] = v; return _P(v)
    def create_publisher(self, typ, topic, qos):
        p = _Pub(typ, topic); self.pubs.append(p); return p
    def create_subscription(self, typ, topic, cb, qos):
        self.subs[topic] = (typ, cb)
    def create_timer(self, period, cb):
        t = _Timer(period, cb); self.timers.append(t); return t
    def get_clock(self): return _Clock()
    def get_logger(self): return self._log
    def destroy_node(self): pass
