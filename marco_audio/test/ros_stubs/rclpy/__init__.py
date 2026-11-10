"""Minimal stand-in for rclpy, for testing node logic without ROS 2 installed."""
_ok = False
OVERRIDES = {}
CLOCK = {"ns": 1_000_000_000_000}
def init(args=None):
    global _ok; _ok = True
def ok(): return _ok
def shutdown():
    global _ok; _ok = False
def spin(node): raise KeyboardInterrupt
