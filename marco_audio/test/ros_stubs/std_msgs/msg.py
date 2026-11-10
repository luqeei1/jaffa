"""Stand-ins for std_msgs/Bool and std_msgs/String, with the type checks rclpy applies."""


class Bool:
    def __init__(self, data=False):
        self.data = data

    def check(self):
        assert isinstance(self.data, bool), type(self.data)


class String:
    def __init__(self, data=""):
        self.data = data

    def check(self):
        assert isinstance(self.data, str), type(self.data)
