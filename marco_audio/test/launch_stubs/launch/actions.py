class DeclareLaunchArgument:
    def __init__(self, name, *, default_value=None, description=None, choices=None):
        assert isinstance(name, str) and name
        assert default_value is None or isinstance(default_value, str), "defaults are text"
        assert description is None or isinstance(description, str)
        self.name, self.default_value, self.description = name, default_value, description


class OpaqueFunction:
    def __init__(self, *, function, args=None, kwargs=None):
        assert callable(function)
        self.function, self.args, self.kwargs = function, list(args or []), dict(kwargs or {})

    def execute(self, context):
        out = self.function(context, *self.args, **self.kwargs)
        assert out is None or isinstance(out, list), "must return a list of things to start"
        return out or []


class LogInfo:
    def __init__(self, *, msg):
        assert isinstance(msg, str)
        self.msg = msg
