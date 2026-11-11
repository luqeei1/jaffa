"""Stand-in for ROS 2's `launch` module, just enough to load a launch file without ROS 2.
The classes take their arguments the same way the real ones do (most are keyword-only)."""


class LaunchDescription:
    def __init__(self, initial_entities=None):
        self.entities = list(initial_entities or [])


class LaunchContext:
    """Holds the settings typed on the command line (always text), like the real one."""
    def __init__(self, launch_configurations):
        assert all(isinstance(v, str) for v in launch_configurations.values())
        self.launch_configurations = dict(launch_configurations)
