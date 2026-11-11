class Node:
    """Records what it was asked to start, and refuses values the real Node would refuse."""
    def __init__(self, *, executable, package=None, name=None, namespace=None, parameters=None,
                 remappings=None, arguments=None, output="log"):
        assert isinstance(executable, str) and isinstance(package, str)
        for group in parameters or []:
            assert isinstance(group, dict)
            for key, value in group.items():
                assert isinstance(key, str)
                assert isinstance(value, (str, int, float, bool)), f"{key}: {type(value).__name__}"
        assert all(isinstance(a, str) for a in arguments or []), "arguments must be text"
        assert output in ("screen", "log", "both", "own_log")
        self.package, self.executable, self.name = package, executable, name
        self.parameters, self.arguments, self.output = parameters or [], list(arguments or []), output
