class LaunchConfiguration:
    def __init__(self, variable_name, *, default=None):
        self.variable_name, self.default = variable_name, default

    def perform(self, context):
        if self.variable_name not in context.launch_configurations:
            if self.default is None:
                raise KeyError(f"launch configuration '{self.variable_name}' does not exist")
            return self.default
        return context.launch_configurations[self.variable_name]
