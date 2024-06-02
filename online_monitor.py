import torch
import numpy as np

class Monitor:
    def __init__(self):
        pass

    def check_for_nans(self, name, value):
        if isinstance(value, dict):
            for key, val in value.items():
                self.check_for_nans(f"{name}.{key}", val)
        elif isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"NaN detected in {name}: {value}")
                raise ValueError(f"NaN detected in {name}")
        elif isinstance(value, np.ndarray):
            if np.isnan(value).any():
                print(f"NaN detected in {name}: {value}")
                raise ValueError(f"NaN detected in {name}")
        else:
            if np.isnan(np.array(value)).any():
                print(f"NaN detected in {name}: {value}")
                raise ValueError(f"NaN detected in {name}")

    def log(self, name, value):
        print(f"{name}: {value}")

    def check_policy_parameters(self, policy):
        for name, param in policy.named_parameters():
            self.check_for_nans(f"policy parameter {name}", param)

    def check_state(self, state):
        self.check_for_nans("state", state)

    def check_actions(self, actions):
        self.check_for_nans("actions", actions)

    def check_intermediate_values(self, name, value):
        self.check_for_nans(name, value)
        self.log(name, value)
