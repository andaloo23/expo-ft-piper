"""Hardware-free fakes for contract tests."""

from __future__ import annotations

import numpy as np


class FakeEnv:
    """Implements the five-operation env contract with the Piper observation
    shapes, no hardware. Used by the WebSocket round-trip test."""

    def __init__(self, language_instruction="pick up the cube", auto_reset_steps=5, **kwargs):
        self.language_instruction = language_instruction
        self.auto_reset_steps = auto_reset_steps
        self.control_hz = kwargs.get("control_hz", 10)
        self._steps = 0
        self.last_action = None

    def reset(self):
        self._steps = 0
        return self.get_observation()

    def get_observation(self):
        img = np.zeros((180, 320, 3), dtype=np.uint8)
        return {
            "exterior_image_1_left": img,
            "exterior_image_2_left": img,
            "wrist_image_left": img,
            "cartesian_position": np.zeros(6, dtype=np.float32),
            "gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": self.language_instruction,
        }

    def step(self, action):
        self._steps += 1
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.last_action = a
        # Fake saturation: the executed action is half the request.
        return {"executed_action": a * 0.5}

    def get_info_for_step(self):
        done = self._steps >= self.auto_reset_steps
        success = False
        reward = 0.0
        mask = 0.0 if done else 1.0
        return done, success, reward, mask

    def get_human_override_action(self):
        return None, False

    def close(self):
        pass
