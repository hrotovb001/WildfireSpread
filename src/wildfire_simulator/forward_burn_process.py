import numpy as np

class ForwardBurnProcess:
    """
    Simulates forward burning up to a given time `t`.

    When called with a state frame (13×H×W) and a burn time `t`, it zeroes
    the fire-mask (channel 8) and arrival-time (channel 9) of every pixel
    whose original arrival time exceeds `t`.
    """
    def __call__(self, frame: np.ndarray, t: float) -> np.ndarray:
        out = frame.copy()
        not_burnt = out[9] > t
        out[8][not_burnt] = 0.0
        out[9][not_burnt] = 0.0
        return out
