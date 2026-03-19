import numpy as np


class Signal:
    def __init__(self, file_name, time_vector, x_signal, y_signal=None, metadata=None):
        self.file_name = file_name
        self.time = np.array(time_vector)
        self.x = np.array(x_signal)
        self.y = np.array(y_signal) if y_signal is not None else None
        self.metadata = metadata if metadata is not None else {}