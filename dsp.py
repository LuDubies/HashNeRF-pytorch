import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter
from typing import List, Union


def split_bands(buffer_in: npt.NDArray, framerate, freq_borders=None) -> List[npt.NDArray]:
    if freq_borders is None:
        freq_borders = [480, 8200]
    rest = buffer_in
    out = []
    for fb in freq_borders:
        lpf = LowpassFilter(framerate, fb)
        low_frames = lpf.filter(rest)
        out.append(low_frames)
        rest = rest - low_frames
    return out


def shift_buffer(buffer_in: npt.NDArray, shift_amount: int, keep_size: bool = False, fill_value: int = 0) -> npt.NDArray:
    dt = buffer_in.dtype
    if shift_amount < 0:
        if keep_size:
            return np.concatenate((buffer_in[-shift_amount:], np.full(-shift_amount, fill_value, dtype=dt)))
        else:
            return buffer_in[-shift_amount:]
    if keep_size:
        return np.concatenate((np.full(shift_amount, fill_value, dtype=dt), buffer_in[:-shift_amount]))
    else:
        return np.concatenate((np.full(shift_amount, fill_value, dtype=dt), buffer_in))


def apply_gain(buffer_in: npt.NDArray, gain: float) -> npt.NDArray:
    dt = buffer_in.dtype
    mod_buffer = buffer_in * gain
    mod_buffer = np.clip(mod_buffer, np.iinfo(dt).min, np.iinfo(dt).max)
    return mod_buffer.astype(dt)


def add_buffers(buffer_one: npt.NDArray, buffer_two: npt.NDArray) -> npt.NDArray:
    dt = buffer_one.dtype
    buffer_two = buffer_two.astype(dt)
    ldiff = len(buffer_one) - len(buffer_two)
    if ldiff < 0:
        buffer_one = np.concatenate((buffer_one, np.full(-ldiff, 0, dtype=dt)))
    if ldiff > 0:
        buffer_two = np.concatenate((buffer_two, np.full(ldiff, 0, dtype=dt)))
    return buffer_one + buffer_two

def addnbuffers(buffers: List[npt.NDArray]) -> Union[npt.NDArray, None]:
    if len(buffers) == 0:
        return None
    else:
        result = buffers[0]
        for i in range(1, len(buffers)):
            result = add_buffers(result, buffers[i])
        return result


# https://stackoverflow.com/questions/24920346/filtering-a-wav-file-using-python
class LowpassFilter:
    def __init__(self, sample_rate: float, cutoff_frequency: float):
        self.sample_rate = sample_rate
        self.cutoff_frequency = cutoff_frequency
        frequency_ratio = self.cutoff_frequency / self.sample_rate
        self.win_size = int(np.sqrt(0.196201 + frequency_ratio ** 2) / frequency_ratio)

    def filter(self, buffer):
        window = np.ones(self.win_size)
        window *= 1./self.win_size
        return lfilter(window, [1], buffer).astype(buffer.dtype)

    def set_cutoff(self, selfcutoff_frequency: float):
        self.cutoff_frequency = selfcutoff_frequency
        frequency_ratio = self.cutoff_frequency / self.sample_rate
        self.win_size = int(np.sqrt(0.196201 + frequency_ratio ** 2) / frequency_ratio)


def apply_ir(buffer_in: npt.NDArray, ir:npt.NDArray, framerate: int, irlengths: float) -> npt.NDArray:
    dt = buffer_in.dtype
    ir_rate = ir.shape[0] / irlengths
    offset = int(np.floor(framerate / ir_rate))
    base_buffer = np.zeros(buffer_in.size, dtype=dt)
    for index, scale in np.ndenumerate(ir):
        scaled_buffer = apply_gain(buffer_in, scale)
        shifted_buffer = shift_buffer(scaled_buffer, offset * index[0])
        base_buffer = add_buffers(base_buffer, shifted_buffer)

    return base_buffer

def stretch_ir(ir:npt.NDArray, irlength: float, framerate: int) -> npt.NDArray:
    # stretch ir to fit sound framerate
    ir_rate = ir.shape[0] / irlength
    offset = int(np.floor(framerate / ir_rate))
    stretched = np.zeros((ir.shape[0] * offset,))
    for idx, val in np.ndenumerate(ir):
        j = idx[0] * offset
        stretched[j] = val
    return stretched[:-(offset-1)]







