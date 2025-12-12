import numpy as np
from librosa import pyin, note_to_hz
import phoneme, typemes
from config import metric_config as config

def f0_heuristic(y: np.ndarray, sr: int, method: str | None) -> float:
    """
    Estiamtes the fundmental frequency differential in an
    audiio segment via some method. Supply one of the following
    for method:
      "upspeak_coarse":
        last and first voiced f0 difference
      "upspeak_fifths":
        average voiced f0 differences for last and first fifths
        of audio duration
      "peak_to_peak":
        maximum and minimum voiced f0 difference
      None:
        Uses default metric specified in config.py::f0_heuristic
    """
    if method == None:
        method = config['f0_heuristic']

    if method == 'upspeak_coarse':
        return upspeak_coarse(y, sr)
    elif method == 'upspeak_fifths':
        return upspeak_fifths(y, sr)
    elif method == 'peak_to_peak':
        return peak_to_peak(y, sr)
    else:
        print(f"Error: invalid method supplied")
        return -1

def f0_estimate(y, sr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper for librosa.pyin Fundamental Frequency estimation.

    The YIN algorithm uses signal minima to estimate period
    and parabolically interpolates periods before computing
    f0.

    The pYIN algorithm then performs Viterbi decoding on YIN
    f0 estimates and their probabilities to estimate the
    most likely f0 sequence.
    """
    return pyin(
        y=y,
        fmin=note_to_hz('C2'),
        fmax=note_to_hz('E4'),
        sr=sr,
        fill_na=None,
        pad_mode='symmetric',
        # frame_length = int(np.min([2048, len(y)]))
    )

def upspeak_coarse(y, sr) -> float:
    """
    Coarsely estimates the change in fundamental frequency
    in an audio segment by finding the difference between the
    last and first voiced f0 estimates.
    """
    (f0, voiced_flag, voiced_prob) = f0_estimate(y, sr)

    # f0 = f0[np.where(voiced_flag == True)]
    # f0 = f0[np.where(~np.isnan(f0))]
    if len(f0) < 2: # segment too short for upspeak info
        return float('inf')
    return float(f0[-1] - f0[0])

def upspeak_fifths(y, sr) -> float:
    """
    Estimates the change in fundamental frequency in an audio
    segment by finding the difference between the average
    voiced f0 estimates in the first and last fifth of the
    voiced f0 estimates.
    """
    (f0, voiced_flag, voiced_prob) = f0_estimate(y, sr)

    # f0 = f0[np.where(voiced_flag == True)]
    # f0 = f0[np.where(~np.isnan(f0))]
    if len(f0) < 2: # segment too short for upspeak info
        return float('inf')
    n = int(np.ceil(len(f0) / 5))
    return float(np.average(f0[-n:]) - np.average(f0[:n]))

def peak_to_peak(y, sr) -> float:
    """
    Finds the peak-to-peak differential in maximum and minimum
    voiced f0 estimates for an audio signal.
    """
    (f0, voiced_flag, voiced_prob) = f0_estimate(y, sr)

    # f0 = f0[np.where(voiced_flag == True)]
    # f0 = f0[np.where(~np.isnan(f0))]
    if len(f0) < 2: # segment too short for upspeak info
        return float('inf')
    return float(np.max(f0) - np.min(f0))