import librosa
import numpy as np

from jamcoder import fade
def spectral_interp(source: np.ndarray, target: np.ndarray, crossfade_length):
    # do an stft
    src_stft = librosa.stft(source[-crossfade_length])
    tgt_stft = librosa.stft(target[crossfade_length])

    # interpolate using fade
    for t in range(crossfade_length):
        prev_A, curr_A = fade(t, crossfade_length)
        breakpoint()

