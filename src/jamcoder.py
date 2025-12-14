import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import pickle
from scipy.io.wavfile import write
from g2p_en import G2p
import nltk

from dataloader import PhonemeLoader, PhonemeMemLoader, strip_stress
from phoneme import Phoneme, PhonemeInstance
from typemes import Typeme, standard_typeme_tree
from choose import choose_phoneme
from config import synth_config as config

def naive_synthesis(voice: PhonemeLoader, typemes: Typeme, target_text: str, method: str) \
        -> tuple[list[str], list[PhonemeInstance]]:
    """
    Returns a list of optimal source phoneme instances for
    the synthesis of the passed target text.
    """
    g2p = G2p()
    target_phonemes: list[str] = g2p(target_text)
    target_phonemes = ['' if p in typemes['silence'].child_names() else p for p in target_phonemes]
    # target_phonemes = ['', *target_phonemes, '']

    source_phonemes: list[PhonemeInstance] = []
    print(target_phonemes)

    pre = ''
    for i in range(len(target_phonemes)):
        if i < len(target_phonemes) - 1:
            nex = strip_stress(target_phonemes[i+1])
        else:
            nex = ''
        curr = strip_stress(target_phonemes[i])
        source_phonemes.append(
            choose_phoneme(
                voice,
                typemes,
                curr,
                method,
                pre=pre,
                nex=nex,
            )
        )

        pre = curr

    return target_phonemes, source_phonemes

def fade(t: int, length: int) \
        -> tuple[float, float]:
    prev_A, curr_A = ((length-t) / length, t / length)

    return prev_A, curr_A

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='jamcoder')
    parser.add_argument('-nc','--no-crossfade', action='store_true',
        default=config['no-crossfade'])
    parser.add_argument('-w','--crossfade-overlap', type=float,
        default=config['crossfade-overlap'],
        help='Sets the overlap of the crossfade, from 0.0 to 1.0. Higher means more overlap.')
    parser.add_argument('-nd', '--no-dual-similarity', action='store_true',
        default=config['no-dual-similarity'])
    parser.add_argument('-o', '--outfile', type=str,
        default=config['outfile'])
    parser.add_argument('-u', '--voice', type=str, required=True)
    parser.add_argument('-s', '--sentence', nargs='+', required=True)
    parser.add_argument('-d', '--debug', action='store_true',
        default=config['debug'])
    args = parser.parse_args()

    voice = args.voice
    target_text = " ".join(args.sentence)
    crossfade = not args.no_crossfade
    crossfade_overlap = args.crossfade_overlap
    outfile = args.outfile

    try:
        nltk.data.find('averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng')

    if crossfade_overlap < 0.0 or crossfade_overlap > 1.0:
        print(f'ERR: Crossfade overlap out of bounds! Must be between 0.0 and 1.0, Have {crossfade_overlap}')
        exit(-1)

    data_path = Path(__file__).parent.parent.resolve().joinpath(Path('data'))
    voice_path = data_path.joinpath(Path(voice))
    pickle_path = data_path.joinpath(f'{Path(voice)}.pickle')

    try:
        pickle_file = open(pickle_path, 'rb')
        voice = pickle.load(pickle_file)
        pickle_file.close()
    except:
        print(f'no pickle file found at {pickle_path}. creating dataloader...')
        voice = PhonemeMemLoader(voice_path)
        pickle_file = open(pickle_path, 'wb')
        pickle.dump(voice, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    typemes = standard_typeme_tree()

    method = 'dual_equality' if args.no_dual_similarity else 'dual_similarity'
    target_phonemes, source_phonemes = naive_synthesis(voice, typemes, target_text, method)

    synth_wav = np.array([])
    sr =  None
    for i in range(len(source_phonemes)):
        instance = source_phonemes[i]

        if args.debug:
            print(f'[{target_phonemes[i]}]: word {instance.word} interval {instance.interval}')

        (wav, sr), grid = voice.get_word_data(instance.word)

        grid_i = grid['phonetic'][instance.interval]

        # select specific phoneme wav
        wav_start = int(np.floor(sr * grid_i.xmin))
        wav_end = int(np.floor(sr * grid_i.xmax))
        wav_length = wav_end - wav_start

        wav_seg = wav[wav_start:wav_end]

        # Crossfading
        if crossfade and i != 0:
            '''
            An overlap of 1 means that the when the previous track starts
            fading out, the next track immediatley starts to fade in.
            An overlap of 0 means that the next track starts when the previous
            track is done fading out completely.
            '''
            try:
                prev_instance = source_phonemes[i-1]

                prev_grid_i = grid['phonetic'][prev_instance.interval]
            except IndexError:
                synth_wav = np.concatenate((synth_wav, wav_seg))
                continue

            prev_wav_start = int(np.floor(sr * prev_grid_i.xmin))
            prev_wav_end = int(np.floor(sr * prev_grid_i.xmax))
            prev_wav_length = prev_wav_end - prev_wav_start

            if prev_wav_length < wav_length:
                crossfade_length = int(np.floor(0.1 * prev_wav_length))
            else:
                crossfade_length = int(np.floor(0.1 * wav_length))

            wav_overlap = wav_seg[0:crossfade_length]

            # zero-pad to establish overlap buffer
            zerobuf = np.zeros((int(np.floor((1-crossfade_overlap) * crossfade_length))))
            wav_seg = np.concatenate((zerobuf, wav_seg))

            for t in range(crossfade_length):
                prev_A, curr_A = fade(t, crossfade_length)
                synth_wav[len(synth_wav) - crossfade_length + t] *= prev_A
                synth_wav[len(synth_wav) - crossfade_length + t] += \
                    (wav_overlap[t] * curr_A)
            synth_wav = np.concatenate((synth_wav, wav_seg[crossfade_length:]))
        else:
            synth_wav = np.concatenate((synth_wav, wav_seg))

    write(outfile, sr, synth_wav)
