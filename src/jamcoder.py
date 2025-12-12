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
from typemes import Typeme
from config import synth_config as config

def choose_phoneme(voice: PhonemeLoader, typemes: Typeme, target: str, pre: str = None, nex: str = None) \
        -> PhonemeInstance:
    """
    Chooses the most optimal source phoneme from the supplied voice
    to use as the basis for the synthesised target phoneme.
    """
    phoneme = voice.get_phoneme(target)

    both_instance, pre_instance, nex_instance, none_instance = None, None, None, None
    for instance in phoneme.instances:
        if pre == instance.pre and nex == instance.nex:
            if both_instance == None \
                    or both_instance.intonation > instance.intonation:
                both_instance = instance
        
        elif pre == instance.pre:
            if pre_instance == None \
                    or pre_instance.intonation > instance.intonation:
                pre_instance = instance
        
        elif nex == instance.nex:
            if nex_instance == None \
                    or nex_instance.intonation > instance.intonation:
                nex_instance = instance
        else:
            if none_instance == None \
                    or none_instance.intonation > instance.intonation:
                none_instance = instance
    
    # VERY LAZY
    if both_instance != None:
        return both_instance
    if pre_instance != None:
        return pre_instance
    if nex_instance != None:
        return nex_instance
    if none_instance != None:
        return none_instance

def naive_synthesis(voice: PhonemeLoader, typemes: Typeme, target_text: str) \
        -> list[PhonemeInstance]:
    """
    """
    g2p = G2p()
    target_phonemes = g2p(target_text)
    target_phonemes = ['' if p == ' ' else p for p in target_phonemes]
    
    source_phonemes = []
    print(target_phonemes)
    
    pre = None
    for i in range(len(target_phonemes)):
        if i < len(target_phonemes) - 1:
            nex = strip_stress(target_phonemes[i+1])
        else:
            nex = None
        curr = strip_stress(target_phonemes[i])
        source_phonemes.append(
            choose_phoneme(
                voice,
                typemes,
                curr,
                pre=pre,
                nex=nex
            )
        )

        pre = curr
    
    return target_phonemes, source_phonemes

def print_usage():
    print('Usage: python jamcoder.py [VOICE WEIGHT] [VOICE WEIGHT]...[VOICE WEIGHT] TARGET_TEXT')
    print('  VOICE: voice with WAV, TextGrid data in data/')
    print('  WEIGHT: weight to apply to voice, 0.0-1.0.\n    For now, this must be 1 for one voice ONLY')
    print('  TARGET_TEXT: target text string to synthesise')
    print('If no voice and weight are supplied, synthesis defers to src/config.py::synth_config.')

def init_typemes() \
        -> Typeme:
    typemes = Typeme('phoneme', 0)
    vowels, diphthongs, semivowels, consonants = \
        typemes.declare_children(
            ('vowels', 'diphthongs', 'semivowels', 'consonants'))

    front, mid, back = \
        vowels.declare_children(
            ('front', 'mid', 'back'))
    liquids, glides = \
        semivowels.declare_children(
            ('liquids', 'glides'))
    nasals, stops, fricatives, whisper, affricates = \
        typemes.declare_children(
            ('nasals', 'stops', 'fricatives', 'whisper', 'affricates'))
    voiced_stops, unvoiced_stops = \
        stops.declare_children(
            ('voiced stops', 'unvoiced stops'))
    voiced_fricatives, unvoiced_fricatives = \
        fricatives.declare_children(
            ('voiced_fricatives', 'unvoiced_fricatives'))

    front.declare_children(('IY', 'IH', 'EH', 'AE'))
    mid.declare_children(('AA', 'ER', 'AH', 'AO'))
    back.declare_children(('UW', 'UH', 'OW'))
    diphthongs.declare_children(('AY', 'OY', 'AW', 'EY'))
    liquids.declare_children(('W', 'L'))
    glides.declare_children(('R', 'Y'))
    nasals.declare_children(('M', 'N', 'NG'))
    voiced_stops.declare_children(('B', 'D', 'G'))
    unvoiced_stops.declare_children(('P', 'T', 'K'))
    voiced_fricatives.declare_children(('V', 'TH', 'Z', 'ZH'))
    unvoiced_fricatives.declare_children(('F', 'S', 'SH'))
    whisper.declare_children(('H', 'HH'))
    affricates.declare_children(('JH', 'CH'))

    return typemes

if __name__ == '__main__':
    try:
        nltk.data.find('averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng')

    if len(sys.argv) < 2 or len(sys.argv) % 2 != 0:
        print_usage()
        exit(-1)

    voices, weights = [], []
    for i in range(1, len(sys.argv)-1, 2):
        voice, weight = sys.argv[i], sys.argv[i+1]
        try: 
            weight = float(weight)
            assert type(voice) == str
        except:
            print_usage()
            exit(-1)
        voices.append(voice)
        weights.append(weight)

    target_text = sys.argv[-1]

    """
    All code from this point on is temporary and assumes only one voice
    is being synthesised with weight 1.
    """
    if len(voices) > 1 or type(weights[0]) != float or \
            weights[0] < 0.0 or weights[0] > 1.0:
        print_usage()
        exit(-1)
    
    data_path = Path(__file__).parent.parent.resolve().joinpath(Path('data'))
    voice_path = data_path.joinpath(Path(voices[0]))
    pickle_path = data_path.joinpath(f'{Path(voices[0])}.pickle')

    try:
        pickle_file = open(pickle_path, 'rb')
        voice = pickle.load(pickle_file)
        pickle_file.close()
    except:
        print(f'no pickle file found at {pickle_path}. creating dataloader...')
        voice = PhonemeMemLoader(voice_path)

    typemes = init_typemes()
    target_phonemes, source_phonemes = naive_synthesis(voice, typemes, target_text)

    synth_wav = np.array([])
    sr = None
    for i in range(len(source_phonemes)):
        instance = source_phonemes[i]

        if config['debug']:
            print(f'[{target_phonemes[i]}]: word {instance.word} interval {instance.interval}')

        (wav, sr), grid = voice.get_word_data(instance.word)

        grid_i = grid['phonetic'][instance.interval]
        
        wav_start = int(np.floor(sr * grid_i.xmin))
        wav_end = int(np.floor(sr * grid_i.xmax))
        wav_seg = wav[wav_start:wav_end]

        synth_wav = np.concatenate((synth_wav, wav_seg))

    write('synth.wav', sr, synth_wav)