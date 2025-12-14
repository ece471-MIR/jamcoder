import sys
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import pickle
import textgrids
import librosa

from phoneme import Phoneme, PhonemeInstance
from intonation import f0_heuristic

def strip_stress(name: str) \
        -> str:
    if name == '' or name[-1] not in ['0', '1', '2', '3']:
        return name
    return name[:-1]

class PhonemeLoader(ABC):
    """
    PhonemeLoader class:
      Accepts phoneme_subpath (absolute or relative)
      Stores wavs and phoneme TextGrid annotations,
      which can be queried by word or phoneme.
    Abstract class for PhonemeDiskLoader, PhonemeMemLoader.
    Do not instantiate directly.
    """
    @abstractmethod
    def __init__(self, phoneme_subpath: str):
        pass

    def __len__(self) \
            -> int:
        return self.num_phonemes

    def list_phonemes(self) \
            -> list[str]:
        """
        Returns a list of the names of all phoneme present
        across all TextGrid annotations including '' and 'XX'.
        """
        return list(self.phoneme_dict.keys())

    @abstractmethod
    def get_word_data(self, word: str) \
            -> tuple[tuple[np.ndarray, int], textgrids.TextGrid]:
        """
        Returns the following data tuple for the queried word:
        (
          (waveform vector, sampling rate),
          TextGrid
        )
        """
        pass

    def get_phoneme(self, name: str) \
            -> Phoneme:
        """
        Returns the Phoneme object corresponding to the queried phoneme.
        """
        return self.phoneme_dict[name]

    def get_phoneme_data(self, name: str) \
            -> list[tuple[PhonemeInstance, tuple[np.ndarray, int], textgrids.Interval]]:
        """
        Returns a list containing informaiton about all instances of the
        queried phoneme:
        list[
          (
            PhonemeInstance,
            (waveform vector, sampling rate),
            TextGrid Interval
          )
        ]
        """
        try:
            phoneme = self.phoneme_dict[name]
        except KeyError:
            return []
        data = []

        for instance in phoneme.instances:
            (wav, sr), grid = self.get_word_data(instance.word)

            grid_i = grid['phonetic'][instance.interval]
            wav_start = int(np.floor(sr * grid_i.xmin))
            wav_end = int(np.floor(sr * grid_i.xmax))

            data.append((
                instance,
                (wav[wav_start:wav_end], sr),
                grid_i
            ))

        return data

class PhonemeDiskLoader(PhonemeLoader):
    """
    PhonemeDiskLoader class (PhonemeLoader with wavfiles stored on-disk)
      Accepts phoneme_subpath (absolute or relative)
      Stores wavs and phoneme TextGrid annotations,
      which can be queried by word or phoneme.
    """
    def __init__(self, phoneme_subpath: str):
        print(f"Loading annotations from {phoneme_subpath}...")

        phoneme_path = Path.cwd().joinpath(Path(phoneme_subpath)).resolve()
        if not phoneme_path.exists():
            print(f"Error: {phoneme_path} does not exist")
            exit(-1)

        self.voice = phoneme_path.stem
        self.grid_dict = {}
        self.phoneme_dict = {}

        grid_files = sorted(phoneme_path.glob("*.TextGrid"))
        for grid_file in grid_files:
            stem = grid_file.stem

            wav_file = phoneme_path.joinpath(Path(f'{stem}.wav'))
            if not wav_file.exists():
                print(f'Warning: {wav_file.name} does not exist')
                continue

            try:
                grid = textgrids.TextGrid(grid_file)
            except:
                print(f'Warning: Could not open {grid_file.name} as TextGrid')
                continue

            try:
                wav, sr = librosa.load(wav_file, sr=None, mono=True)
            except:
                print(f'Warning: Could not load {grid_file.name} as WAV')
                continue

            self.grid_dict[stem] = (str(wav_file), grid)

            pre = None
            for interval in range(len(grid['phonetic'])):
                grid_i = grid['phonetic'][interval]
                name = strip_stress(grid_i.text)

                wav_start = int(np.floor(sr * grid_i.xmin))
                wav_end = int(np.floor(sr * grid_i.xmax))

                intonation = f0_heuristic(wav[wav_start:wav_end], sr, None)

                if (name not in self.phoneme_dict):
                    self.phoneme_dict[name] = Phoneme(name, [])

                instance = PhonemeInstance(
                    self.phoneme_dict[name],
                    stem,
                    interval,
                    intonation,
                    pre=pre,
                    nex=None
                )

                self.phoneme_dict[name].append(instance)

                if pre != None:
                    (self.phoneme_dict[pre][stem, interval-1]).nex = self.phoneme_dict[name][stem, interval]
                pre = name


        self.num_phonemes = len(self.phoneme_dict)
        print(f'Loaded {len(self.grid_dict)} wavfile and textgrid pairs from {phoneme_subpath}')

    def get_word_data(self, word: str) -> tuple[tuple[np.ndarray, int], textgrids.TextGrid]:
        (wav_file, grid) = self.grid_dict[word]
        wav, sr = librosa.load(wav_file, sr=None, mono=True)

        return ((wav, sr), grid)

class PhonemeMemLoader(PhonemeLoader):
    """
    PhonemeMemLoader class (PhonemeLoader with wavfiles stored in-memory)
      Accepts phoneme_subpath (absolute or relative)
      Stores wavs and phoneme TextGrid annotations,
      which can be queried by word or phoneme.
    """
    def __init__(self, phoneme_subpath: str):
        print(f"Loading annotations from {phoneme_subpath}...")

        phoneme_path = Path.cwd().joinpath(Path(phoneme_subpath)).resolve()
        if not phoneme_path.exists():
            print(f"Error: {phoneme_path} does not exist")
            exit(-1)

        self.voice = phoneme_path.stem
        self.grid_dict = {}
        self.phoneme_dict = {}

        grid_files = sorted(phoneme_path.glob("*.TextGrid"))
        for grid_file in grid_files:
            stem = grid_file.stem

            wav_file = phoneme_path.joinpath(Path(f'{stem}.wav'))
            if not wav_file.exists():
                print(f'Warning: {wav_file.name} does not exist')
                continue

            try:
                grid = textgrids.TextGrid(grid_file)
            except:
                print(f'Warning: Could not open {grid_file.name} as TextGrid')
                continue

            try:
                wav, sr = librosa.load(wav_file, sr=None, mono=True)
            except:
                print(f'Warning: Could not load {grid_file.name} as WAV')
                continue

            pre = None
            self.grid_dict[stem] = ((wav, sr), grid)
            for interval in range(len(grid['phonetic'])):
                grid_i = grid['phonetic'][interval]
                name = str(strip_stress(grid_i.text))

                wav_start = int(np.floor(sr * grid_i.xmin))
                wav_end = int(np.floor(sr * grid_i.xmax))

                intonation = f0_heuristic(wav[wav_start:wav_end], sr, None)

                if (name not in self.phoneme_dict):
                    self.phoneme_dict[name] = Phoneme(name, [])

                instance = PhonemeInstance(
                    self.phoneme_dict[name],
                    stem,
                    interval,
                    intonation,
                    pre='' if not pre else pre,
                    nex=''
                )

                self.phoneme_dict[name].append(instance)

                if pre != None:
                    (self.phoneme_dict[pre][stem, interval-1]).nex = name
                pre = name

        self.num_phonemes = len(self.phoneme_dict)
        print(f'Loaded {len(self.grid_dict)} wavfile and textgrid pairs from {phoneme_subpath}')

    def get_word_data(self, word: str) -> tuple[tuple[np.ndarray, int], textgrids.TextGrid]:
        return self.grid_dict[word]

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3 :
        print('Usage: python dataloader.py VOICE [PHONEME]')
        print('  VOICE: voice with WAV, TextGrid data in data/')
        print('  PHONEME: phoneme to observe data for')
        print('Pass a PHONEME to query the cached data.')
        print('Pass no PHONEME to reload and repickle the data.')
        exit(-1)

    voice_name = sys.argv[1]
    root_path = Path(__file__).parent.parent.resolve()
    voice_path = root_path.joinpath(Path(f'data/{voice_name}'))
    pickle_path = root_path.joinpath(Path(f'data/{voice_name}.pickle'))

    if len(sys.argv) == 2:
        print('=== PhonemeLoader repickle ==============')
        voice = PhonemeMemLoader(voice_path)
        all_phonemes = voice.list_phonemes()
        print(f'{len(all_phonemes)} phonemes found: {all_phonemes}')
        try:
            pickle_file = open(pickle_path, 'wb')
            pickle.dump(voice, pickle_file)
            pickle_file.close()
            print(f'wrote data to {pickle_path}')
        except:
            print(f'pickling to {pickle_path} failed')
            exit(-1)
        exit(0)


    print('=== PhonemeLoader query =================')
    try:
        pickle_file = open(pickle_path, 'rb')
        voice = pickle.load(pickle_file)
        pickle_file.close()
    except:
        print('Depickling failed. Recreating dataset and pickling...')
        voice = PhonemeMemLoader(voice_path)

        try:
            pickle_file = open(pickle_path, 'wb')
            pickle.dump(voice, pickle_file)
            pickle_file.close()
            print(f'wrote data to {pickle_path}')
        except:
            print(f'pickling to {pickle_path} failed')
            exit(-1)

    # list all phonemes represented in dataset
    all_phonemes = voice.list_phonemes()
    print(f'{len(all_phonemes)} phonemes found: {all_phonemes}')


    # get the word, wav section and textgrid interval of every instance
    name = sys.argv[2]
    example = voice.get_phoneme_data(name)

    # print data for every instance of phoneme
    print(f'{len(example)} instances of phoneme "{name}" in {voice_name}:')
    for (instance, (wav_seg, sr), grid_i) in example:
        print(f'  interval {instance.interval} of word {instance.word}:')
        print(f'    intonation: {instance.intonation}')
        print(f'    wav_seg duration: {len(wav_seg) / sr}s')
        print(f'    grid_i: {grid_i}')
