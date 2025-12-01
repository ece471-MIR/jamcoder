import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import textgrids
import librosa

from phoneme import Phoneme

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

    def __len__(self) -> int:
        return self.num_phonemes
    
    def list_phonemes(self) -> list[str]:
        """
        Returns a list of all phonemes present across all TextGrid annotations.
        Includes ''.
        """
        return list(self.phoneme_dict.keys())

    @abstractmethod
    def get_word_data(self, word: str) -> tuple[tuple[np.ndarray, int], textgrids.TextGrid]:
        """
        Returns the following data tuple for the queried word:
        (
          (waveform vector, sampling rate),
          TextGrid
        )
        """
        pass
    
    def get_phoneme(self, phoneme: str) -> Phoneme:
        """
        Returns the Phoneme object corresponding to the queried phoneme.
        """
        return self.phoneme_dict[phoneme]
    
    def get_phoneme_data(self, phoneme: str) -> list[tuple[tuple[str, int], tuple[np.ndarray, int], textgrids.Interval]]:
        """
        Returns a list containing all instances of the queried phoneme:
        list[
          (
            (word, interval index),
            (waveform vector, sampling rate),
            TextGrid Interval
          )
        ]
        """
        phoneme_obj = self.phoneme_dict[phoneme]
        data = []

        for instance in range(len(phoneme_obj)):
            word, i = phoneme_obj[instance]
            wav, grid = self.get_word_data(word)

            grid_i = grid['phonetic'][i]
            t_start = int(np.floor(wav[1] * grid_i.xmin))
            t_end = int(np.ceil(wav[1] * grid_i.xmax))

            data.append((
                (word, i),
                (wav[0][t_start:t_end], wav[1]),
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
                self.grid_dict[stem] = (str(wav_file), grid)
            except:
                print(f'Warning: Could not open {grid_file.name} as TextGrid')
                continue

            for i in range(len(grid['phonetic'])):
                phoneme = grid['phonetic'][i].text
                
                if (phoneme in self.phoneme_dict):
                    self.phoneme_dict[phoneme].append(stem, i)
                else:
                    self.phoneme_dict[phoneme] = Phoneme(phoneme, [stem], [i])

        self.num_phonemes = len(self.phoneme_dict)
        print(f'Loaded {len(self.grid_dict)} wavfile and textgrid pairs from {phoneme_subpath}')

    def get_word_data(self, word: str) -> tuple[tuple[np.ndarray, int], textgrids.TextGrid]:
        (wav_file, grid) = self.grid_dict[word]
        wav = librosa.load(wav_file, sr=None, mono=True)

        return (wav, grid)

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
                wav = librosa.load(wav_file, sr=None, mono=True)
            except:
                print(f'Warning: Could not load {grid_file.name} as WAV')
                continue

            self.grid_dict[stem] = (wav, grid)
            for i in range(len(grid['phonetic'])):
                phoneme = grid['phonetic'][i].text
                
                if (phoneme in self.phoneme_dict):
                    self.phoneme_dict[phoneme].append(stem, i)
                else:
                    self.phoneme_dict[phoneme] = Phoneme(phoneme, [stem], [i])

        self.num_phonemes = len(self.phoneme_dict)
        print(f'Loaded {len(self.grid_dict)} wavfile and textgrid pairs from {phoneme_subpath}')
    
    def get_word_data(self, word: str) -> tuple[tuple[np.ndarray, int], textgrids.TextGrid]:
        return self.grid_dict[word]

if __name__ == '__main__':
    print('=== PhonemeLoader demo ==================')
    root_path = Path(__file__).parent.parent.resolve()
    james_path = root_path.joinpath(Path('data/james'))
    # james = PhonemeDiskLoader(james_path)
    james = PhonemeMemLoader(james_path)

    print('\n=== data examples ==================')

    # list all phonemes represented in dataset
    all_phonemes = james.list_phonemes()
    print(f'{len(all_phonemes)} phonemes found: {all_phonemes}')

    # pick a phoneme any phoneme
    phoneme = 'UH1'

    # get the word, wav section and textgrid interval of every instance
    example = james.get_phoneme_data(phoneme)

    # print data for every instance of phoneme
    print(f'{len(example)} instances of phoneme "{phoneme}":')
    for ((word, i), (wav_seg, sr), grid_i) in example:
        print(f'  interval {i} of word {word}:')
        print(f'    wav_seg duration: {len(wav_seg) / sr}s')
        print(f'    grid_i: {grid_i}')
