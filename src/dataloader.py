import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import textgrids
import librosa

from phoneme import Phoneme, PhonemeInstance
from intonation import f0_heuristic

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
        phoneme = self.phoneme_dict[name]
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
                name = grid_i.text
                
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
                name = grid_i.text
                
                wav_start = int(np.floor(sr * grid_i.xmin))
                wav_end = int(np.floor(sr * grid_i.xmax))
                
                intonation = f0_heuristic(wav[wav_start:wav_end], sr, None)
                print(intonation)
                
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
    name = 'UH1'

    # get the word, wav section and textgrid interval of every instance
    example = james.get_phoneme_data(name)

    # print data for every instance of phoneme
    print(f'{len(example)} instances of phoneme "{name}":')
    for (instance, (wav_seg, sr), grid_i) in example:
        print(f'  interval {instance.interval} of word {instance.word}:')
        print(f'    intonation: {instance.intonation}')
        print(f'    wav_seg duration: {len(wav_seg) / sr}s')
        print(f'    grid_i: {grid_i}')
