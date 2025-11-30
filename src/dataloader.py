import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import textgrids
import librosa

class PhonemeLoader(ABC):
    """
    PhonemeLoader  
      do not instantiate directly  
      abstrat class for PhonemeDiskLoader, PhonemeMemLoader  
    """
    @abstractmethod
    def __init__(self, phoneme_subpath: str):
        """
        phoneme_subpath: absolute or relative path of data directory
        """
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> tuple[tuple[np.ndarray, int], textgrids.TextGrid]:
        pass

    def __len__(self) -> int:
        return self.num_grids
    
    def get_grid(self, key: str) -> textgrids.TextGrid:
        return self.grid_dict[key][1]

class PhonemeDiskLoader(PhonemeLoader):
    """
    PhonemeDiskLoader  
      on-disk approach where wavfiles not loaded into memory, but grids are  
      all textgrid annotations stored as TextGrid objects  
      i think it all gets cached anyway so what was even the point  
    """
    def __init__(self, phoneme_subpath: str):
        print(f"Loading annotations from {phoneme_subpath}...")

        phoneme_path = Path.cwd().joinpath(Path(phoneme_subpath)).resolve()
        if not phoneme_path.exists():
            print(f"Error: {phoneme_path} does not exist")
            exit(-1)

        self.voice = phoneme_path.stem
        self.grid_dict = {}

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

        self.num_grids = len(self.grid_dict)
        print(f"Loaded {self.num_grids} wavfile and textgrid pairs from {phoneme_subpath}")

    def __getitem__(self, key: str) -> tuple[tuple[np.ndarray, int], textgrids.TextGrid]:
        (wav_file, grid) = self.grid_dict[key]
        wav = librosa.load(wav_file, sr=None, mono=True)

        return (wav, grid)

class PhonemeMemLoader(PhonemeLoader):
    """
    PhonemeMemLoader  
      in-memory approach where wavfiles and grids loaded into memory  
      all textgrid annotations stored as TextGrid objects  
    """
    def __init__(self, phoneme_subpath: str):
        print(f"Loading annotations from {phoneme_subpath}...")

        phoneme_path = Path.cwd().joinpath(Path(phoneme_subpath)).resolve()
        if not phoneme_path.exists():
            print(f"Error: {phoneme_path} does not exist")
            exit(-1)

        self.voice = phoneme_path.stem
        self.grid_dict = {}

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

        self.num_grids = len(self.grid_dict)
        print(f'Loaded {self.num_grids} wavfile and textgrid pairs from {phoneme_subpath}')

    def __getitem__(self, key: str) -> tuple[tuple[np.ndarray, int], textgrids.TextGrid]:
        return self.grid_dict[key]

if __name__ == '__main__':
    print('=== PhonemeLoader demo ==================')
    root_path = Path(__file__).parent.parent.resolve()
    james_path = root_path.joinpath(Path('data/james'))
    # james = PhonemeDiskLoader(james_path)
    james = PhonemeMemLoader(james_path)

    print('\n=== data examples ==================')

    example = james['teacher'] # pick a wav+textgrid   
    print(f'wav: {example[0]}') # wavfile (nd.array, sampling_rate)
    print(f'duration: {example[1].xmin}-{example[1].xmax}')
    print(example[1]['orthographic'][1]) # primary orthographic interval

    # all phonetic intervals
    for i in range(len(example[1]['phonetic'])):
        sclomp = example[1]['phonetic'][i]
        print(f' [{i}] {sclomp.text:<8}\t{sclomp.xmin}-{sclomp.xmax}')
        # print(f' [{i}] {sclomp}}')