import metrics

class Phoneme:
    """
    Phoneme class stores:
      phoneme {string} (e.g. 'UW1')
      the following lists with corresponding elements:
        words {list[strings]}
        intonations per f0_heuristic {list[float]}
        interval indices within words {list[int]}
        previous phonemes within words {list[Phoneme}]
        next phonemes within words {list[Phoneme]}

    Number of words and intervals must be equal.
    Words can repeat.
    """
    def __init__(
            self,
            phoneme: str,
            
            words: list[str] = [],
            intonations: list[float] = [],
            intervals: list[int] = [],
            prevs: list[Phoneme] = [],
            nexts: list[Phoneme] = []):
        self.num_words = len(words)
        assert self.num_words == len(intonations) \
            and self.num_words == len(intervals) \
            and self.num_words == len(prevs) \
            and self.num_words == len(nexts), \
        'Number of words, intonations, intervals within words, previous phonemes and next phonemes differ.'

        self.phoneme = phoneme
        self.words = words
        self.intonations = intonations
        self.intervals = intervals
        self.prevs = prevs
        self.nexts = nexts

    def __len__(self) -> int:
        return self.num_words
    
    def __getitem__(self, idx: int) -> tuple[str, int, Phoneme, Phoneme]:
        return (
            self.words[idx],
            self.intonations[idx],
            self.intervals[idx],
            self.prevs[idx],
            self.nexts[idx]
        )

    def __setitem__(self, idx: int, x: tuple[str, int]) -> tuple[str, int]:
        assert type(x) == tuple and len(x) == 4 \
        and type(x[0]) == str and type(x[1]) == int \
        and type(x[2]) == Phoneme and type(x[3]) == Phoneme, \
            'element must be of type tuple[str, int, Phoneme, Phoneme]'
        self.words[idx] = x[0]
        self.intervals[idx] = x[1]
        self.prevs[idx] = x[2]
        self.nexts[idx] = x[3]
        return x

    def append(self, word: str, interval: int, pre: Phoneme, nex: Phoneme) \
            -> tuple[str, int, Phoneme, Phoneme]:
        """
        Append word and corresponding interval index, previous phoneme
        and next phoneme within word as an instance of the phoneme.
        Returns the word and interval index.
        """
        assert type(word) == str, \
            '"word" argument must be of type str'
        assert type(interval) == int, \
            '"interval" argument must be of type int'
        assert type(pre) == Phoneme, \
            '"pre" argument must be of type Phoneme'
        assert type(nex) == Phoneme, \
            '"nex" argument must be of type Phoneme'
        
        self.words.append(word)
        self.intervals.append(interval)
        self.prevs.append(pre)
        self.nexts.append(nex)
        self.num_words += 1
        return (word, interval, pre, nex)
