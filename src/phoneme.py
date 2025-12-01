class Phoneme:
    """
    Phoneme class stores:
      phoneme (string) (e.g. 'UW1')
      words list[strings]
      corresponding interval indices within words list[int]
    Number of words and intervals must be equal.
    Words can repeat.
    """
    def __init__(
            self,
            phoneme: str,
            
            words: list[str] = [],
            intervals: list[int] = []):
        self.num_words = len(words)
        assert self.num_words == len(intervals), ...
        'Different number of words and word intervals.'

        self.phoneme = phoneme
        self.words = words
        self.intervals = intervals

    def __len__(self) -> int:
        return self.num_words
    
    def __getitem__(self, idx: int) -> tuple[str, int]:
        return (self.words[idx], self.intervals[idx])

    def __setitem__(self, idx: int, x: tuple[str, int]) -> tuple[str, int]:
        assert type(x) == tuple and len(x) == 2 \
        and type(x[0]) == str and type(x[1] == int ), \
            'element must be of type tuple[str, int]'
        self.words[idx] = x[0]
        self.intervals[idx] = x[1]
        return x

    def append(self, word: str, interval: int) -> tuple[str, int]:
        """
        Append word and corresponding interval index within word
        as an instance of the phoneme.
        Returns the word and interval index.
        """
        assert type(word) == str, \
            '"word" argument must be of type str'
        assert type(interval) == int, \
            '"interval" argument must be of type int'
        
        self.words.append(word)
        self.intervals.append(interval)
        self.num_words += 1
        return (word, interval)
