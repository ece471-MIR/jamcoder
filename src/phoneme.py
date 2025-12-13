from __future__ import annotations
import intonation
from typing import Self

class PhonemeInstance:
    """
    PhonemeInstance class stores:
      phoneme {Phoneme}
      word {string}
      interval index within word {int}
      intonation per f0_heuristic {float}
      previous phoneme within word {PhonemeInstance}
      next phoneme within word {PhonemeInstance}
    """
    def __init__(
            self,
            phoneme: Phoneme,

            word: str,
            interval: int,

            intonation: float,
            pre: str | None = None,
            nex: str | None = None):

        self.phoneme = phoneme
        self.word = word
        self.interval = interval
        self.intonation = intonation
        self.pre = pre
        self.nex = nex

class Phoneme:
    """
    Phoneme class stores:
      phoneme name {string} (e.g. 'UW1')
      phoneme instance list {list[PhonemeInstance]},
      where each PhomemeInstance object contains:
      the following lists with corresponding elements:
        parent phoneme {Phoneme}
        word {string}
        interval index within word {int}
        intonation per f0_heuristic {float}
        name of previous phoneme within word {str}
        name of next phoneme within word {str}
    """
    def __init__(
            self,
            name: str,
            instances: list[PhonemeInstance] = []):
        self.name = name
        self.instances = instances
        self.num_instances = len(instances)

    def __len__(self) \
            -> int:
        return self.num_instances

    def __getitem__(self, word_interval: tuple[string, int]) \
            -> PhonemeInstance | None:
        word, interval = word_interval

        for i in range(self.num_instances):
            if self.instances[i].word == word and self.instances[i].interval == interval:
                return self.instances[i]
        return None

    def append(self, instance: PhonemeInstance) \
            -> PhonemeInstance:
        """
        Append PhonemeInstance to Phoneme.
        Returns the supplied PhonemeInstance.
        """
        assert type(instance) == PhonemeInstance, \
            f'"phoneme" argument must be of type Phoneme, received {type(instance)}'

        self.instances.append(instance)
        self.num_instances += 1
        return PhonemeInstance
