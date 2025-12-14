from __future__ import annotations

class Typeme:
    """
    Typeme class stores:
      typeme name {str}
      parent typeme {Typeme}
      child typemes {list[Typeme]}
    During initialisation, accepts a list of
    child typeme names for which to create
    Typeme objects
    """
    def __init__(
            self,
            name: str,

            depth: int = 0,
            parent: Typeme = None,
            spawn: tuple[str] = ()):
        self.name = name
        self.depth = depth
        self.parent = parent
        self.children = []
        for kid in spawn:
            self.children.append(Typeme(kid, depth+1, parent=self))
    
    def __getitem__(self, name: str) \
            -> Typeme:
        """
        Returns Typeme for queried typeme name if it corresponds to
        the queried typeme or any of its child typemes.
        Returns None otherwise.
        """
        if self.name == name:
            return self

        for child in self.children:
            if child[name] != None:
                return child[name]
        return None
    
    def child_names(self) \
            -> list[str]:
        return [child.name for child in self.children]
    
    def set_depth(self, depth):
        if self.depth == depth:
            return
        self.depth = depth
        for child in self.children:
            child.set_depth(depth+1)

    def adopt(self, children: list[Typeme] = []):
        """
        Babyes
        """
        for child in children:
            child.parent = self
            child.set_depth(self.depth + 1)
            self.children.append(child)
    
    def similarity(self, other: Typeme) \
            -> int:
        """
        Returns the depth of the deepest shared Typeme in the tree if
        it exists, where depth is relative to the top-level Typeme.
        Requires both Typemes to belong to the same Typeme tree, with
        the same Typeme root node.
        Returns 0 if the top level Typeme is the deepest shared typeme.
        Returns -1 if there is no shared Typeme.
        """
        if other == None:
            return -1
        
        self_side, self_depth = self, self.depth
        other_side, other_depth = other, other.depth

        while self_depth > other_depth:
            self_side = self_side.parent
            self_depth -= 1
        while other_depth > self_depth:
            other_side = other_side.parent
            other_depth -= 1
        
        while self_depth >= 0:
            # print(f'{self_side.name} and {other_side.name}')
            if self_side.name == other_side.name:
                return self_depth
            self_side = self_side.parent
            other_side = other_side.parent
            self_depth -= 1 

        return -1

def standard_typeme_tree() \
        -> Typeme:
    """
    Returns a hierarchical tree of phoneme types.
    """

    front = Typeme(name='front', depth=2,
        spawn=('IY', 'IH', 'EH', 'AE'))
    mid = Typeme(name='mid', depth=2,
        spawn=('AA', 'ER', 'AH', 'AO'))
    back = Typeme(name='back', depth=2,
        spawn=('UW', 'UH', 'OW'))
    diphthongs = Typeme(name='diphthongs', depth=1,
        spawn=('AY', 'OY', 'AW', 'EY'))
    liquids = Typeme(name='liquids', depth=2,
        spawn=('W', 'L'))
    glides = Typeme(name='glides', depth=2,
        spawn=('R', 'Y'))
    nasals = Typeme(name='nasals', depth=2,
        spawn=('M', 'N', 'NG'))
    voiced_stops = Typeme(name='voiced_stops', depth=3,
        spawn=('B', 'D', 'G'))
    unvoiced_stops = Typeme(name='unvoiced_stops', depth=3,
        spawn=('P', 'T', 'K'))
    voiced_fricatives = Typeme(name='voiced_fricatives', depth=3,
        spawn=('V', 'TH', 'Z', 'ZH'))
    unvoiced_fricatives = Typeme(name='unvoiced_fricatives', depth=3,
        spawn=('F', 'S', 'SH'))
    whisper = Typeme(name='whisper', depth=2,
        spawn=('H', 'HH'))
    affricates = Typeme(name='affricates', depth=2,
        spawn=('JH', 'CH'))
    silence = Typeme(name='silence', depth=1,
        spawn=('', ' ', '.', '!', ',', '?', '...', '-'))
    
    vowels = Typeme(name='vowels', depth=1)
    vowels.adopt([front, mid, back])

    semivowels = Typeme(name='semivowels', depth=1)
    semivowels.adopt([liquids, glides])

    stops = Typeme(name='stops', depth=2)
    stops.adopt([voiced_stops, unvoiced_stops])

    fricatives = Typeme(name='fricatives', depth=2)
    fricatives.adopt([voiced_fricatives, unvoiced_fricatives])

    consonants = Typeme(name='consonants', depth=1)
    consonants.adopt([nasals, stops, fricatives, whisper, affricates])
    
    typemes = Typeme(name='phonemes', depth=0)
    typemes.adopt([vowels, diphthongs, semivowels, consonants, silence])
    
    return typemes
