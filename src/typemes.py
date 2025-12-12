from __future__ import annotations

class Typeme:
    """
    Typeme class stores:
      typeme name {str}
      parent typeme {Typeme}
      child typemes {list[Typeme]}
    """
    def __init__(
            self,
            name: str,

            depth: int = 0,
            parent: Typeme = None,
            children: list[Typeme] = []):
        self.name = name
        self.depth = depth
        self.parent = parent
        self.children = children
    
    def __getitem__(self, name: str) \
            -> Typeme:
        """
        Returns Typeme for queried typeme name if it corresponds to
        the queried typeme or any of its child typemes.
        Returns None otherwise.
        """
        if self.name == name:
            return self

        for child in children:
            if child[name] != None:
                return child
        return None
    
    def declare_children(self, names: tuple[str] = ()) \
            -> tuple[Typeme]:
        """
        Creates child typemes for all supplied names.
        Returns a list of their Typemes.
        """
        children = ()
        for name in names:
            children = children + (Typeme(name, self.depth + 1, self),)
        return children
    
    def similarity(self, other: Typeme) \
            -> int:
        """
        Returns the depth of the deepest shared Typeme in the tree if
        it exists, where depth is relative to the top-level Typeme.
        Returns 0 if the top level Typeme is the deepest shared typeme.
        Returns -1 if there is no shared Typeme.
        """
        
        depth = self.depth if self.depth < other.depth else other.depth
        self_side, self_depth = self, self.depth
        other_side, other_depth = other, other.depth

        while self_depth > depth:
            self_side = self_side.parent
            self_depth -= 1
        while other_depth > depth:
            other_side = other_side.parent
            other_depth -= 1
        
        while depth >= 0:
            if self_side.name == other_side.name:
                return depth
            self_side = self_side.parent
            other_side = other_side.parent
            depth -= 1 

        return -1
    


