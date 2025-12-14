from dataloader import PhonemeLoader, PhonemeMemLoader, strip_stress
from phoneme import Phoneme, PhonemeInstance
from typemes import Typeme

def choose_phoneme(voice: PhonemeLoader, typemes: Typeme, target: str, method: str | None, pre: str = None, nex: str = None) \
        -> PhonemeInstance:
    """
    Chooses the most optimal source phoneme from the supplied voice
    to use as the basis for the synthesised target phoneme.
    """

    if method == 'dual_similarity':
        return choose_dual_similarity(voice, typemes, target, pre, nex)
    elif method == 'dual_equality':
        return choose_dual_equality(voice, typemes, target, pre, nex)
    else:
        print(f"Error: invalid method supplied")
        exit(-1)

def choose_dual_similarity(voice: PhonemeLoader, typemes: Typeme, target: str, pre: str = None, nex: str = None) \
        -> PhonemeInstance:
    """
    Chooses the source phoneme whose previous and next phonemes
    have the most similar typemes to those of the target phoneme,
    falling back on intonation as a tiebreaker.
    """
    phoneme = voice.get_phoneme(target)
    targ_pre = typemes[pre] if pre else None
    targ_nex = typemes[nex] if nex else None

    best_dual_similarity = float('-inf')
    best_instance = None
    # both_instance, pre_instance, nex_instance, none_instance = None, None, None, None
    for instance in phoneme.instances:
        dual_similarity = 0
        
        src_pre = typemes[instance.pre] if instance.pre else None
        src_nex = typemes[instance.nex] if instance.nex else None

        if targ_pre:
            dual_similarity += targ_pre.similarity(src_pre)
        if targ_nex:
            dual_similarity += targ_nex.similarity(src_nex)
        
        if not best_instance or dual_similarity > best_dual_similarity or (
                dual_similarity == best_dual_similarity and best_instance and \
                best_instance and instance.intonation < best_instance.intonation):
            best_dual_similarity = dual_similarity
            best_instance = instance

    targ_pre_name = '' if targ_pre == None else targ_pre.name
    targ_nex_name = '' if targ_nex == None else targ_nex.name
    src_pre_name = '' if instance.pre == None else instance.pre
    src_nex_name = '' if instance.nex == None else instance.nex

    # print(f'sim=\t{best_dual_similarity}\t {src_pre_name}->{targ_pre_name} && {src_nex_name}->{targ_nex_name}')
    return best_instance

def choose_dual_equality(voice: PhonemeLoader, typemes: Typeme, target: str, pre: str = None, nex: str = None) \
        -> PhonemeInstance:
    """
    Chooses the source phoneme which shares the most previous and
    next phonemes with the target phoneme, falling back on
    intonation as a tiebreaker.
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

    if both_instance != None:
        return both_instance
    if pre_instance:
        if nex_instance:
            if nex_instance.intonation < pre_instance.intonation:
                return nex_instance
            else:
                return pre_instance
        else:
            return pre_instance
    if nex_instance != None:
        return nex_instance
    if none_instance != None:
        return none_instance