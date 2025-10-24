class bcolors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    PURPLE = '\033[35m'
    MAGENTA = '\033[35m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'


class INDEX_TYPE:
    NONE = 0,
    RDF_GRAPH = 1,
    FAISS = 2,


class NER_TYPE:
    NONE = 0,
    SPACY = 1,

class CG_TYPE:
    '''
    Context generator
    '''
    NONE=0,
    RDF_COMPOSITION=1,
    RDF_LET_IT_REASON=2,


class CM_TYPE:
    '''
    Relation Composition matrix
    '''
    NONE = 0,
    APPROXIMATE = 1,

class FEW_SHOT_TYPE:
    NONE = 0,
    RANDOM = 1,
    STATIC = 2,
    ANN_SIMILAR = 3,