from abc import ABC, abstractmethod
from utils import CG_TYPE

class BaseCG(ABC):
    '''
    After the graph traversal, some paths have been generated.
    The context generator is responsible for formatting/composing the path into comprehensive context for the LLM
    '''
    def __init__(self, cg_type: CG_TYPE):
        self.type = cg_type

    def getType(self):
        return self.type
    
    @abstractmethod 
    def generateContext(self, paths: "list[list[(str,str,str)]]") -> str:
        '''
        Generate context from a path list, where each path is a list of RDF triplet tuples (<subject>,<relation>,<object>).
        '''
        pass
