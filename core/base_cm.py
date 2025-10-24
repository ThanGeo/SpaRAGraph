
from abc import ABC, abstractmethod

from utils import CM_TYPE

class BaseCM(ABC):
    def __init__(self, type: CM_TYPE):
        self.type = type

    def getType(self):
        return self.type
    
    @abstractmethod
    def getCombinedRelation(self, path):
        pass
