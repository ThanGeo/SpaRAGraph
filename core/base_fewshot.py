
from abc import ABC, abstractmethod

from utils import FEW_SHOT_TYPE

class BaseFewShot(ABC):
    def __init__(self, type: FEW_SHOT_TYPE):
        self.type = type

    def getType(self):
        return self.type
    
    @abstractmethod
    def getKshot(self, text: str, k: int):
        pass
