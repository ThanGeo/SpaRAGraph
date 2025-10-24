from utils import INDEX_TYPE

from abc import ABC, abstractmethod

class BaseIndex(ABC):
    def __init__(self, data_input, index_type: INDEX_TYPE):
        self.data_input = data_input
        self.type = index_type

    def getType(self):
        return self.type

    def getDatasetPath(self):
        return self.data_input

    @abstractmethod
    def generateContext(self, referenced_entities: str):
        pass