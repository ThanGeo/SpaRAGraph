from abc import ABC, abstractmethod
from utils import NER_TYPE

class BaseNER(ABC):
    def __init__(self, ner_type: NER_TYPE):
        self.type = ner_type

    def getType(self):
        return self.type

    @abstractmethod
    def extract_entities(self, text: str) -> "list[str]":
        pass

    