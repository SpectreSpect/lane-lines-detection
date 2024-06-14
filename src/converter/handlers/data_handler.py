from abc import ABC, abstractmethod

class DataHandler(ABC):
    @abstractmethod
    def load(self, path: str) -> list:
        pass
    
    @abstractmethod
    def save(self, path: str, validation_split: int):
        pass