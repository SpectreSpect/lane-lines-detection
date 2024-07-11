from abc import ABC, abstractmethod

class AbstractPalette():
    @abstractmethod
    def get_color(self, value):
        pass