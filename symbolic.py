from interfaces import Wire, MagField
import numpy as np, matplotlib.pyplot as plt
from numpy import ndarray

class SymWire(Wire):
    def __init__(self, R, L, wp, V, ρ, s):
        super().__init__(R, L, wp, V, ρ, s)
        raise NotImplementedError #TODO: make wire path symbolic before
    
class SymMagField(MagField):
    def __init__(self, wires):
        super().__init__(wires)
        raise NotImplementedError
    
    def calc(self, grid:ndarray):
        raise NotImplementedError