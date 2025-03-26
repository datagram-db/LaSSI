from enum import Enum


class PairwiseCases(Enum):
    NonImplying = 0
    Implying = 1
    MutuallyExclusive = 2
    Equivalent = 3
