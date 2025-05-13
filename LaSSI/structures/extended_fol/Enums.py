from enum import Enum


class PairwiseCases(Enum):
    Indifferent = 0
    Implying = 1
    ConflictingImplication = 2
    Equivalent = 3

def print_case(premise, case: PairwiseCases, consequence):
    if case == PairwiseCases.Indifferent:
        return f"{premise} ? {consequence}"
    elif case == PairwiseCases.Implying:
        return f"{premise} ⇒ {consequence}"
    elif case == PairwiseCases.Equivalent:
        return f"{premise} ≡ {consequence}"
    elif case == PairwiseCases.ConflictingImplication:
        return f"{premise} ≢ {consequence}"
