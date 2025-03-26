__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2025, Functional Match"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPLv3"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

from dataclasses import dataclass, is_dataclass, fields

@dataclass(frozen=True, eq=True, order=True)
class Variable:
    name: str

@dataclass(frozen=True, eq=True, order=True)
class Ignore:
    pass

ignore_instance = Ignore()

def var(x):
    if x == "$":
        raise TypeError("$ is not a valid variable name")
    return Variable(x)


@dataclass(frozen=True, eq=True, order=True)
class JSONPath:
    expression: str


def _structural_match(query:object, target:object, match_result):
    if (query is None) or (type(query).__name__ == "Ignore"):
        return match_result
    if type(query).__name__ == "Variable":
        match_result[query.name] = target
        return match_result
    elif type(query).__name__ == "tuple":
        if not query[0] == type(target).__name__:
            return None
        from FunctionalMatch.utils import FrozenDict
        assert isinstance(query[1], FrozenDict)
        for k, v in query[1].items():
            if not hasattr(target, k):
                return None
            match_result = _structural_match(v, getattr(target, k), match_result)
        return match_result
    elif is_dataclass(query):
        if not type(target).__name__ == type(query).__name__:
            return None
        for field in fields(target):
            match_result = _structural_match(getattr(query, field.name), getattr(target, field.name), match_result)
            if match_result is None:
                return match_result
        return match_result
    else:
        if query == target:
            return match_result
        else:
            return None


def equi_join_dictionaries(ls):
    if len(ls) <= 1:
        return ls[0]
    else:
        left = ls[0]
        right = ls[1]
        tmp = []
        for left_dict in left:
            for right_dict in right:
                if all(map(lambda y: left_dict[y] == right_dict[y], set(left_dict.keys()).intersection(right_dict.keys()))):
                    tmp.append(left_dict.update(right_dict))
        if len(tmp) == 0:
            return tmp
        ls.insert(0, tmp)
        return equi_join_dictionaries(ls)




