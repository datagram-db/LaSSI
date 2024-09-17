import copy
import dataclasses
import itertools
import json
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from typing import Tuple

import pandas as pd

from logical_repr.Sentences import formula_from_dict, FNot, FAnd, Formula, FOr, FBinaryPredicate, make_variable


# from logical_repr.Sentences import formula_from_dict, FBinaryPredicate, FUnaryPredicate, FNot
#
@dataclasses.dataclass(order=True, frozen=True, eq=True)
class RWVariable(Formula):
    name: str




def structure_dictionary(dd):
    if dd is None or (len(dd)) == 0:
        return pd.DataFrame()
    L = []
    for x,y in dd.items():
        VALS = []
        OBJS = []
        for val, obj in y:
            VALS.append(val)
            OBJS.append(obj)
        if len(dd) == 1:
            return pd.DataFrame({x: VALS, "obj": OBJS})
        L.append(pd.DataFrame({x: VALS, "obj": OBJS}))
    return reduce(lambda x, y: pd.merge(x,y,how="outer"), L)




if __name__ == "__main__":
    with open("/home/giacomo/Scaricati/newcastle_sentences.txt_logical_rewriting.json", "r") as f:
        list_json = json.load(f)
        list_json = formula_from_dict(list_json)
        qq = FBinaryPredicate("have", make_variable("x"), make_variable("y"), 1.0, frozenset())
        for formula in list_json:
            d = defaultdict(list)
            print(formula.matchWith(qq, d, None))
            dd = structure_dictionary(d)
            print(dd)