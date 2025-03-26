__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2025, Functional Match"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPLv3"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"


from dataclasses import dataclass, is_dataclass, fields
from typing import Union, List, Tuple

from dacite import from_dict

from FunctionalMatch import JSONPath
from FunctionalMatch.ReturningFirstObjects import MatchedObjects
from FunctionalMatch.functions.Reference import Reference
from FunctionalMatch.functions.structural_match import Variable
from FunctionalMatch.utils import FrozenDict

# def replace_with(obj, d: dict):
#     ref = Reference.from_object(obj)
#     if ref in d:
#         obj = d[ref]
#     elif obj in d:
#         obj = d[obj]
#     elif is_dataclass(obj):
#         obj_inst = dict()
#         for field in fields(obj):
#             obj_inst[field.name] = replace_with(getattr(obj, field.name), d)
#         return from_dict(type(obj), obj_inst)
#     if isinstance(obj, Reference):
#         return obj.get()
#     else:
#         return obj

def replace_with_v2(d:Tuple[Tuple], original:dict):
    assert isinstance(original, FrozenDict) or isinstance(original, dict)
    if isinstance(original, dict):
        original = FrozenDict.from_dictionary(original)
    if len(original) == 0:
        return original
    for k, v in d:
        assert isinstance(k, str)
        if isinstance(v, JSONPath):
            from FunctionalMatch.PropositionalLogic import var_interpret
            original = original.update(k, var_interpret(v, original))
        elif isinstance(v, str):
            if v in original:
                original = original.update(k, original.get(v))
        else:
            from FunctionalMatch import instantiate
            from FunctionalMatch.Match import evaluate_structural_function
            original = original.update(k, evaluate_structural_function(instantiate(v, original)))
    return original


@dataclass(frozen=True, order=True, eq=True)
class ReplaceWith:
    replacement: Tuple[Tuple[Union[Variable,JSONPath],
                            Union[Variable,JSONPath,object]]]

    def __call__(self, obj: FrozenDict) -> FrozenDict:
        return replace_with_v2(self.replacement, obj)


@dataclass(frozen=True, order=True, eq=True)
class RewriteAs:
    isShallow: bool
    replacement: Tuple[Tuple[Union[Variable,JSONPath], Union[Variable,JSONPath,object]]]



@dataclass(frozen=True, order=True, eq=True)
class UpdateWith:
    """
    Updates a match (:jsonpath) with either a value or some transformation function (value)
    """
    jsonpath: str
    value: object

    def __call__(self, obj: MatchedObjects) -> MatchedObjects:
        assert obj.variables["^"] == obj
        from FunctionalMatch.PropositionalLogic import var_interpret
        l = var_interpret(JSONPath(self.jsonpath), obj.variables, True)
        if len(l) == 0:
            return obj
        dd = dict()
        for x in l:
            if callable(self.value):
                dd[x] = self.value(x)
            else:
                dd[x] = self.value
        return ReplaceWith(**dd)(obj)