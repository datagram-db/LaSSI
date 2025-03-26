__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2025, Functional Match"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPLv3"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"


from dataclasses import is_dataclass, fields, dataclass
from typing import Union

from dacite import from_dict, Config

from FunctionalMatch.utils import FrozenDict


# # from FunctionalMatch.utils import FrozenDict
#
# def replaceWith(query:object, replacement_map, value_map):
#     raise RuntimeError("Not implemented")
#     # for key, value in replacement_map.items():
#     #     from FunctionalMatch.PropositionalLogic import var_interpret
#     #     result = var_interpret(key, value_map)
#     #     if result is not None:
#     #         if isinstance(result, list):
#     #             for x in result:
#

def instantiate(query:object, kwargs:Union[dict,FrozenDict]):
    if type(query).__name__ == "Variable":
        return kwargs.get(query.name, None)
    if type(query).__name__ == "Ignore":
        return None
    elif is_dataclass(query):
        from FunctionalMatch.Match import ExternalMatchByExtesion, evaluate_structural_function
        if isinstance(query, ExternalMatchByExtesion) or type(query).__name__ == "ExternalMatchByExtesion":
            return instantiate(query.structural_map(lambda x: instantiate(x, kwargs)).callMe(), kwargs)
        else:
            obj_inst = dict()
            for field in fields(query):
                from FunctionalMatch.Match import evaluate_structural_function
                obj_inst[field.name] = evaluate_structural_function(instantiate(getattr(query, field.name), kwargs))
            return from_dict(type(query), obj_inst, Config(check_types=False))
    else:
        return query

