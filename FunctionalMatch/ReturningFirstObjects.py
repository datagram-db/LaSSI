__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2025, Functional Match"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPLv3"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

from dataclasses import dataclass
from typing import Union

from FunctionalMatch import JSONPath
from FunctionalMatch.utils import FrozenDict

@dataclass(frozen=True, eq=True, order=True)
class MatchedObjects:
    obj: object
    variables: FrozenDict

@dataclass(frozen=True, eq=True, order=True)
class Invent:
    """
    This class invents a new object by using the dictionary passed at the argument
    """
    pattern: object

    def __call__(self, kwargs: FrozenDict):
        from FunctionalMatch import instantiate
        return MatchedObjects(instantiate(self.pattern, kwargs), kwargs)


@dataclass(frozen=True, eq=True, order=True)
class FromVariable:
    """
    Returns one of the objects defined within the variable
    """
    variable: str

    def __call__(self, kwargs: FrozenDict):
        from FunctionalMatch.functions.structural_match import Variable
        assert self.variable in kwargs
        from FunctionalMatch.PropositionalLogic import var_interpret
        result = var_interpret(Variable(self.variable), kwargs)
        return MatchedObjects(result, kwargs.update("^", result))


@dataclass(frozen=True, eq=True, order=True)
class FromJSONPath:
    """
    Returns one of the objects defined within the variable
    """
    path: Union[str,JSONPath]

    def __call__(self, kwargs: FrozenDict):
        from FunctionalMatch import JSONPath
        assert self.path in kwargs
        from FunctionalMatch.PropositionalLogic import var_interpret
        result = var_interpret(JSONPath(self.path) if isinstance(self.path, str) else self.path, kwargs)
        return MatchedObjects(result, kwargs.update("^", result))

