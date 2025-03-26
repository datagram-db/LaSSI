__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2025, Functional Match"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPLv3"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

from typing import get_args

def where(matches, condition):
    from FunctionalMatch.PropositionalLogic import Prop
    assert isinstance(condition, get_args(Prop))
    if len(matches) == 0:
        return True, matches
    result = [idx for idx, x in enumerate(matches) if condition.interpretation(x)]
    return len(result)>0, result