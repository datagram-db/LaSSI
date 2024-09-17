__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024,  Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__status__ = "Production"
from functools import lru_cache


@lru_cache(maxsize=4095)
def ld(s, t):
    if not s: return len(t)
    if not t: return len(s)
    if s[0] == t[0]: return ld(s[1:], t[1:])
    l1 = ld(s, t[1:])
    l2 = ld(s[1:], t)
    l3 = ld(s[1:], t[1:])
    return 1 + min(l1, l2, l3)


def lev(x, y):
    if len(x) == 0 and len(y) == 0:
        return 1.0
    L = max(len(x), len(y))
    return float(L - ld(x, y)) / float(L)


def MultiLevenshtein(x, y):
    L = y.split("\s+")
    overallDistance = 1.0
    for xX in x.split("\s+"):
        overallDistance *= lev(xX, min(L, key=lambda x: lev(x, xX)))
    return overallDistance
