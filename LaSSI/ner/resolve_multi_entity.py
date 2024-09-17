__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox, Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

from LaSSI.similarities.levenshtein import lev
from LaSSI.structures.meuDB.meuDB import MeuDBEntry


# from gsmtosimilarity.TwoGrams import TwoGramSetSimilarity
# from gsmtosimilarity.levenshtein import lev


def build_loc_result(text, type, start_char, end_char, monad, conf, id):
    if isinstance(id, str):
        return [MeuDBEntry(text,type,start_char,end_char,monad,conf, id)]
    from collections.abc import Iterable
    if isinstance(id, Iterable):
        return map(lambda x: MeuDBEntry(text,type,start_char,end_char,monad,conf, x), id)
    else:
        return [MeuDBEntry(text,type,start_char,end_char,monad,conf, str(id))]


class ResolveMultiNamedEntity:

    def __init__(self, threshold, forinsert):
        self.threshold = threshold
        self.forinsert = forinsert
        self.result = []
        self.s = None
        self.fa = None

    def test(self, current, rest, k, v, start, end, type):
        if len(rest) == 0:
            if k >= self.forinsert:
                for j in build_loc_result(current, type, start, end, v, k, v):
                    self.result.append(j)
        else:
            next = current + " " + rest[0][0]
            val = lev(next.lower(), v.lower())
            if val < k:
                if k >= self.forinsert:
                    for j in build_loc_result(current, type, start, end, v, k, v):
                        self.result.append(j)
            else:
                self.test(next, rest[1:], val, v, start, rest[0][2], type)

    def start(self, stringa, s, fa, nlp, type):
        self.s = s
        self.fa = fa
        self.result.clear()
        for sentence in nlp(stringa).sentences:
            ls = [(token.text, token.start_char, token.end_char) for token in sentence.tokens]
            for i in range(len(ls)):
                m = s.fuzzyMatch(self.threshold, ls[i][0])
                for k, v in m.items():
                    for candidate in v:
                        # cand = s.get(candidate)
                        newK = lev(ls[i][0].lower(), candidate.lower())
                        if newK >= self.threshold:
                            self.test(ls[i][0], ls[i + 1:], newK, candidate, ls[i][1], ls[i][2], type)
        return self.result