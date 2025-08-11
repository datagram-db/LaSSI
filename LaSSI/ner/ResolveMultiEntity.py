# # Prallelized version:
# __author__ = "Oliver R. Fox, Giacomo Bergami"
# __copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
# __credits__ = ["Oliver R. Fox, Giacomo Bergami"]
# __license__ = "GPL"
# __version__ = "2.0"
# __maintainer__ = "Oliver R. Fox, Giacomo Bergami"
# __status__ = "Production"
#
# import itertools
# import multiprocessing
# from concurrent.futures import ThreadPoolExecutor
# from typing import List
#
# from LaSSI.similarities.levenshtein import lev
# from LaSSI.structures.kernels.Sentence import lemmatize_verb
# from LaSSI.structures.meuDB.meuDB import MeuDBEntry
#
#
# # from gsmtosimilarity.TwoGrams import TwoGramSetSimilarity
# # from gsmtosimilarity.levenshtein import lev
#
#
# def build_loc_result(text, type, start_char, end_char, monad, conf, id, src):
#     if isinstance(id, str):
#         yield MeuDBEntry(text, type, start_char, end_char, monad, conf, id, src)
#     from collections.abc import Iterable
#     if isinstance(id, Iterable):
#         for x in id:
#             yield MeuDBEntry(text, type, start_char, end_char, monad, conf, x, src)
#     else:
#         yield MeuDBEntry(text, type, start_char, end_char, monad, conf, str(id), src)
#
# import asyncio
#
# def background(f):
#     def wrapped(*args, **kwargs):
#         return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
#
#     return wrapped
#
# class ResolveMultiNamedEntity:
#
#     def __init__(self, threshold, forinsert, src, parmo=None, trustworthiness_source=1.0):
#         self.trustworthiness_source = trustworthiness_source
#         self.parmo = parmo
#         self.threshold = threshold
#         self.forinsert = forinsert
#         self.result = []
#         self.s = None
#         self.fa = None
#         self.src = src
#
#     def _test(self, current, rest, k, v, start, end, type: str | List[str]):
#         if len(rest) == 0:
#             if k >= self.forinsert:
#                 if isinstance(type, list):
#                     type = self.parmo.most_specific_type(type)
#                 yield from build_loc_result(current, type, start, end, v, k * self.trustworthiness_source, v, self.src)
#         else:
#             next = current + " " + rest[0][0]
#             val = lev(next.lower(), v.lower())
#             if val < k:
#                 if k >= self.forinsert:
#                     if isinstance(type, list):
#                         type = self.parmo.most_specific_type(type)
#                     yield from build_loc_result(current, type, start, end, v, k * self.trustworthiness_source, v,
#                                               self.src)
#             else:
#                 yield from self._test(next, rest[1:], val, v, start, rest[0][2], type)
#
#
#     def start(self, stringa, s, fa, nlp, type):
#         self.s = s
#         self.fa = fa
#         self.result.clear()
#         self.holding_futures = []
#         cpus = int(multiprocessing.cpu_count() / 3 * 2)
#         with ThreadPoolExecutor(max_workers=cpus) as executor:
#             future = executor.submit(pow, 323, 1235)
#             # print(future.result())
#             for sentence in nlp(stringa).sentences:
#                 # List of lemmatized and non-lemmatized words from sentence
#                 ls = [(token.text, token.start_char, token.end_char) for token in sentence.tokens] + [
#                     (lemmatize_verb(token.text), token.start_char, token.end_char) for token in sentence.tokens]
#                 if type is None:
#                     for i in range(len(ls)):
#                         self.holding_futures.append(executor.submit(self.untyped_match, i, ls, s))
#                 else:
#                     for i in range(len(ls)):
#                         self.holding_futures.append(executor.submit(self.typed_match, i, ls, s, type))
#
#         self.result = [x.result() for x in self.holding_futures]
#         return itertools.chain.from_iterable(self.result)
#
#     def typed_match(self, i, ls, s, type):
#         m = s.fuzzyMatch(self.threshold, ls[i][0])
#         for k, v in m.items():
#             for candidate in v:
#                 # cand = s.get(candidate)
#                 newK = lev(ls[i][0].lower(), candidate.lower())
#                 if newK >= self.threshold:
#                     yield from self._test(ls[i][0], ls[i + 1:], newK, candidate, ls[i][1], ls[i][2], type)
#                 else:
#                     yield from []
#         if len(m)==0:
#             yield from []
#
#     def untyped_match(self, i, ls, s):
#         m = s.typedFuzzyMatch(self.threshold, ls[i][0])
#         for k, v in m.items():
#             for candidate, candidate_type in v:
#                 # cand = s.get(candidate)
#                 newK = lev(ls[i][0].lower(), candidate.lower())
#                 if newK >= self.threshold:
#                     yield from self._test(ls[i][0], ls[i + 1:], newK, candidate, ls[i][1], ls[i][2],
#                                         [candidate_type])
#                 else:
#                     yield from []
#         if len(m)==0:
#             yield from []


__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox, Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

from functools import lru_cache
from typing import List

from LaSSI.similarities.levenshtein import lev
from LaSSI.structures.kernels.Sentence import lemmatize_verb
from LaSSI.structures.meuDB.meuDB import MeuDBEntry


# from gsmtosimilarity.TwoGrams import TwoGramSetSimilarity


def build_loc_result(text, type, start_char, end_char, monad, conf, id, src):
    if isinstance(id, str):
        return [MeuDBEntry(text, type, start_char, end_char, monad, conf, id, src)]
    from collections.abc import Iterable
    if isinstance(id, Iterable):
        return map(lambda x: MeuDBEntry(text, type, start_char, end_char, monad, conf, x, src), id)
    else:
        return [MeuDBEntry(text, type, start_char, end_char, monad, conf, str(id), src)]


class ResolveMultiNamedEntity:

    def __init__(self, threshold, forinsert, src, parmo=None, trustworthiness_source=1.0):
        self.trustworthiness_source = trustworthiness_source
        self.parmo = parmo
        self.threshold = threshold
        self.forinsert = forinsert
        self.result = []
        self.s = None
        self.fa = None
        self.src = src

    def test(self, current, rest, k, v, start, end, type: str | List[str]):
        if len(rest) == 0:
            if k >= self.forinsert:
                if isinstance(type, list):
                    type = self.parmo.most_specific_type(type)
                for j in build_loc_result(current, type, start, end, v, k * self.trustworthiness_source, v, self.src):
                    self.result.append(j)
        else:
            next = current + " " + rest[0][0]
            val = lev(next.lower(), v.lower())
            if val < k:
                if k >= self.forinsert:
                    if isinstance(type, list):
                        type = self.parmo.most_specific_type(type)
                    for j in build_loc_result(current, type, start, end, v, k * self.trustworthiness_source, v,
                                              self.src):
                        self.result.append(j)
            else:
                self.test(next, rest[1:], val, v, start, rest[0][2], type)

    def start(self, stringa, s, fa, nlp, type):
        self.s = s
        self.fa = fa
        self.result.clear()
        for sentence in nlp(stringa).sentences:
            tokens = [(token.text, token.start_char, token.end_char) for token in sentence.tokens]

            for i, (text, start_char, end_char) in enumerate(tokens):
                terms_to_check = [text, lemmatize_verb(text)]

                for term in terms_to_check:
                    term = term.lower()

                    if type is None:
                        m = s.typedFuzzyMatch(self.threshold, term)
                        for k, v in m.items():
                            for candidate, candidate_type in v:
                                # cand = s.get(candidate)
                                newK = lev(term, candidate.lower())
                                if newK >= self.threshold:
                                    self.test(term, tokens[i + 1:], newK, candidate, start_char, end_char, [candidate_type])
                    else:
                        m = s.fuzzyMatch(self.threshold, term)
                        for k, v in m.items():
                            for candidate in v:
                                # cand = s.get(candidate)
                                newK = lev(term, candidate.lower())
                                if newK >= self.threshold:
                                    self.test(term, tokens[i + 1:], newK, candidate, start_char, end_char, type)

        return self.result
