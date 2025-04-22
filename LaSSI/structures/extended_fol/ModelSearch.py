# from LaSSI.Parmenides.TBox.ExpandConstituents import CasusHappening, test_pairwise_sentence_similarity, isImplication
# from logical_repr.Sentences import FUnaryPredicate, FBinaryPredicate, FNot
# from logical_repr.rewrite_kernels import make_not
from pydatagramdb import result

from LaSSI.structures.extended_fol.Formulae import *
from LaSSI.Parmenides.Parmenides import CasusHappening


class ModelSearchBasis:
    def __init__(self, original, constituents):
        self.original = original
        self.unary = []
        self.binary = []
        if isinstance(original, FUnaryPredicate):
            self.unary.insert(0, original)
        elif isinstance(original, FBinaryPredicate):
            self.binary.insert(0, original)
        else:
            raise Exception("Unexpected expression: " + str(original))
        from LaSSI.structures.extended_fol.TBoxReasoning import non_redundant_constituents
        for constituent in constituents:
            if isinstance(constituent, FUnaryPredicate) or (isinstance(constituent, FNot) and isinstance(constituent.arg, FUnaryPredicate)):
                if not (constituent == original) and non_redundant_constituents(constituent, False):
                    self.unary.append(constituent)
            elif isinstance(constituent, FBinaryPredicate) or (isinstance(constituent, FNot) and isinstance(constituent.arg, FBinaryPredicate)):
                if not (constituent == original) and non_redundant_constituents(constituent, False):
                    self.binary.append(constituent)
            else:
                raise Exception("Unexpected expression: "+str(constituent))

    def all(self):
        return self.unary + self.binary



class ModelSearch:
    def __init__(self):
        self.pairwise_similarity_cache = dict()
        # self.kb = kb
        self.main_cache = dict()

    def searchInSet(self, lhs, rhsSet):
        foundImplication = False
        foundEquivalence = False
        for rhs in rhsSet:
            from LaSSI.Parmenides.TBox.ExpandConstituents import test_pairwise_sentence_similarity
            val = test_pairwise_sentence_similarity(self.pairwise_similarity_cache, lhs, rhs, shift=False)
            if (val == CasusHappening.EXCLUSIVES):
                # val = test_pairwise_sentence_similarity(dict(), lhs, rhs, kb=self.kb, shift=False)
                return val
            if (val == CasusHappening.EQUIVALENT): ## To check: if I found at least one equivalence after rewriting, then that's it.
                return CasusHappening.EQUIVALENT
            from LaSSI.Parmenides.TBox.ExpandConstituents import isImplication
            if isImplication(val):
                # val = test_pairwise_sentence_similarity(dict(), lhs, rhs, kb=self.kb)
                test_pairwise_sentence_similarity({}, lhs, rhs, shift=False)
                return CasusHappening.GENERAL_IMPLICATION
        return CasusHappening.INDIFFERENT

    def compare(self, objLHS:ModelSearchBasis, objRHS:ModelSearchBasis)->'CasusHappening':
        cp = (objLHS.original, objRHS.original)
        if cp in self.main_cache:
            return self.main_cache[cp]
        if (objLHS.original == objRHS.original):
            self.main_cache[cp] = CasusHappening.EQUIVALENT
            return self.main_cache[cp]
        elif ((objLHS.original == make_not(objRHS.original)) or
              (objLHS.original == make_not(objLHS.original)) or
              (make_not(objLHS.original) in objRHS.unary) or
              (make_not(objLHS.original) in objRHS.binary) or
              (make_not(objRHS.original) in objLHS.unary) or
              (make_not(objRHS.original) in objLHS.binary)):
            self.main_cache[cp] = CasusHappening.EXCLUSIVES
            return self.main_cache[cp]
        elif ((objLHS.original in objRHS.unary) or
              (objLHS.original in objRHS.binary)):
            self.main_cache[cp] = CasusHappening.GENERAL_IMPLICATION
            return self.main_cache[cp]
        else:
            # Performing the constituents search:
            for lhs in objLHS.unary:
                negForm = make_not(lhs) if not isinstance(lhs, FNot) else lhs.arg
                if negForm in objRHS.unary:
                    self.main_cache[cp] = CasusHappening.EXCLUSIVES
                    return self.main_cache[cp]
            for lhs in objLHS.binary:
                negForm = make_not(lhs) if not isinstance(lhs, FNot) else lhs.arg
                if negForm in objRHS.binary:
                    self.main_cache[cp] = CasusHappening.EXCLUSIVES
                    return self.main_cache[cp]
            for lhs in objLHS.unary:
                if lhs in objRHS.unary:
                    self.main_cache[cp] = CasusHappening.GENERAL_IMPLICATION
                    return self.main_cache[cp]
            for lhs in objLHS.binary:
                if lhs in objRHS.binary:
                    self.main_cache[cp] = CasusHappening.GENERAL_IMPLICATION
                    return self.main_cache[cp]
            # Performing the exhaustive search:
            elems = set()
            firstConst = None
            for lhs in objLHS.unary:
                val = self.searchInSet(lhs, objRHS.unary)
                if val == CasusHappening.EXCLUSIVES:
                    self.main_cache[cp] = val
                    return val
                elif val != CasusHappening.INDIFFERENT:
                    elems.add(val)
                    if firstConst is None:
                        firstConst = val
                    # return val
            from LaSSI.Parmenides.TBox.ExpandConstituents import simplifyConstituents
            result = simplifyConstituents(elems)
            # assert (firstConst is None) or (result == firstConst)
            if result != CasusHappening.INDIFFERENT:
                self.main_cache[cp] = result
                return result
            elems = set()
            firstConst = None
            for lhs in objLHS.binary:
                elems = {CasusHappening.INDIFFERENT}
                val = self.searchInSet(lhs, objRHS.binary)
                if val == CasusHappening.EXCLUSIVES:
                    self.main_cache[cp] = val
                    return val
                elif val != CasusHappening.INDIFFERENT:
                    elems.add(val)
                    if firstConst is None:
                        firstConst = val
                    # return val
            # from LaSSI.Parmenides.TBox.ExpandConstituents import simplifyConstituents
            # result = simplifyConstituents(elems)
            # # assert (firstConst is None) or (result == firstConst)
            self.main_cache[cp] = result
            return self.main_cache[cp]
