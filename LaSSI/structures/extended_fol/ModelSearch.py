from Parmenides.TBox.ExpandConstituents import CasusHappening, test_pairwise_sentence_similarity, isImplication
from logical_repr.Sentences import FUnaryPredicate, FBinaryPredicate, FNot
from logical_repr.rewrite_kernels import make_not


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
        for constituent in constituents:
            if isinstance(constituent, FUnaryPredicate) or (isinstance(constituent, FNot) and isinstance(constituent.arg, FUnaryPredicate)):
                if not (constituent == original):
                    self.unary.append(constituent)
            elif isinstance(constituent, FBinaryPredicate) or (isinstance(constituent, FNot) and isinstance(constituent.arg, FBinaryPredicate)):
                if not (constituent == original):
                    self.binary.append(constituent)
            else:
                raise Exception("Unexpected expression: "+str(constituent))



class ModelSearch:
    def __init__(self, kb):
        self.pairwise_similarity_cache = dict()
        self.kb = kb
        self.main_cache = dict()

    def searchInSet(self, lhs, rhsSet):
        foundImplication = False
        foundEquivalence = False
        for rhs in rhsSet:
            val = test_pairwise_sentence_similarity(self.pairwise_similarity_cache, lhs, rhs, kb=self.kb, shift=False)
            if (val == CasusHappening.EXCLUSIVES):
                # val = test_pairwise_sentence_similarity(dict(), lhs, rhs, kb=self.kb, shift=False)
                return val
            if (val == CasusHappening.EQUIVALENT): ## To check: if I found at least one equivalence after rewriting, then that's it.
                return CasusHappening.EQUIVALENT
            if isImplication(val):
                # val = test_pairwise_sentence_similarity(dict(), lhs, rhs, kb=self.kb)
                foundImplication = True
        return CasusHappening.GENERAL_IMPLICATION if foundImplication else CasusHappening.INDIFFERENT

    def compare(self, objLHS:ModelSearchBasis, objRHS:ModelSearchBasis)->CasusHappening:
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
                negForm = make_not(lhs)
                if negForm in objRHS.unary:
                    self.main_cache[cp] = CasusHappening.EXCLUSIVES
                    return self.main_cache[cp]
            for lhs in objLHS.binary:
                negForm = make_not(lhs)
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
            for lhs in objLHS.unary:
                val = self.searchInSet(lhs, objRHS.unary)
                if val != CasusHappening.INDIFFERENT:
                    self.main_cache[cp] = val
                    return val
            for lhs in objLHS.binary:
                val = self.searchInSet(lhs, objRHS.binary)
                if val != CasusHappening.INDIFFERENT:
                    self.main_cache[cp] = val
                    return val
            self.main_cache[cp] = CasusHappening.INDIFFERENT
            return self.main_cache[cp]
