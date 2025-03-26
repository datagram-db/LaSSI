import os.path
import pickle
from collections import defaultdict

from FunctionalMatch.example.LaSSI.eFOLsemantics.Enums import PairwiseCases
from FunctionalMatch.example.LaSSI.eFOLsemantics.ModelSearch import ModelSearch, ModelSearchBasis
from FunctionalMatch.example.LaSSI.eFOLsemantics.TBoxReasoning import TBoxReasoningSingleton
from FunctionalMatch.example.parmenides.Formulae import FVariable, FNot, FBinaryPredicate, FUnaryPredicate
from FunctionalMatch.example.parmenides.Parmenides import CasusHappening, Parmenides, ParmenidesSingleton

def isImplication(x):
    return x == CasusHappening.GENERAL_IMPLICATION or x == CasusHappening.LOSE_SPEC_IMPLICATION or x == CasusHappening.INSTANTIATION_IMPLICATION or x == CasusHappening.MISSING_1ST_IMPLICATION


d_transformCaseWhenOneArgIsNegated = None


def transformCaseWhenOneArgIsNegated(orig: CasusHappening):
    """Negation of a more-valued logic (where we have more than just satisfiability or not)"""
    global d_transformCaseWhenOneArgIsNegated
    if d_transformCaseWhenOneArgIsNegated is None:
        d_transformCaseWhenOneArgIsNegated = {CasusHappening.NONE: CasusHappening.NONE,
                                              CasusHappening.INDIFFERENT: CasusHappening.INDIFFERENT,
                                              CasusHappening.EQUIVALENT: CasusHappening.EXCLUSIVES,
                                              CasusHappening.EXCLUSIVES: CasusHappening.EQUIVALENT,
                                              CasusHappening.GENERAL_IMPLICATION: CasusHappening.INDIFFERENT,
                                              CasusHappening.LOSE_SPEC_IMPLICATION: CasusHappening.INDIFFERENT,
                                              CasusHappening.INSTANTIATION_IMPLICATION: CasusHappening.INDIFFERENT,
                                              CasusHappening.MISSING_1ST_IMPLICATION: CasusHappening.INDIFFERENT}
    return d_transformCaseWhenOneArgIsNegated[orig]


def compare_variable(d, lhs, rhs):
    cp = (lhs, rhs)
    if (cp not in d) and (lhs == rhs):
        d[cp] = CasusHappening.EQUIVALENT
    if cp in d:
        return d[cp]
    if (lhs == rhs):
        val = CasusHappening.EQUIVALENT
    elif lhs is None:
        val = CasusHappening.MISSING_1ST_IMPLICATION
    elif rhs is None:
        val = CasusHappening.INDIFFERENT
    elif (lhs == FNot(rhs)) or (rhs == FNot(lhs)):
        val = CasusHappening.EXCLUSIVES
    elif isinstance(lhs, FNot):
        val = transformCaseWhenOneArgIsNegated(compare_variable(d, lhs.arg, rhs))
    elif isinstance(rhs, FNot):
        val = transformCaseWhenOneArgIsNegated(compare_variable(d, lhs, rhs.arg))
    else:
        assert isinstance(lhs, FVariable)
        assert isinstance(rhs, FVariable)
        kb = ParmenidesSingleton.get()
        nameEQ = kb.name_eq(lhs.name, rhs.name)
        specEQ = kb.name_eq(lhs.specification, rhs.specification)
        if lhs.spec_negation != rhs.spec_negation:
            specEQ = transformCaseWhenOneArgIsNegated(specEQ)
        copCompareInv = compare_variable(d, rhs.cop, lhs.cop)
        val = CasusHappening.INDIFFERENT
        if (nameEQ == specEQ) and (specEQ == copCompareInv):
            d[cp] = specEQ
            return d[cp]
        if nameEQ == CasusHappening.INDIFFERENT:
            nameAgainstSpec = kb.name_eq(lhs.name, rhs.specification)
            if nameAgainstSpec == CasusHappening.EQUIVALENT and lhs.name is not None and rhs.specification is not None:
                val = CasusHappening.INSTANTIATION_IMPLICATION
        elif nameEQ == CasusHappening.EQUIVALENT:
            if (specEQ == copCompareInv):
                val = specEQ
            elif (specEQ == CasusHappening.EQUIVALENT):
                if copCompareInv == CasusHappening.MISSING_1ST_IMPLICATION:
                    val = CasusHappening.LOSE_SPEC_IMPLICATION
                else:
                    val = copCompareInv
            else:
                if specEQ == CasusHappening.MISSING_1ST_IMPLICATION:
                    val = CasusHappening.INSTANTIATION_IMPLICATION
                else:
                    val = specEQ
        elif isImplication(nameEQ):
            nameAgainstSpec = kb.name_eq(lhs.name, rhs.specification)
            if (specEQ == copCompareInv) and (specEQ == CasusHappening.EQUIVALENT):
                val = nameEQ  ## If everything is equivalent, then it is implying as the arguments are
            elif (specEQ == CasusHappening.EQUIVALENT) and (copCompareInv == CasusHappening.EXCLUSIVES):
                val = CasusHappening.EXCLUSIVES
            elif lhs.specification is None and nameAgainstSpec == CasusHappening.EQUIVALENT:
                val = CasusHappening.INSTANTIATION_IMPLICATION
        elif nameEQ == CasusHappening.EXCLUSIVES:
            if (specEQ == copCompareInv) and (specEQ == CasusHappening.EQUIVALENT):
                val = CasusHappening.EXCLUSIVES
    d[cp] = val
    return d[cp]


def simplifyConstituentsAcross(constituentCollection):

    if isinstance(constituentCollection, CasusHappening):
        return constituentCollection
    if CasusHappening.INDIFFERENT in constituentCollection:
        return CasusHappening.INDIFFERENT
    elif CasusHappening.EXCLUSIVES in constituentCollection:
        return CasusHappening.EXCLUSIVES
    elif CasusHappening.MISSING_1ST_IMPLICATION in constituentCollection:
        if not CasusHappening.GENERAL_IMPLICATION in constituentCollection and \
                not CasusHappening.INSTANTIATION_IMPLICATION in constituentCollection and \
                not CasusHappening.LOSE_SPEC_IMPLICATION in constituentCollection:
            return CasusHappening.MISSING_1ST_IMPLICATION
        else:
            return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.INSTANTIATION_IMPLICATION in constituentCollection:
        if not CasusHappening.GENERAL_IMPLICATION in constituentCollection and \
                not CasusHappening.MISSING_1ST_IMPLICATION in constituentCollection and \
                not CasusHappening.LOSE_SPEC_IMPLICATION in constituentCollection:
            return CasusHappening.INSTANTIATION_IMPLICATION
        else:
            return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.LOSE_SPEC_IMPLICATION in constituentCollection:
        if not CasusHappening.GENERAL_IMPLICATION in constituentCollection and \
                not CasusHappening.MISSING_1ST_IMPLICATION in constituentCollection and \
                not CasusHappening.INSTANTIATION_IMPLICATION in constituentCollection:
            return CasusHappening.LOSE_SPEC_IMPLICATION
        else:
            return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.GENERAL_IMPLICATION in constituentCollection:
        return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.INDIFFERENT in constituentCollection:
        return CasusHappening.INDIFFERENT
    elif CasusHappening.EQUIVALENT in constituentCollection:
        return CasusHappening.EQUIVALENT
    else:
        return CasusHappening.INDIFFERENT


def simplifyConstituents(constituentCollection):
    """
    This function calculates the simplication of the constituent collection to the set of elements required to
    boil down a set of elements to one single constituent
    """
    if isinstance(constituentCollection, CasusHappening):
        return constituentCollection
    elif CasusHappening.EXCLUSIVES in constituentCollection:
        return CasusHappening.EXCLUSIVES
    elif CasusHappening.EQUIVALENT in constituentCollection:
        return CasusHappening.EQUIVALENT
    elif CasusHappening.MISSING_1ST_IMPLICATION in constituentCollection:
        if not CasusHappening.GENERAL_IMPLICATION in constituentCollection and \
                not CasusHappening.INSTANTIATION_IMPLICATION in constituentCollection and \
                not CasusHappening.LOSE_SPEC_IMPLICATION in constituentCollection:
            return CasusHappening.MISSING_1ST_IMPLICATION
        else:
            return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.INSTANTIATION_IMPLICATION in constituentCollection:
        if not CasusHappening.GENERAL_IMPLICATION in constituentCollection and \
                not CasusHappening.MISSING_1ST_IMPLICATION in constituentCollection and \
                not CasusHappening.LOSE_SPEC_IMPLICATION in constituentCollection:
            return CasusHappening.INSTANTIATION_IMPLICATION
        else:
            return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.LOSE_SPEC_IMPLICATION in constituentCollection:
        if not CasusHappening.GENERAL_IMPLICATION in constituentCollection and \
                not CasusHappening.MISSING_1ST_IMPLICATION in constituentCollection and \
                not CasusHappening.INSTANTIATION_IMPLICATION in constituentCollection:
            return CasusHappening.LOSE_SPEC_IMPLICATION
        else:
            return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.GENERAL_IMPLICATION in constituentCollection:
        return CasusHappening.GENERAL_IMPLICATION
    else:
        return CasusHappening.INDIFFERENT


def test_pairwise_sentence_similarity(d, x, y, store=True, shift=True):
    if shift:
        if (y, x) in d:
            test_shift = d[(y, x)]
        else:
            test_shift = test_pairwise_sentence_similarity(d, y, x, store,  False)
        if test_shift == CasusHappening.EQUIVALENT or test_shift == CasusHappening.EXCLUSIVES:
            d[(x, y)] = test_shift
            return test_shift
    val = CasusHappening.NONE
    if y is None and x is None:
        val = CasusHappening.EQUIVALENT
    elif y is None:
        val = CasusHappening.GENERAL_IMPLICATION
    elif x is None:
        val = CasusHappening.INDIFFERENT
    elif (x == y):
        val = CasusHappening.EQUIVALENT
    elif (isinstance(x, FNot) and isinstance(y, FNot)):
        val = test_pairwise_sentence_similarity(d, x.arg, y.arg, False)
        if isImplication(val):
            val = CasusHappening.INDIFFERENT
    elif (x == FNot(y)) or (y == FNot(x)):
        val = CasusHappening.EXCLUSIVES
    elif isinstance(x, FNot):
        val = transformCaseWhenOneArgIsNegated(test_pairwise_sentence_similarity(d, x.arg, y, False))
    elif isinstance(y, FNot):
        val = transformCaseWhenOneArgIsNegated(test_pairwise_sentence_similarity(d, x, y.arg, False))
    else:
        if (x.meta != y.meta):
            val = CasusHappening.INDIFFERENT
        else:
            assert isinstance(x, FBinaryPredicate) or isinstance(x, FUnaryPredicate)
            assert isinstance(y, FBinaryPredicate) or isinstance(y, FUnaryPredicate)
            xprop = set() if x.properties is None else x.properties
            yprop = set() if y.properties is None else y.properties
            keyCmp, keyCmpInv = defaultdict(set), defaultdict(set)
            keys = set(map(lambda z: z[0], xprop)).union(map(lambda z: z[0], yprop))
            dLHS = dict(xprop)
            dRHS = dict(yprop)
            keyComparison = (None, None)
            for key in keys:
                if key in dLHS and key in dRHS:
                    for xx in dLHS[key]:
                        for yy in dRHS[key]:
                            keyCmp[key].add(compare_variable(d, xx, yy))
                            keyCmpInv[key].add(compare_variable(d, yy, xx))
                elif key in dLHS:
                    keyCmp[key].add(CasusHappening.INDIFFERENT)
                    keyCmpInv[key].add(CasusHappening.GENERAL_IMPLICATION)
                else:
                    keyCmp[key].add(CasusHappening.GENERAL_IMPLICATION)
                    keyCmpInv[key].add(CasusHappening.INDIFFERENT)
            keyCmp = {key: simplifyConstituents(val) for key, val in keyCmp.items()}
            keyCmpInv = {key: simplifyConstituents(val) for key, val in keyCmpInv.items()}
            keyCmpElements, keyCmpElementsInv = CasusHappening.EQUIVALENT, CasusHappening.EQUIVALENT
            keyComparisonOutcome = copKeyComparisonOutcome = CasusHappening.INDIFFERENT
            if len(keyCmp) > 0:
                keyCmpElements = simplifyConstituentsAcross({keyCmp[key] for key in keyCmp})
                keyCmpElementsInv = simplifyConstituentsAcross({keyCmpInv[key] for key in keyCmpInv})
            if isinstance(x, FBinaryPredicate) and isinstance(y, FBinaryPredicate):
                if (x.rel != y.rel):
                    val = CasusHappening.INDIFFERENT
                else:
                    keyComparison = (x.src, y.src)
                    srcCmp = compare_variable(d, x.src, y.src)
                    if (srcCmp == CasusHappening.INDIFFERENT):
                        val = CasusHappening.INDIFFERENT
                    else:
                        dstCmp = compare_variable(d, x.dst, y.dst)
                        if (dstCmp == CasusHappening.INDIFFERENT):
                            val = CasusHappening.INDIFFERENT
                        else:
                            keyComparisonOutcome = compare_variable(d, x.src, y.src)
                            copKeyComparisonOutcome = compare_variable(d, x.src.cop, y.src.cop)
                            if (srcCmp == CasusHappening.EXCLUSIVES) and (dstCmp == CasusHappening.EXCLUSIVES):
                                val = CasusHappening.INDIFFERENT
                            elif (srcCmp == CasusHappening.EXCLUSIVES) and (dstCmp != CasusHappening.INDIFFERENT):
                                val = CasusHappening.EXCLUSIVES
                            elif (dstCmp == CasusHappening.EXCLUSIVES) and (srcCmp != CasusHappening.INDIFFERENT):
                                val = CasusHappening.EXCLUSIVES
                            elif srcCmp == CasusHappening.EQUIVALENT:
                                val = dstCmp
                            elif dstCmp == CasusHappening.EQUIVALENT:
                                val = srcCmp
                            else:
                                val = simplifyConstituents({srcCmp, dstCmp})
            elif isinstance(y, FUnaryPredicate) and isinstance(x, FUnaryPredicate):
                if (x.rel != y.rel):
                    val = CasusHappening.INDIFFERENT
                else:
                    keyComparison = (x.arg, y.arg)
                    val = compare_variable(d, x.arg, y.arg)
                keyComparisonOutcome = compare_variable(d, x.arg, y.arg)
                copKeyComparisonOutcome = compare_variable(d, x.arg.cop, y.arg.cop)
            else:
                raise ValueError("Unexpected comparison between " + str(x) + " and" + str(y))
            if val != CasusHappening.INDIFFERENT:
                if val == CasusHappening.EQUIVALENT:
                    if keyComparisonOutcome == CasusHappening.EQUIVALENT:
                        if isImplication(keyCmpElements):
                            if copKeyComparisonOutcome == CasusHappening.EQUIVALENT:  # and (
                                # (y.arg is not None) and (y.arg.cop is not None)):
                                if CasusHappening.INDIFFERENT in set(keyCmp.values()):
                                    val = CasusHappening.INDIFFERENT
                                elif CasusHappening.LOSE_SPEC_IMPLICATION in set(keyCmp.values()):
                                    val = CasusHappening.INDIFFERENT
                                elif keyCmpElements == CasusHappening.INSTANTIATION_IMPLICATION or CasusHappening.INSTANTIATION_IMPLICATION in set(
                                        keyCmp.values()):  # or keyCmpElements == CasusHappening.GENERAL_IMPLICATION:
                                    val = keyCmpElements
                                else:
                                    val = CasusHappening.INDIFFERENT
                            else:
                                val = keyCmpElements
                        elif keyCmpElementsInv == CasusHappening.LOSE_SPEC_IMPLICATION:
                            val = CasusHappening.INSTANTIATION_IMPLICATION
                        else:
                            val = keyCmpElements
                    else:
                        val = keyCmpElements
                elif isImplication(val):
                    if keyCmpElements != CasusHappening.EQUIVALENT:
                        if CasusHappening.INDIFFERENT in keyCmp.values():
                            val = CasusHappening.INDIFFERENT
                        elif keyCmpElementsInv == CasusHappening.LOSE_SPEC_IMPLICATION:
                            val = CasusHappening.INSTANTIATION_IMPLICATION
                        else:
                            val = keyCmpElements
                elif val == CasusHappening.EXCLUSIVES:
                    if keyCmpElements == CasusHappening.INDIFFERENT:
                        val = CasusHappening.INDIFFERENT
                    elif keyCmpElements == CasusHappening.EXCLUSIVES:
                        val = CasusHappening.INDIFFERENT
    if store:
        d[(x, y)] = val
    return val


def instantiate_rules(constituents, expansion_dictionary, final_constituents, isImpl):
    for constituent in constituents:
        from FunctionalMatch.example.LaSSI.eFOLsemantics.TBoxReasoning import TBoxReasoningSingleton
        s = TBoxReasoningSingleton.knowledge_expand(constituent, isImpl)
        # s.add(constituent)
        expansion_dictionary[constituent] = s
    for y in expansion_dictionary.values():
        final_constituents = final_constituents.union(set(y.keys()))
    # return {(x, y): CasusHappening.NONE for x in final_constituents for y in
    #         final_constituents}


class ExpandConstituents:
    def __init__(self, cache_folder, constituents):
        """
        This class provides the expansion for each of the sentences, as well as caching the direction of the implication for each of the formulae
        """
        print("Setting up the rule expander...")
        # self.kb = kb

        self.constituents = list(constituents)
        _ied = os.path.join(cache_folder, "_ied.pickle")
        _ic = os.path.join(cache_folder, "_ic.pickle")
        _eed = os.path.join(cache_folder, "_eed.pickle")
        _ec = os.path.join(cache_folder, "_ec.pickle")
        _exp = TBoxReasoningSingleton.get_ke_file_name()

        if (os.path.exists(_ied) and os.path.exists(_ic) and os.path.exists(_eed) and os.path.exists(_ec) and os.path.exists(_exp)):
            with open(_ied, "rb") as f:
                self.impl_expansion_dictionary = pickle.load(f)
            with open(_ic, "rb") as f:
                self.impl_constituents = pickle.load(f)
            with open(_eed, "rb") as f:
                self.eq_expansion_dictionary = pickle.load(f)
            with open(_ec, "rb") as f:
                self.eq_constituents = pickle.load(f)
        else:
            self.impl_expansion_dictionary = dict()
            self.impl_constituents = set()
            self.eq_expansion_dictionary = dict()
            self.eq_constituents = set()

            if not all(map(lambda x: isinstance(x, FBinaryPredicate) or isinstance(x, FUnaryPredicate),
                           self.constituents)):
                raise ValueError(
                    "Error: all the rules within the set of rules must represent Predicates to be assessed, be them unary or binary")

            # Expanding the constituents

            print("Expanding the constituents...")
            # self.outcome_implication_dictionary =
            instantiate_rules(self.constituents, self.eq_expansion_dictionary, self.eq_constituents,
                              False)
            instantiate_rules(self.constituents, self.impl_expansion_dictionary, self.impl_constituents,
                              True)
            # self.outcome_eq_dictionary =

            with open(_ied, "wb") as f:
                pickle.dump(self.impl_expansion_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(_ic, "wb") as f:
                pickle.dump(self.impl_constituents, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(_eed, "wb") as f:
                pickle.dump(self.eq_expansion_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(_ec, "wb") as f:
                pickle.dump(self.eq_constituents, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.result_cache = dict()
        self.ms = ModelSearch()
        self.lhsOrigDict = dict()
        self.rhsOrigDict = dict()
        self.inv_idx = dict()
        print("Splitting across unary and binary constituents for each sentence...")
        for i, sentence in enumerate(self.constituents):
            self.inv_idx[sentence] = i
            self.lhsOrigDict[i] = ModelSearchBasis(sentence, self.impl_expansion_dictionary[sentence].keys())
            self.rhsOrigDict[i] = ModelSearchBasis(sentence, self.eq_expansion_dictionary[sentence].keys())

    def getImplExpansions(self, idx):
        return self.lhsOrigDict[idx].all() if idx in self.lhsOrigDict else []

    def getEqExpansions(self, idx):
        return self.rhsOrigDict[idx].all() if idx in self.lhsOrigDict else []

    def getConstituentIDX(self, ith):
        from FunctionalMatch.example.LaSSI.eFOLsemantics.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.getConstituentIdx(ith)

    def getIthExpandedConstituent(self, ith):
        from FunctionalMatch.example.LaSSI.eFOLsemantics.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.getConstituentFromIdx(ith)

    def getImplExpansionExplanation(self, idx):
        constituent = self.constituents[idx]
        assert idx == self.inv_idx[constituent]
        from FunctionalMatch.example.LaSSI.eFOLsemantics.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.subGraphImpl(constituent)

    def getEqExpansionExplanation(self, idx):
        constituent = self.constituents[idx]
        assert idx == self.inv_idx[constituent]
        from FunctionalMatch.example.LaSSI.eFOLsemantics.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.subGraphEq(constituent)

    def determine(self, i: int, j: int):
        if (i == j):
            self.result_cache[(i, j)] = PairwiseCases.Equivalent
        assert i < len(self.constituents)
        assert j < len(self.constituents)
        if (i, j) in self.result_cache:
            return self.result_cache[(i, j)]
        val = PairwiseCases.NonImplying

        lhsOrig = self.lhsOrigDict[i]
        rhsOrig = self.rhsOrigDict[j]
        tmp = self.ms.compare(lhsOrig, rhsOrig)
        if tmp == CasusHappening.EXCLUSIVES:
            val = PairwiseCases.MutuallyExclusive
        elif tmp == CasusHappening.EQUIVALENT:
            val = PairwiseCases.Equivalent
        elif isImplication(tmp):
            val = PairwiseCases.Implying
        else:
            val = PairwiseCases.NonImplying

        self.result_cache[(i, j)] = val
        return val

    def getIDXGraph(self):
        from FunctionalMatch.example.LaSSI.eFOLsemantics.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.getIDXGraph()

    def getConstituentFromIDX(self, idx):
        from FunctionalMatch.example.LaSSI.eFOLsemantics.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.getConstituentFromIdx(idx)
