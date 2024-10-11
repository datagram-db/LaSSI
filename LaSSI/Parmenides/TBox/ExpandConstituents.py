import os.path
import pickle
from collections import defaultdict
from enum import Enum

from LaSSI.structures.extended_fol.ModelSearch import ModelSearchBasis, ModelSearch
from LaSSI.structures.extended_fol.sentence_expansion import PairwiseCases
from LaSSI.structures.extended_fol.Sentences import FVariable, FNot, FBinaryPredicate, FUnaryPredicate


class ExpandConstituents:
    def __init__(self, expander):
        self.expander = expander
        self.constituent_expansion_map = defaultdict(set)

    def expand_formula(self, formula):
        ### Defining the expansion of one single rule
        if formula is None or formula in self.constituent_expansion_map:
            return
        for expansion in self.expander(formula):
            self.constituent_expansion_map[formula].add(expansion)
            self.expand_formula(formula)
        if formula not in self.constituent_expansion_map:
            self.constituent_expansion_map[formula] = set()

    def getExpansionLeaves(self):
        return {x for x, y in self.constituent_expansion_map.items() if len(y) == 0}


# def get_formula_expansion(expander, formula):
#     ec = ExpandConstituents(expander)
#     ec.expand_formula(formula)
#     return ec.getExpansionLeaves()


class CasusHappening(Enum):
    EQUIVALENT = 0
    EXCLUSIVES = 1
    INDIFFERENT = 2
    NONE = 3
    GENERAL_IMPLICATION = 8
    LOSE_SPEC_IMPLICATION = 9
    INSTANTIATION_IMPLICATION = 10
    MISSING_1ST_IMPLICATION = 12


def isImplication(x):
    return x == CasusHappening.GENERAL_IMPLICATION or x == CasusHappening.LOSE_SPEC_IMPLICATION or x == CasusHappening.INSTANTIATION_IMPLICATION or x == CasusHappening.MISSING_1ST_IMPLICATION


d_transformCaseWhenOneArgIsNegated = None


def transformCaseWhenOneArgIsNegated(orig: CasusHappening):
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


def compare_variable(d, lhs, rhs, kb):
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
        val = transformCaseWhenOneArgIsNegated(compare_variable(d, lhs.arg, rhs, kb))
    elif isinstance(rhs, FNot):
        val = transformCaseWhenOneArgIsNegated(compare_variable(d, lhs, rhs.arg, kb))
    else:
        assert isinstance(lhs, FVariable)
        assert isinstance(rhs, FVariable)
        nameEQ = kb.name_eq(lhs.name, rhs.name)
        specEQ = kb.name_eq(lhs.specification, rhs.specification)
        copCompareInv = compare_variable(d, rhs.cop, lhs.cop, kb)
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
                # if rhs.specification is None:
                #     val = CasusHappening.LOSE_SPEC_IMPLICATION
                # el
                if specEQ == CasusHappening.MISSING_1ST_IMPLICATION:
                    val = CasusHappening.INSTANTIATION_IMPLICATION
                else:
                    val = specEQ
                #
                # if specEQ == CasusHappening.MISSING_1ST_IMPLICATION:
                #     val = CasusHappening.LOSE_SPEC_IMPLICATION
                # else:
                #     val = specEQ
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
    # if (rhs.specification is None) and (lhs.specification is None):
    #     d[cp] = CasusHappening.INDIFFERENT
    # else:
    #     nameSpecEQ = kb.name_eq(lhs.name, rhs.specification)
    #     nameSpec2EQ = kb.name_eq(lhs.specification, rhs.specification)
    #
    #     raise ValueError("More refined comparison in ExpandConstituents::compare_variable: YET TO BE IMPLEMENTED!")
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


def test_pairwise_sentence_similarity(d, x, y, store=True, kb=None, shift=True):
    if shift:
        if (y, x) in d:
            test_shift = d[(y, x)]
        else:
            test_shift = test_pairwise_sentence_similarity(d, y, x, store, kb, False)
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
        val = test_pairwise_sentence_similarity(d, x.arg, y.arg, False, kb)
        if isImplication(val):
            val = CasusHappening.INDIFFERENT
    elif (x == FNot(y)) or (y == FNot(x)):
        val = CasusHappening.EXCLUSIVES
    elif isinstance(x, FNot):
        val = transformCaseWhenOneArgIsNegated(test_pairwise_sentence_similarity(d, x.arg, y, False, kb))
    elif isinstance(y, FNot):
        val = transformCaseWhenOneArgIsNegated(test_pairwise_sentence_similarity(d, x, y.arg, False, kb))
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
                            keyCmp[key].add(compare_variable(d, xx, yy, kb))
                            keyCmpInv[key].add(compare_variable(d, yy, xx, kb))
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
                    srcCmp = compare_variable(d, x.src, y.src, kb)
                    if (srcCmp == CasusHappening.INDIFFERENT):
                        val = CasusHappening.INDIFFERENT
                    else:
                        dstCmp = compare_variable(d, x.dst, y.dst, kb)
                        if (dstCmp == CasusHappening.INDIFFERENT):
                            val = CasusHappening.INDIFFERENT
                        else:
                            keyComparisonOutcome = compare_variable(d, x.src, y.src, kb)
                            copKeyComparisonOutcome = compare_variable(d, x.src.cop, y.src.cop, kb)
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
                    val = compare_variable(d, x.arg, y.arg, kb)
                keyComparisonOutcome = compare_variable(d, x.arg, y.arg, kb)
                copKeyComparisonOutcome = compare_variable(d, x.arg.cop, y.arg.cop, kb)
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


def instantiate_rules(e, constituents, expansion_dictionary, final_constituents, isImpl):
    for constituent in constituents:
        s = set(e(constituent, isImpl))
        s.add(constituent)
        expansion_dictionary[constituent] = s
    for y in expansion_dictionary.values():
        final_constituents = final_constituents.union(set(y))
    # return {(x, y): CasusHappening.NONE for x in final_constituents for y in
    #         final_constituents}


class ExpandConstituents:
    def __init__(self, cfg, expander, list_of_impl_rules):
        print("Setting up the rule expander...")
        self.expander = expander

        self.constituents = list(list_of_impl_rules)

        if (os.path.exists(cfg['hand_dataset'] + "_ied.pickle") and
                os.path.exists(cfg['hand_dataset'] + "_ic.pickle") and
                os.path.exists(cfg['hand_dataset'] + "_eed.pickle") and
                os.path.exists(cfg['hand_dataset'] + "_ec.pickle")):
            with open(cfg['hand_dataset'] + "_ied.pickle", "rb") as f:
                self.impl_expansion_dictionary = pickle.load(f)
            with open(cfg['hand_dataset'] + "_ic.pickle", "rb") as f:
                self.impl_constituents = pickle.load(f)
            with open(cfg['hand_dataset'] + "_eed.pickle", "rb") as f:
                self.eq_expansion_dictionary = pickle.load(f)
            with open(cfg['hand_dataset'] + "_ec.pickle", "rb") as f:
                self.eq_constituents = pickle.load(f)
        else:
            self.impl_expansion_dictionary = dict()
            self.impl_constituents = set()
            # self.constituents_eq = list(list_of_impl_rules)
            self.eq_expansion_dictionary = dict()
            self.eq_constituents = set()

            if not all(map(lambda x: isinstance(x, FBinaryPredicate) or isinstance(x, FUnaryPredicate),
                           self.constituents)):
                raise ValueError(
                    "Error: all the rules within the set of rules must represent Predicates to be assessed, be them unary or binary")

            # Expanding the constituents
            # self.outcome_implication_dictionary =
            instantiate_rules(self.expander, self.constituents, self.impl_expansion_dictionary, self.impl_constituents,
                              True)
            # self.outcome_eq_dictionary =
            instantiate_rules(self.expander, self.constituents, self.eq_expansion_dictionary, self.eq_constituents,
                              False)
            with open(cfg['hand_dataset'] + "_ied.pickle", "wb") as f:
                pickle.dump(self.impl_expansion_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(cfg['hand_dataset'] + "_ic.pickle", "wb") as f:
                pickle.dump(self.impl_constituents, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(cfg['hand_dataset'] + "_eed.pickle", "wb") as f:
                pickle.dump(self.eq_expansion_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(cfg['hand_dataset'] + "_ec.pickle", "wb") as f:
                pickle.dump(self.eq_constituents, f, protocol=pickle.HIGHEST_PROTOCOL)

        # with open("/home/giacomo/dump_impl.json", "w") as f:
        #     from gsmtosimilarity.graph_similarity import EnhancedJSONEncoder
        #     import json
        #     json.dump({str(k):[str(x) for x in v] for k,v in self.impl_expansion_dictionary.items()}, f, cls=EnhancedJSONEncoder, indent=4)
        # with open("/home/giacomo/dump_eq.json", "w") as f:
        #     from gsmtosimilarity.graph_similarity import EnhancedJSONEncoder
        #     import json
        #     json.dump({str(k):[str(x) for x in v] for k,v in self.eq_expansion_dictionary.items()}, f, cls=EnhancedJSONEncoder, indent=4)
        # exit(101)
        self.result_cache = dict()
        self.ms = ModelSearch(self.expander.g)
        # exit(102)
        # for x in self.impl_constituents:
        #     for y in self.eq_constituents:
        #         test_pairwise_sentence_similarity(self.outcome_implication_dictionary, x, y, True, self.expander.g)

    def determine(self, i: int, j: int):
        if (i == j):
            self.result_cache[(i, j)] = PairwiseCases.Equivalent
        assert i < len(self.constituents)
        assert j < len(self.constituents)
        if (i, j) in self.result_cache:
            return self.result_cache[(i, j)]
        val = PairwiseCases.NonImplying

        lhsOrig = ModelSearchBasis(self.constituents[i], self.impl_expansion_dictionary[self.constituents[i]])
        rhsOrig = ModelSearchBasis(self.constituents[j], self.eq_expansion_dictionary[self.constituents[j]])
        tmp = self.ms.compare(lhsOrig, rhsOrig)
        if tmp == CasusHappening.EXCLUSIVES:
            val = PairwiseCases.MutuallyExclusive
        elif tmp == CasusHappening.EQUIVALENT:
            val = PairwiseCases.Equivalent
        elif isImplication(tmp):
            val = PairwiseCases.Implying
        else:
            val = PairwiseCases.NonImplying

        # expansionLeft = self.impl_expansion_dictionary[self.constituents_impl[i]]
        # y = self.constituents_impl[j]
        # # expansionRight = self.expansion_dictionary[self.set_of_rules[j]]
        # result = set()
        # for x in expansionLeft:
        #     #for y in expansionRight:
        #         assert (x,y) in self.outcome_implication_dictionary
        #         tmp = self.outcome_implication_dictionary[(x,y)]
        #         if tmp == CasusHappening.EXCLUSIVES:
        #             val = PairwiseCases.MutuallyExclusive
        #             break
        #         else:
        #             result.add(tmp)
        #     # if val == PairwiseCases.MutuallyExclusive:
        #     #     break
        # if val != PairwiseCases.MutuallyExclusive:
        #     if CasusHappening.EQUIVALENT in result:
        #         val = PairwiseCases.Equivalent
        #     elif CasusHappening.GENERAL_IMPLICATION in result:
        #         val = PairwiseCases.Implying
        #     else:
        #         val = PairwiseCases.NonImplying
        self.result_cache[(i, j)] = val
        return val
