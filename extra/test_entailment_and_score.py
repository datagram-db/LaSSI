from LaSSI.Parmenides.Parmenides import ParmenidesSingleton
from LaSSI.Parmenides.TBox.ExpandConstituents import test_pairwise_sentence_similarity

from LaSSI.structures.extended_fol.Formulae import FUnaryPredicate, FVariable, FNot, FAnd
from extra.test_allex import ncc, ncl

traffic = FVariable('traffic', 'ENTITY')
t = FUnaryPredicate("be", traffic, 1)
tcc = FUnaryPredicate("be", traffic, 2, frozenset({"SPACE": (ncc,)}.items()))
tn = FUnaryPredicate("be", traffic, 2, frozenset({"SPACE": (ncl,)}.items()))
n_tcc = FNot(tcc)

s_3 = FAnd((t, n_tcc))

if __name__ == "__main__":
    # print(s_3)
    # val = test_pairwise_sentence_similarity({}, t, n_tcc, shift=False)
    # print(val)
    # val = test_pairwise_sentence_similarity({}, tcc, t, shift=False)
    # print(val)
    # val = test_pairwise_sentence_similarity({}, n_tcc, t, shift=False)
    # print(val)
    # val = test_pairwise_sentence_similarity({}, t, tcc, shift=False)
    # print(val)
    ParmenidesSingleton.instance()
    ParmenidesSingleton.init("/home/giacomo/projects/LaSSI/catabolites", "giacomo", "omocaig",
                             "localhost", 5432, False, "/home/giacomo/projects/LaSSI/parmenides.ttl")

    val = test_pairwise_sentence_similarity({}, tn, tcc, shift=False)
    print(val)
    val = test_pairwise_sentence_similarity({}, tcc, tn, shift=False)
    print(val)