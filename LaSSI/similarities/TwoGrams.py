__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"
import math

# from gsmtosimilarity.conceptnet.ConceptNet5 import ConceptNet5Numberbatch
from gsmtosimilarity.huggingface.HuggingFace import HuggingFace
from gsmtosimilarity.levenshtein import lev, MultiLevenshtein


def get2Grams(x):
    pairs = set()
    for xX in x.split("\s+"):
        numPairs = len(xX) - 1
        if numPairs == 0:
            pairs.add(xX)
        if numPairs < 0:
            numPairs = 0
        for i in range(numPairs):
            pairs.add(xX[i:i + 2])
    return pairs


def TwoGramSetSimilarity(x, y):
    if (x == y):
        return 1
    elif (len(x) == 0) or (len(y) == 0):
        return 0.0
    pairs1 = get2Grams(x)
    pairs2 = get2Grams(y)
    return len(pairs1.intersection(pairs2))/math.sqrt(len(pairs1)*len(pairs2))

def TwoGramKernel(x, y):
    grams = get2Grams(x).union(get2Grams(y))
    return sum(map(lambda z: x.count(z)*y.count(z), grams))

def TwoGramNormalKernel(x, y):
    return TwoGramKernel(x,y)/math.sqrt(TwoGramKernel(x,x)*TwoGramKernel(y,y))


def testsims(x,y):
    print(f"{x} vs. {y}")
    print(TwoGramNormalKernel(x, y))
    print(TwoGramKernel(x, y))
    print(TwoGramSetSimilarity(x, y))
    print(lev(x,y))
    print(MultiLevenshtein(x,y))
    # cn = ConceptNet5Numberbatch("en", 0.8)
    # print(cn.string_similarity(x, y))
    hf = HuggingFace()
    print(hf.string_similarity(x, y))


if __name__ == "__main__":
    testsims("mouse", "miuse")
    testsims("begin", "begun")
    testsims("cat", "mouse")
    testsims("cat", "kitten")
    testsims("cat", "housecat")
    testsims("air to earth", "earth to air")
