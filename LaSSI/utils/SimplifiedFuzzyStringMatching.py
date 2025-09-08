__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__status__ = "Production"
from collections import defaultdict, OrderedDict
from typing import List


class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


class SimplifiedFuzzyStringMatching:
    def __init__(self):
        self.unique_string = TwoWayDict()
        self.shift0_2grams_containment = defaultdict(List[int])
        self.gram_multiplicity = defaultdict(dict)
        self.objectGramSize = defaultdict(int)
        self.stringCount = 0


    def clear(self):
        self.unique_string.clear()
        self.shift0_2grams_containment.clear()
        self.gram_multiplicity.clear()
        self.objectGramSize.clear()

    def compareStringHashmap1(self, str, map, vec):
        numPairs = len(str) - 1
        if (numPairs == 0):
            map[str] = 0
            vec.append(1)
        else:
            if (numPairs < 0):
                numPairs = 0
            singleGrams = 0
            for i in range(numPairs):
                s = str[i:i+2]
                if len(s)>0:
                    if s not in map:
                        map[s] = singleGrams
                        singleGrams += 1
                        vec.append(1)
                    else:
                        vec[map[s]] += 1


    def compareStringHashmap2(self, string, map, vec):
        for token in string.split("\s+"):
            self.compareStringHashmap1(token, map, vec)
        for k in map:
            val = map[k]
            map[k] = vec[val]

    def put(self, s:str):
        s = s.lower()
        if s not in self.unique_string:
            self.gram_multiplicity[s] = defaultdict(int)
            vec = list()
            self.compareStringHashmap2(s, self.gram_multiplicity[s], vec)
            self.objectGramSize[s] = sum(vec)
            for first in self.gram_multiplicity[s]:
                if first not in self.shift0_2grams_containment:
                    self.shift0_2grams_containment[first] = list()
                self.shift0_2grams_containment[first].append(self.stringCount)
            self.unique_string[s] = self.stringCount
            self.stringCount += 1
            return self.stringCount, False
        return self.unique_string[s], True

    def get(self, ss):
        if ss in self.unique_string:
            return self.unique_string[ss]
        return -1

    def compareString_letterPairs(self, str, pairs):
        numPairs = len(str) - 1
        if numPairs == 0:
            pairs.add(str)
        if numPairs<0:
            numPairs = 0
        for i in range(numPairs):
            pairs.add(str[i:i+2])

    def compareString_wordLetterPairs(self, strMixed:str):
        L = set()
        for s in strMixed.split("\s+"):
            self.compareString_letterPairs(s, L)
        return L


    def getTwoGramAndString(self, argument:str, map:dict):
        if argument in self.gram_multiplicity:
            map.clear()
            for k, v in self.gram_multiplicity[argument].items():
                map[k] = v
        else:
            self.compareStringHashmap2(argument, map, list())

    def fuzzyMatch(self, threshold:float, objectString:str):
        poll = OrderedDict()
        objectGrams = self.compareString_wordLetterPairs(objectString)
        candidates = set()
        for gram in objectGrams:
            if gram in self.shift0_2grams_containment:
                for x in self.shift0_2grams_containment[gram]:

                    candidates.add(x)
        m1 = dict()
        self.compareStringHashmap2(objectString, m1, list())
        ogSize = sum(m1.values())
        for k in candidates:
            associatedToElement = self.get(k)
            m2 = dict()
            self.getTwoGramAndString(associatedToElement, m2)
            score = 0.0
            e = 0.0
            greater = m1 if len(m1) > len(m2) else m2
            lesser = m2 if len(m1) > len(m2) else m1
            for k1, v1 in lesser.items():
                if k1 in greater:
                    e += min(v1, greater[k1])*1.0
            leftCount = 0
            retmap = dict()
            retlist = list()
            self.compareStringHashmap2(associatedToElement, retmap, retlist)
            leftCount = sum(retlist)
            score = (e * 2.0) / ((leftCount + ogSize) * 1.0)

            if (threshold >= 0.0) and score >= threshold:
                if score not in poll:
                    poll[score] = set()
                poll[score].add(k)
        return poll