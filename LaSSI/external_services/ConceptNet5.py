__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

import math
import os.path
import urllib

from LaSSI.utils.SimplifiedFuzzyStringMatching import SimplifiedFuzzyStringMatching


def kernel_n(x, y, K):
    return float(K(x, y) / math.sqrt(K(x, x) * K(y, y)))


# Create ConceptNetServiceClass similar to GeoNames
class ConceptNetService(object):

    def __init__(self, psql, nlp):
        from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import DBFuzzyStringMatching
        self.s = DBFuzzyStringMatching(psql, "conceptnet")
        self.nlp = nlp  # StanzaService().nlp_token

    def get_value(self, x):
        return self.name_to_id[x]

    def resolve_u(self, recallThreshold, precisionThreshold, s, type):
        from LaSSI.ner.ResolveMultiEntity import ResolveMultiNamedEntity
        ar = ResolveMultiNamedEntity(recallThreshold, precisionThreshold, "conceptnet5")
        return ar.start(s, self.s, self, self.nlp, type)



class ConceptNet5Numberbatch:
    def __init__(self, lan, minTheta):
        import pandas as pd
        import numpy
        if lan is None:
            lan = "en"
        self.minTheta = minTheta
        self.dictionary = SimplifiedFuzzyStringMatching()
        self.mini_h5 = os.path.join("catabolites", "mini.h5")
        if not os.path.exists(self.mini_h5):
            print("Downloading Mini HDF5 data...")
            urllib.request.urlretrieve(
                "http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/19.08/mini.h5", self.mini_h5)
        self.f = pd.read_hdf(self.mini_h5, 'mat', encoding='utf-8')
        self.data_pts = dict()
        for x in self.f.index:
            concept = x.split("/")
            if concept[2] == lan:
                key = concept[3].replace("_", " ")
                idx, present = self.dictionary.put(key)
                if not present:
                    self.data_pts[idx] = (numpy.array(self.f.loc[x]))

    def get_embedding(self, x):
        return self.dictionary.fuzzyMatch(self.minTheta, x)

    def string_similarity(self, x, y):
        from sentence_transformers import util
        L = {z for k, v in self.get_embedding(x).items() for z in v}
        R = {z for k, v in self.get_embedding(y).items() for z in v}
        K = util.pairwise_dot_score
        finalScore = 0.0
        for inL in L:
            vec = self.data_pts[inL]
            for inR in R:
                score = kernel_n(vec, self.data_pts[inR], K)  # Compute vector similarity
                if score > finalScore:
                    finalScore = score
        return finalScore


if __name__ == "__main__":
    # Example
    cn5n = ConceptNet5Numberbatch("en", 0.8)
    print(cn5n.string_similarity("cat", "mouse"))