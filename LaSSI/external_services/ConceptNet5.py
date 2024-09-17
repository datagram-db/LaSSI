__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"
import math
import os.path
import urllib.request

import numpy
# import pandas as pd
# from sentence_transformers import util
#
# from crawltogsm.write_to_log import write_to_log
# from gsmtosimilarity.conceptnet.SimplifiedFuzzyStringMatching import SimplifiedFuzzyStringMatching
# from gsmtosimilarity.database.FuzzyStringMatchDatabase import DBFuzzyStringMatching, FuzzyStringMatchDatabase
# from gsmtosimilarity.resolve_multi_entity import ResolveMultiNamedEntity
# from gsmtosimilarity.stanza_pipeline import StanzaService


def kernel_n(x, y, K):
    return float(K(x, y) / math.sqrt(K(x, x) * K(y, y)))


# Create ConceptNetServiceClass similar to GeoNames
class ConceptNetService(object):
    _instance = None

    def __init__(self, psql, nlp):
        # from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import DBFuzzyStringMatching
        # from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import FuzzyStringMatchDatabase
        from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import DBFuzzyStringMatching
        self.s = DBFuzzyStringMatching(psql, "conceptnet")
        # self.name_to_id = dict()
        self.nlp = nlp#StanzaService().nlp_token
        #
        # if not os.path.exists("mini.h5"):
        #     print("Downloading Mini HDF5 data...")
        #     urllib.request.urlretrieve(
        #         "http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/19.08/mini.h5",
        #         "mini.h5")
        # self.f = pd.read_hdf("mini.h5", 'mat', encoding='utf-8')
        # self.data_pts = dict()
        # for x in self.f.index:
        #     concept = x.split("/")
        #     if concept[2] == 'en':
        #         key = concept[3].replace("_", " ")
        #         self.no_file_init(key, x)

    def no_file_init(self, x, id):
        if x not in self.name_to_id:
            self.name_to_id[x.lower()] = set()
        self.name_to_id[x.lower()].add(id)
        self.s.put(x)

    def get_value(self, x):
        return self.name_to_id[x]

    def resolve_u(self, recallThreshold, precisionThreshold, s, type):
        from LaSSI.ner.resolve_multi_entity import ResolveMultiNamedEntity
        ar = ResolveMultiNamedEntity(recallThreshold, precisionThreshold)
        return ar.start(s, self.s, self, self.nlp, type)

    # def __new__(cls):
    #     if cls._instance is None:
    #         # write_to_log(None, "Initialising ConceptNet...")
    #         cls._instance = super(ConceptNetService, cls).__new__(cls)
    #
    #     return cls._instance


# class ConceptNet5Numberbatch:
#     def __init__(self, lan, minTheta):
#         if lan is None:
#             lan = "en"
#         self.minTheta = minTheta
#         self.dictionary = SimplifiedFuzzyStringMatching()
#         if not os.path.exists("mini.h5"):
#             print("Downloading Mini HDF5 data...")
#             urllib.request.urlretrieve(
#                 "http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/19.08/mini.h5", "mini.h5")
#         self.f = pd.read_hdf("mini.h5", 'mat', encoding='utf-8')
#         self.data_pts = dict()
#         for x in self.f.index:
#             concept = x.split("/")
#             if concept[2] == lan:
#                 key = concept[3].replace("_", " ")
#                 idx, present = self.dictionary.put(key)
#                 if not present:
#                     self.data_pts[idx] = (numpy.array(self.f.loc[x]))
#
#     def get_embedding(self, x):
#         return self.dictionary.fuzzyMatch(self.minTheta, x)
#
#     def string_similarity(self, x, y):
#         L = {z for k, v in self.get_embedding(x).items() for z in v}
#         R = {z for k, v in self.get_embedding(y).items() for z in v}
#         K = util.pairwise_dot_score
#         finalScore = 0.0
#         for inL in L:
#             vec = self.data_pts[inL]
#             for inR in R:
#                 score = kernel_n(vec, self.data_pts[inR], K)  # Compute vector similarity
#                 if score > finalScore:
#                     finalScore = score
#         return finalScore


# if __name__ == "__main__":
#     # Example
#     cn5n = ConceptNet5Numberbatch("en", 0.8)
#     print(cn5n.string_similarity("cat", "mouse"))
