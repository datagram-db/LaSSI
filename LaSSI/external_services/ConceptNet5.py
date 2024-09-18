__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"
import math

def kernel_n(x, y, K):
    return float(K(x, y) / math.sqrt(K(x, x) * K(y, y)))


# Create ConceptNetServiceClass similar to GeoNames
class ConceptNetService(object):

    def __init__(self, psql, nlp):
        from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import DBFuzzyStringMatching
        self.s = DBFuzzyStringMatching(psql, "conceptnet")
        self.nlp = nlp#StanzaService().nlp_token

    def get_value(self, x):
        return self.name_to_id[x]

    def resolve_u(self, recallThreshold, precisionThreshold, s, type):
        from LaSSI.ner.resolve_multi_entity import ResolveMultiNamedEntity
        ar = ResolveMultiNamedEntity(recallThreshold, precisionThreshold, "conceptnet5")
        return ar.start(s, self.s, self, self.nlp, type)

