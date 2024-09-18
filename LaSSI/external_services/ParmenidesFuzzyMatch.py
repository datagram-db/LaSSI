__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

import math

class ParmenidesFuzzyMatch(object):

    def __init__(self, psql, nlp, parmo):
        from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import DBFuzzyStringMatching
        self.s = DBFuzzyStringMatching(psql, "parmenides")
        self.nlp = nlp#StanzaService().nlp_token
        self.parmo = parmo

    def resolve_u(self, recallThreshold, precisionThreshold, s):
        from LaSSI.ner.resolve_multi_entity import ResolveMultiNamedEntity
        ar = ResolveMultiNamedEntity(recallThreshold, precisionThreshold, "parmenides", parmo=self.parmo)
        return ar.start(s, self.s, self, self.nlp, None)