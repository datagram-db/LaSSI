__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

from functools import lru_cache

from LaSSI.similarities.HuggingFace import HuggingFace
from LaSSI.similarities.TwoGrams import TwoGramSetSimilarity
from LaSSI.similarities.levenshtein import MultiLevenshtein, lev
from LaSSI.utils.configurations import LegacySemanticConfiguration


# from gsmtosimilarity.TwoGrams import TwoGramSetSimilarity
# from gsmtosimilarity.conceptnet.ConceptNet5 import ConceptNet5Numberbatch
# from gsmtosimilarity.huggingface.HuggingFace import HuggingFace
# from gsmtosimilarity.levenshtein import lev, MultiLevenshtein


class StringSimilarity(object):
    def __init__(self, cfg:LegacySemanticConfiguration, selector):
        self.cfg = cfg
        self.hugging = self.concept = self.pooling = None
        kind = getattr(cfg, selector)
        if kind == 'HuggingFace':
            self.hugging = HuggingFace(self.cfg.HuggingFace)
        elif kind == 'ConceptNetPooling':
            from LaSSI.external_services.ConceptNet5 import ConceptNet5Numberbatch
            self.concept = ConceptNet5Numberbatch(self.cfg.ConceptNet5Numberbatch_lan,
                                                  self.cfg.ConceptNet5Numberbatch_minTheta)
        elif kind == 'Prevailing':
            self.pooling = StringSimilarity(cfg, getattr(self.cfg, f"prevailing_{selector}"))
        self.kind = kind


    def string_similarity(self, x, y):
        if self.kind == "Levenshtein":
            return lev(x,y)
        elif self.kind == 'MultiLevenshtein':
            return MultiLevenshtein(x,y)
        elif self.kind == 'HuggingFace':
            return self.hugging.string_similarity(x,y)
        elif self.kind == 'ConceptNetPooling':
            return self.concept.string_similarity(x, y)
        elif self.kind == 'Prevailing':
            return self.semanticPrevail(x,y)

    def semanticPrevail(self, x, y):
        ml = MultiLevenshtein(x, y)
        sk = TwoGramSetSimilarity(x,y)
        score = max(ml,sk)
        if score>0.85:
            return score
        return self.pooling.string_similarity(x, y)










