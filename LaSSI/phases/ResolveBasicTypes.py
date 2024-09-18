__author__ = "Oliver Robert Fox, Giacomo Bergami"
__copyright__ = "Copyright 2020, Giacomo Bergami"
__credits__ = ["Oliver Robert Fox", "Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

from LaSSI.external_services.Services import Services
from LaSSI.structures.meuDB.meuDB import MeuDBEntry, MeuDB


class ResolveBasicTypes():
    def __init__(self, recall_threshold:float, precision_threshold:float):
        self.recall_threshold = recall_threshold
        self.precision_threshold = precision_threshold
        self.services = Services.getInstance()
        self.stanza_service = self.services.getStanzaNLP()

    def resolve_basic_types(self, list_sentences):
        db = list()
        for idx, (sentence, withTime) in enumerate(zip(list_sentences, self.services.resolveTimeUnits(list_sentences))):
            entities = []
            multi_entity_unit = []
            for x in self.services.getFuzzyParmenides().resolve_u(self.recall_threshold, self.precision_threshold, sentence):
                multi_entity_unit.append(x)

            ## 1) Time Parsing
            for time in withTime:
                time = MeuDBEntry.from_dict_with_src(time, "SUTime")
                multi_entity_unit.append(time)

            for x in self.services.getGeoNames().resolve_u(self.recall_threshold, self.precision_threshold, sentence, "GPE"):
                multi_entity_unit.append(x)

            for x in self.services.getConcepts().resolve_u(self.recall_threshold, self.precision_threshold, sentence, "ENTITY"):
                multi_entity_unit.append(x)

            ## 2) Typed entity parsing
            results = self.stanza_service(sentence)
            for ent in results.ents:
                # monad = ""
                entity = ent.text
                monad = entity.replace(" ", "")
                if ent.type == "ORG":  # Remove spaces to create one word 'ORG' entities
                    entities.append([entity, monad])
                from LaSSI.similarities.levenshtein import lev
                multi_entity_unit.append(MeuDBEntry(ent.text,ent.type,ent.start_char,ent.end_char,monad,lev(monad.lower(), ent.text.lower()),"Stanza"))

            # Loop through all entities and replace in sentence before passing to NLP server
            for entity in entities:
                sentence = sentence.replace(entity[0], entity[1])

            db.append(MeuDB(sentence, multi_entity_unit))
        return db

def ExplainTextWithNER(self, sentences):
    return ResolveBasicTypes(self.recall_threshold, self.precision_threshold).resolve_basic_types(sentences)