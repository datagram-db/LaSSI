from StanfordNLPExtractor.OldWrapper import OldWrapper
from nltk import WordNetLemmatizer

import LaSSI.Parmenides.paremenides
from LaSSI.external_services.Existentials import Existentials
from LaSSI.external_services.ParmenidesFuzzyMatch import ParmenidesFuzzyMatch


class Services:
    __instance = None

    @staticmethod
    def getInstance(logger=None)->'Services':
        """ Static access method. """
        if Services.__instance == None:
            Services(logger)
        if hasattr(Services, '__instance'):
            assert Services.__instance is not None
            return Services.__instance
        elif hasattr(Services, '_Services__instance'):
            assert Services._Services__instance is not None
            return Services._Services__instance
        raise RuntimeError("ERROR: cannot found instantiated field")

    def setParmenides(self, parmenides):
        self.parmenides = parmenides
        from LaSSI.external_services.ParmenidesFuzzyMatch import ParmenidesFuzzyMatch
        self.fuzzyParmenides = ParmenidesFuzzyMatch(self.postgres, self.stanza.nlp_token, self.parmenides)

    def getParmenides(self):
        return self.parmenides

    def getFuzzyParmenides(self):
        return self.fuzzyParmenides

    def getGeoNames(self):
        return self.geonames

    def getStanza(self):
        return self.stanza

    def getStanzaNLPToken(self):
        return self.stanza.nlp_token

    def getStanzaNLP(self):
        return self.stanza.nlp

    def lemmatize_sentence(self, text):
        doc = self.stanza.nlp(text)
        lemmas = set()
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.pos.lower() != 'aux':
                    lemmas.add(word.lemma)
        return lemmas

    def getStanzaSTNLP(self):
        return self.stanza.stNLP

    def resolveTimeUnits(self, sentences):
        if self.old_java_Service is None:
            self.logger("init old java service")
            self.old_java_Service = OldWrapper.getInstance()
        return self.old_java_Service.getTimeUnits(sentences)

    def getWTLemmatizer(self):
        return self.lemmatizer

    def getConcepts(self):
        return self.conceptnet

    def getGSMString(self, sentences):
        if self.old_java_Service is None:
            self.logger("init old java service")
            self.old_java_Service = OldWrapper.getInstance()
        return self.old_java_Service.generateGSMDatabase(sentences)

    def getExistentials(self):
        return self.existentials

    def log(self, message):
        self.logger(message)

    def __init__(self, logger=None):
        """ Virtually private constructor. """
        if Services.__instance != None:
            raise Exception("This class is a singleton!")
        elif logger == None:
            raise Exception("The first initialization should provide a non-None logger!")
        else:
            from LaSSI.external_services.Stanza import StanzaService
            from LaSSI.external_services.GeoNames import GeoNamesService
            from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import FuzzyStringMatchDatabase
            from LaSSI.external_services.ConceptNet5 import ConceptNetService
            from StanfordNLPExtractor.OldWrapper import OldWrapper
            from LaSSI.external_services.ParmenidesFuzzyMatch import ParmenidesFuzzyMatch
            self.logger = logger
            self.logger("init parmenides")
            self.parmenides = None
            self.logger("retrieving postgres")
            self.postgres = FuzzyStringMatchDatabase.instance()
            self.logger("init stanza")
            self.stanza = StanzaService()
            self.logger("init geonames wrapper")
            self.geonames = GeoNamesService(self.postgres, self.stanza.nlp_token)
            self.logger("init conceptnet wrapper")
            self.conceptnet = ConceptNetService(self.postgres, self.stanza.nlp_token)
            self.logger("init fuzzyParmenides wrapper")
            self.fuzzyParmenides = None
            self.logger("init old java service")
            self.old_java_Service = None
            self.logger("init WordNet Lemmatizer")
            self.lemmatizer = WordNetLemmatizer()
            self.logger("init existentials")
            self.existentials = Existentials()
            Services.__instance = self
