from nltk import WordNetLemmatizer

import LaSSI.Parmenides.paremenides


class Services:
    __instance = None

    @staticmethod
    def getInstance(logger=None):
        """ Static access method. """
        if Services.__instance == None:
            Services(logger)
        return Services.__instance

    def getParmenides(self):
        return self.parmenides

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
        return {word.lemma for sent in doc.sentences for word in sent.words}

    def getStanzaSTNLP(self):
        return self.stanza.stNLP

    def resolveTimeUnits(self, sentences):
        return self.old_java_Service.getTimeUnits(sentences)

    def getWTLemmatizer(self):
        return self.lemmatizer

    def getConcepts(self):
        return self.conceptnet

    def getGSMString(self, sentences):
        return self.old_java_Service.generateGSMDatabase(sentences)

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
            self.logger = logger
            self.logger("init parmenides")
            self.parmenides = LaSSI.Parmenides.paremenides.Parmenides()
            self.logger("retrieving postgres")
            self.postgres = FuzzyStringMatchDatabase.instance()
            self.logger("init stanza")
            self.stanza = StanzaService()
            self.logger("init geonames")
            self.geonames = GeoNamesService(self.postgres, self.stanza.nlp_token)
            self.logger("init conceptnet")
            self.conceptnet = ConceptNetService(self.postgres, self.stanza.nlp_token)
            self.logger("init old java service")
            self.old_java_Service = OldWrapper.getInstance()
            self.logger("init WordNet Lemmatizer")
            self.lemmatizer = WordNetLemmatizer()
            Services.__instance = self
