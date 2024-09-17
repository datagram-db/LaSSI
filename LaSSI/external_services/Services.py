import LaSSI.Parmenides.paremenides


class Services:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Services.__instance == None:
            Services()
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

    def resolveTimeUnits(self, sentences):
        return self.old_java_Service.getTimeUnits(sentences)

    def getConcepts(self):
        return self.conceptnet

    def getGSMString(self, sentences):
        return self.old_java_Service.generateGSMDatabase(sentences)

    def __init__(self):
        """ Virtually private constructor. """
        if Services.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            from LaSSI.external_services.Stanza import StanzaService
            from LaSSI.external_services.GeoNames import GeoNamesService
            from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import FuzzyStringMatchDatabase
            from LaSSI.external_services.ConceptNet5 import ConceptNetService
            from StanfordNLPExtractor.OldWrapper import OldWrapper

            print("init parmenides")
            self.parmenides = LaSSI.Parmenides.paremenides.Parmenides()
            print("retrieving postgres")
            self.postgres = FuzzyStringMatchDatabase.instance()
            print("init stanza")
            self.stanza = StanzaService()
            print("init geonames")
            self.geonames = GeoNamesService(self.postgres, self.stanza.nlp_token)
            print("init conceptnet")
            self.conceptnet = ConceptNetService(self.postgres, self.stanza.nlp_token)
            print("init old java service")
            self.old_java_Service = OldWrapper.getInstance()

            Services.__instance = self
