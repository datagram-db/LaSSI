import csv
import os


from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import DBFuzzyStringMatching, FuzzyStringMatchDatabase


# from crawltogsm.write_to_log import write_to_log
# from CleanPipeline import CleanPipeline
# from crawltogsm.write_to_log import write_to_log
# from gsmtosimilarity.conceptnet.SimplifiedFuzzyStringMatching import SimplifiedFuzzyStringMatching
# from gsmtosimilarity.database.FuzzyStringMatchDatabase import DBFuzzyStringMatching, FuzzyStringMatchDatabase
# from gsmtosimilarity.resolve_multi_entity import ResolveMultiNamedEntity
# from gsmtosimilarity.stanza_pipeline import StanzaService


class Admin5:
    def __init__(self):
        self.geoIdToAdmin5 = {}

        with open(os.path.join("adminCode5.txt"), 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                geoId = row[0].strip()
                admin5 = row[1].strip()
                self.geoIdToAdmin5[geoId] = admin5

    @staticmethod
    def instance():
        if not hasattr(Admin5, 'adm5'):
            Admin5.adm5 = Admin5()
        return Admin5.adm5

    def get_admin5(self, geoId):
        return self.geoIdToAdmin5.get(geoId)


class GeoNameField:
    def __init__(self, lin):
        line = lin.split("\t")
        self.isADM = False
        self.geonameid = line[0]
        self.feature_class = line[6]
        self.feature_code = line[7]
        self.country_code = self.rectify(line[8])
        self.cont = True
        self.path = ""
        self.analyse(self.country_code)
        self.adm1 = self.rectify(line[10])
        self.analyse(self.adm1)
        self.adm2 = self.rectify(line[11])
        self.analyse(self.adm2)
        self.adm3 = self.rectify(line[12])
        self.analyse(self.adm3)
        self.adm4 = self.rectify(line[13])
        self.analyse(self.adm4)
        self.adm5 = Admin5.instance().get_admin5(self.geonameid)
        self.analyse(self.adm5)
        if self.path.startswith("\t"):
            self.path = self.path[1:]
        if not self.isADM:
            self.path = self.path + "\t" + "P" + str(int(self.geonameid)).zfill(10)
        self.name = line[1]
        self.ascii = line[2]
        self.others = line[3].split(",")

    @staticmethod
    def rectify(x):
        x = x.strip()
        return x if x else None

    def analyse(self, x):
        if self.cont and x is not None:
            self.path += "\t" + x
        else:
            self.cont = False


class GeoNamesService():
    # _instance = None

    def __init__(self, psql, nlp):
        self.s = DBFuzzyStringMatching(psql, "geonames")
        self.nlp = nlp
        #from LaSSI.external_services.Stanza import StanzaService
        # self.s = SimplifiedFuzzyStringMatching()
        # self.name_to_id = dict()
        # if file is not None and os.path.exists(file):
        #     with open(file) as f:
        #         for line in f.readlines():
        #             g = GeoNameField(line)
        #             self.no_file_init(g.name, g.geonameid)
        #             self.no_file_init(g.ascii, g.geonameid)
        #             for x in g.others:
        #                 self.no_file_init(x, g.geonameid)

    # def no_file_init(self, x, id):
    #     if x not in self.name_to_id:
    #         self.name_to_id[x.lower()] = set()
    #     self.name_to_id[x.lower()].add(id)
    #     self.s.put(x)
    #
    # def get_value(self, x):
    #     return self.name_to_id[x]

    def resolve_u(self, recallThreshold, precisionThreshold, s, type):
        from LaSSI.ner.resolve_multi_entity import ResolveMultiNamedEntity
        # if self.nlp is None:
        #     from LaSSI.external_services.Services import Services
        #     self.nlp = Services.getInstance().getStanzaNLPToken().nlp_token
        ar = ResolveMultiNamedEntity(recallThreshold, precisionThreshold)
        return ar.start(s, self.s, self, self.nlp, type)

    # def __new__(cls):
    #     if cls._instance is None:
    #         # write_to_log(None, "Initialising GeoNames...")
    #         cls._instance = super(GeoNamesService, cls).__new__(cls)
    #
    #     return cls._instance


# if __name__ == "__main__":
#     stringa = "Newcastle city centre is trafficked"
#     s = GeoNamesService()
#     s.no_file_init("Newcastle Upon Tyne", "n/t/uk/e/w")
#     s.no_file_init("Tyne and Wear", "t/uk/e/w")
#     s.no_file_init("London", "l/uk/e/w")
#     s.no_file_init("Rome", "r/l/i/e/w")
#     print(s.resolve_u(0.1, 0.6, stringa))
