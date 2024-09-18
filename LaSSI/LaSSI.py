__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2020, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import collections
import io
import json
import os.path

import pkg_resources

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.external_services.Services import Services
from LaSSI.external_services.utilities.DatabaseConfiguration import DatabaseConfiguration, load_db_configuration
from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import FuzzyStringMatchDatabase
from LaSSI.external_services.web_cralwer.ScraperConfiguration import ScraperConfiguration
from LaSSI.files.JSONDump import json_dumps
from LaSSI.ner.ResolveBasicTypes import ResolveBasicTypes
from LaSSI.structures.meuDB.meuDB import MeuDB


class LaSSI():
    def __init__(self, dataset_name:str,
                       fuzzyDBs:str|DatabaseConfiguration,
                       transformation:SentenceRepresentation=SentenceRepresentation.Logical,
                        sentences: ScraperConfiguration|str|collections.abc.Iterable=None,
                       logger=None,
                       web_dir=None,
                       recall_threshold=0.1,
                       precision_threshold=0.8,
                       force=False
                 ):
        self.dataset_name = dataset_name
        self.web_dir = web_dir
        if logger is None:
            logger = lambda x: print(x)
        self.logger = logger

        self.logger("init postgres")
        if not isinstance(fuzzyDBs, DatabaseConfiguration):
            fuzzyDBs = str(fuzzyDBs)
            fuzzyDBs = load_db_configuration(fuzzyDBs)

        self.logger("init non-postgres services and the wrapper for the former...")
        self.initServices = Services.getInstance(self.logger)

        self.logger("init postgres services...")
        self.logger(" - Initialising the connection to the database")
        (FuzzyStringMatchDatabase
         .instance()
         .init(fuzzyDBs.db, fuzzyDBs.uname, fuzzyDBs.pw, fuzzyDBs.host, fuzzyDBs.port))

        self.logger(" - Loading the tab files or streaming those remotely, if required.")
        import tempfile
        with tempfile.NamedTemporaryFile() as parmenides_tab:
            with open(parmenides_tab.name, 'w') as f:
                self.initServices.getParmenides().dumpTypedObjectsToTAB(f)
            FuzzyStringMatchDatabase.instance().create_typed_table("parmenides", parmenides_tab.name)
        for k, v in fuzzyDBs.fuzzy_dbs.items():
            self.logger(f" - Loading {k}.")
            FuzzyStringMatchDatabase.instance().create(k, v)

        if sentences is None:
            sentences = open(self.dataset_name, "r")
        self.sentences = sentences
        self.recall_threshold = recall_threshold
        self.precision_threshold = precision_threshold
        self.transformation = transformation
        self.force = force
        self.logger("init file structure")
        from pathlib import Path
        self.catabolites = os.path.join("catabolites", self.dataset_name)
        self.catabolites_viz = os.path.join(self.catabolites, "viz")
        self.internals = os.path.join(self.catabolites, "internals")
        self.internal_graphs_path = os.path.join(self.internals, "graphs")
        self.internal_kernels_path = os.path.join(self.internals, "kernels")
        Path(self.catabolites).mkdir(parents=True, exist_ok=True)
        Path(self.catabolites_viz).mkdir(parents=True, exist_ok=True)
        self.meuDB = os.path.join(self.catabolites, "meuDBs.json")
        self.gsmDB = os.path.join(self.catabolites, "gsmDB.txt")
        self.datagramdb_output = os.path.join(self.catabolites, "datagramdb_output.json")
        self.query_file = pkg_resources.resource_filename("LaSSI.resources", "gsm_query.txt")

    def apply_graph_grammars(self, n):
        from PyDatagramDB import DatagramDB
        d = DatagramDB(self.gsmDB,
                       self.query_file,
                       self.catabolites_viz,
                       isSerializationFull=True,
                       opt_data_schema="pos\nSizeTAtt\nbegin\nSizeTAtt\nend\nSizeTAtt")
        d.run()
        L = []
        for result_graph_file in map(lambda x: os.path.join(self.catabolites_viz, str(x), "result.json"), range(n)):
            with open(result_graph_file, "r") as f:
                raw_json_graph = json.load(f)
                L.append(raw_json_graph)
        if self.web_dir is not None:
            import shutil
            dataset_folder = os.path.join(self.web_dir, "dataset","data") #f"{self.web_dir}/dataset/data"
            if os.path.exists(dataset_folder):
                shutil.rmtree(dataset_folder)
            shutil.copytree(self.catabolites_viz, dataset_folder)

        return L

    def _internal_graph(self, meuDBs, graph_list):
        internal_graphs = []
        sentences = []
        for graph, meuDB in zip(graph_list, meuDBs):
            from LaSSI.structures.provenance.GraphProvenance import GraphProvenance
            g = GraphProvenance(graph, meuDB, self.transformation == SentenceRepresentation.SimpleGraph)
            internal_graph = g.internal_graph()
            sentence = g.sentence()
            internal_graphs.append(internal_graph)
            sentences.append(sentence)



    def sentence_transform(self, sentences):
        if self.transformation != SentenceRepresentation.FullText:
            # from LaSSI.ner.a_resolve_basic_types import ResolveBasicTypes
            from LaSSI.files.FileDumpUtilities import target_file_dump
            n = len(sentences)
            self.logger("generating meuDB")
            meuDB = target_file_dump(self.meuDB,
                             lambda x: [MeuDB.from_dict(k) for k in json.load(x)],
                             lambda: ResolveBasicTypes(self.recall_threshold, self.precision_threshold).resolve_basic_types(sentences),
                             json_dumps,
                             self.force)

            self.logger("generating gsmDB")
            gsmDB = target_file_dump(self.gsmDB,
                                     lambda x: x.read(),
                                     lambda : self.initServices.getGSMString(sentences),
                                     lambda x: x,
                                     self.force)

            self.logger("generating rewritten graphs")
            rewrittenGraphs = target_file_dump(self.datagramdb_output,
                                               json.load,
                                               lambda: self.apply_graph_grammars(n),
                                               json_dumps,
                                               self.force)

            self.logger("rewriting graphs")
            self._internal_graph(meuDB, rewrittenGraphs)
            return []
        else:
            return sentences

    def run(self):
        from LaSSI.SentenceLoader import SentenceLoader
        sentences = SentenceLoader(self.sentences)
        result = self.sentence_transform(sentences)

    def close(self):
        if isinstance(self.sentences, io.IOBase):
            self.sentences.close()
            self.logger("~~DONE~~")

