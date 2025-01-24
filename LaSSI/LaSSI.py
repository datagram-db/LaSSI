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
import time

import numpy as np
import pkg_resources

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.external_services.Services import Services
from LaSSI.external_services.utilities.DatabaseConfiguration import DatabaseConfiguration, load_db_configuration
from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import FuzzyStringMatchDatabase
from LaSSI.external_services.web_cralwer.ScraperConfiguration import ScraperConfiguration
from LaSSI.files.JSONDump import json_dumps, obj_unmarshall, obj_pickle
from LaSSI.phases.ApplyGraphGrammars import ApplyGraphGrammars
from LaSSI.phases.CalculateMatrix import CalculateMatrix
from LaSSI.phases.GetGSMString import GetGSMString
from LaSSI.phases.LogicalRewriting import LogicalRewriting
from LaSSI.phases.ResolveBasicTypes import ExplainTextWithNER
from LaSSI.phases.SemanticGraphRewriting import SemanticGraphRewriting
from LaSSI.similarities.graph_similarity import SimilarityScore
from LaSSI.structures.extended_fol.rewrite_kernels import rewrite_kernels
from LaSSI.structures.extended_fol.sentence_expansion import SentenceExpansion
from LaSSI.structures.internal_graph.Graph import Graph
from LaSSI.structures.internal_graph.InternalData import InternalRepresentation
from LaSSI.structures.meuDB.meuDB import MeuDB
from LaSSI.utils.configurations import LegacySemanticConfiguration


class LaSSI():
    def __init__(self, dataset_name: str,
                 fuzzyDBs: str | DatabaseConfiguration,
                 transformation: SentenceRepresentation = SentenceRepresentation.Logical,
                 sentences: ScraperConfiguration | str | collections.abc.Iterable = None,
                 logger=None,
                 web_dir=None,
                 recall_threshold=0.1,
                 precision_threshold=0.8,
                 force=False,
                 should_benchmark=True,
                 legacy_conf: LegacySemanticConfiguration = None
                 ):
        if legacy_conf is None:
            self.legacy_conf = LegacySemanticConfiguration()
        else:
            self.legacy_conf = legacy_conf
        self.string_rep_dir = None
        self.benchmarking_file = None
        self.create_catabolites_dir(dataset_name)
        self.dataset_name = dataset_name
        tmp = f"{self.dataset_name}_clusters.txt"
        if os.path.isfile(tmp):
            self.clusters_file = tmp
        else:
            self.clusters_file = None
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
        self.should_benchmark = should_benchmark
        self.logger("init file structure")
        from pathlib import Path
        self.catabolites = os.path.join("catabolites", self.catabolites_dir)
        self.catabolites_viz = os.path.join(self.catabolites, "viz")
        self.internals = os.path.join(self.catabolites, "internals.json")
        self.logical_rewriting = os.path.join(self.catabolites, "logical_rewriting.json")
        self.confusion_matrices = os.path.join(self.catabolites, "confusion_matrices_")
        Path(self.catabolites).mkdir(parents=True, exist_ok=True)
        Path(self.catabolites_viz).mkdir(parents=True, exist_ok=True)
        self.meuDB = os.path.join(self.catabolites, "meuDBs.json")
        self.gsmDB = os.path.join(self.catabolites, "gsmDB.txt")
        self.datagramdb_output = os.path.join(self.catabolites, "datagramdb_output.json")
        self.query_file = pkg_resources.resource_filename("LaSSI.resources", "gsm_query.txt")
        self.sc = None

    def create_catabolites_dir(self, dataset_name):
        # if "/" in dataset_name:
        #     name_arr = dataset_name.split("/")
        #     self.catabolites_dir = name_arr[len(name_arr) - 1]
        # else:
        #     self.catabolites_dir = dataset_name
        # if "." in self.catabolites_dir:
        #     name_arr = self.catabolites_dir.split(".")
        #     self.catabolites_dir = name_arr[0]
        from pathlib import Path
        self.catabolites_dir = Path(dataset_name).stem
        self.catabolites_of_dataset = os.path.join("catabolites", self.catabolites_dir)
        self.string_rep_dir = os.path.join(self.catabolites_of_dataset, "string_rep.txt")
        self.benchmarking_file = os.path.join("catabolites", "benchmark.csv")
        if os.path.exists(self.string_rep_dir):
            os.remove(self.string_rep_dir)
        if not os.path.exists(self.benchmarking_file):
            self.write_variable_to_file(self.benchmarking_file, "Dataset, Loading sentences, Generating meuDB, "
                                                                "Loading meuDB, Generating gsmDB, Generating "
                                                                "rewritten graphs, Generating intermediate "
                                                                "representation\n")
        else:
            # If last line is not finished, add new line to ensure next benchmark is written to file correctly
            with open(self.benchmarking_file, 'r') as file:
                if file.readlines()[-1].rstrip('\n').endswith(', '):
                    self.write_variable_to_file(self.benchmarking_file, "\n")

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
            dataset_folder = os.path.join(self.web_dir, "dataset", "data")  # f"{self.web_dir}/dataset/data"
            if os.path.exists(dataset_folder):
                shutil.rmtree(dataset_folder)
            shutil.copytree(self.catabolites_viz, dataset_folder)

        return L

    def write_variable_to_file(self, dir, text):
        try:
            with open(dir, 'a') as file:
                file.write(str(text))
        except Exception as e:
            print(f"An error occurred: {e}")

    def _internal_graph(self, meu_dbs, gsm_list):
        internal_representations = []
        for graph, meu_db in zip(gsm_list, meu_dbs):
            from LaSSI.structures.provenance.GraphProvenance import GraphProvenance
            g = GraphProvenance(graph, meu_db, self.transformation == SentenceRepresentation.SimpleGraph)
            self.logger(f"{meu_db.first_sentence}")
            self.write_variable_to_file(self.string_rep_dir, meu_db.first_sentence)
            internal_graph = g.internal_graph()
            final_form = internal_graph
            if self.transformation == SentenceRepresentation.Logical:
                final_form = g.sentence()
                self.write_variable_to_file(self.string_rep_dir, f" ⇒ {final_form.to_string()}\n")
            internal_representations.append(final_form)
        return internal_representations

    def _logical_rewriting(self, intermediate_representations):
        # logical_representations = []
        #
        # Loop over each Sentence
        # for intermediate_representation in intermediate_representations:
        #     for sentence in intermediate_representation.sentences:
        #     logical_representations.append(rewrite_kernels(intermediate_representation))
        return [rewrite_kernels(x) for x in intermediate_representations]

    def graph_with_logic_similarity(self, x: Graph, y: Graph) -> float:
        if self.sc is None:
            self.sc = SimilarityScore(self.legacy_conf)
        dist = self.sc.graph_distance(x, y) * 1.0
        return 1.0 - dist  # / (1 + dist)

    def fulltext_similarity(self, x: str, y: str) -> float:
        if self.sc is None:
            self.sc = SimilarityScore(self.legacy_conf)
        return self.sc.string_similarity(x, y)

    def _calculate_matrix(self, obj_list):
        if self.transformation == SentenceRepresentation.FullText:
            f = self.fulltext_similarity
        if self.transformation == SentenceRepresentation.Logical:
            from LaSSI.Parmenides.TBox.CrossMatch import DoExpand  # LogicalGraph
            doexp = DoExpand()
            f = SentenceExpansion(obj_list, doexp, self.catabolites_of_dataset)
        elif (self.transformation == SentenceRepresentation.LogicalGraph or
              self.transformation == SentenceRepresentation.SimpleGraph):
            f = self.graph_with_logic_similarity

        matrices = []
        for x in obj_list:
            ls = []
            for y in obj_list:
                ls.append(f(x, y))
            matrices.append(ls)
        # matrices = np.array(matrices)

        return matrices

    def post_hoc_explain(self, lists):
        from LaSSI.files.FileDumpUtilities import target_file_dump
        self.logger("computing similarities")
        confusion_matrices = target_file_dump(self.confusion_matrices + self.transformation.name + ".json",
                                              json.load,
                                              lambda: CalculateMatrix(self, lists),
                                              json_dumps,
                                              self.force)

        if self.clusters_file is not None:
            from LaSSI.similarities.ClusteringTest import test_with_maximal_matching
            clusters = []
            with open(self.clusters_file, "r") as f:
                clusters = json.load(f)
            test_with_maximal_matching(confusion_matrices, clusters,
                                       os.path.join(self.catabolites_of_dataset, self.transformation.name))

    def sentence_transform(self, sentences):
        if self.transformation == SentenceRepresentation.FullText:
            return sentences

        from LaSSI.files.FileDumpUtilities import target_file_dump
        n = len(sentences)
        self.logger("generating meuDB")
        start_time = time.time()
        meu_db, meu_execution_time = target_file_dump(
            self.meuDB,
            lambda x: [MeuDB.from_dict(k) for k in json.load(x)],
            lambda: ExplainTextWithNER(self, sentences),
            json_dumps, self.force, self.should_benchmark
        )
        self.logger(f"Generating meuDB time: {meu_execution_time} seconds")

        self.logger("generating gsmDB")
        gsm_db, gsm_execution_time = target_file_dump(
            self.gsmDB,
            lambda x: x.read(),
            lambda: GetGSMString(self, sentences),
            lambda x: x,
            self.force, self.should_benchmark
        )
        self.logger(f"Generating gsmDB time: {gsm_execution_time} seconds")

        self.logger("generating rewritten graphs")
        rewritten_graphs, rewritten_execution_time = target_file_dump(
            self.datagramdb_output,
            json.load,
            lambda: ApplyGraphGrammars(self, n),
            json_dumps, self.force, self.should_benchmark
        )
        print(f"Generating rewritten graphs time: {rewritten_execution_time} seconds")

        self.logger("generating intermediate representation (before final logical form in eFOL)")
        is_binary = False
        intermediate_representations, intermediate_execution_time = target_file_dump(
            self.internals,
            obj_unmarshall if is_binary else lambda x: [InternalRepresentation.from_dict(k) for k in json.load(x)],
            lambda: SemanticGraphRewriting(self, meu_db, rewritten_graphs),
            obj_pickle if is_binary else json_dumps, not is_binary, self.should_benchmark, is_binary
        )
        # rewrite_kernels(intermediate_representations[0].sentences)
        print(f"Generating intermediate representation time: {intermediate_execution_time} seconds")
        self.write_variable_to_file(self.benchmarking_file,
                                    f"{self.get_execution_time_string(meu_execution_time)}, {gsm_execution_time[0]}, {rewritten_execution_time[0]}, {intermediate_execution_time[0]}\n")

        if self.transformation == SentenceRepresentation.Logical:  # LogicalGraph
            self.logger("[TODO]")
            # intermediate_representations = target_file_dump(self.logical_rewriting,
            #                                           json.load,
            #                                           lambda: LogicalRewriting(self, intermediate_representations),
            #                                           json_dumps,
            #                                           self.force)

    def get_execution_time_string(self, execution_time):
        if execution_time[1] == 'w':
            return f"{execution_time[0]}, 0"
        elif execution_time[1] == 'r':
            return f"0, {execution_time[0]}"

    def run(self):
        from LaSSI.phases.SentenceLoader import SentenceLoader

        start_time = time.time()
        sentences = SentenceLoader(self.sentences)
        end_time = time.time()
        loading_sentences_execution_time = end_time - start_time
        self.logger(f"Loading sentences time: {loading_sentences_execution_time} seconds")
        self.write_variable_to_file(self.benchmarking_file,
                                    f"{self.dataset_name.split('/')[-1].split('.yaml')[0]}, {loading_sentences_execution_time}, ")

        result = self.sentence_transform(sentences)
        # self.post_hoc_explain(result)

    def close(self):
        if isinstance(self.sentences, io.IOBase):
            self.sentences.close()
            self.logger("~~DONE~~")
