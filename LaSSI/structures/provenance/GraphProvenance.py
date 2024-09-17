from LaSSI.external_services.Services import Services
from LaSSI.ner.AssignTypeToSingleton import AssignTypeToSingleton
from LaSSI.structures.internal_graph.EntityRelationship import Grouping
from LaSSI.structures.internal_graph.Graph import Graph


class GraphProvenance:
    def __init__(self, raw_json_graph, meuDB, simplistic):
        # self.nodes = dict()
        self._internal_graph = None
        self._sentence = None
        self._nodes = None
        self._edges = None
        self.raw_json_graph = raw_json_graph
        self.simplistic = simplistic
        self.stanza_row = meuDB
        self.atts_global = AssignTypeToSingleton()
        self.services = Services.getInstance()
        self.parmenides = self.services.getParmenides()

    def internal_graph(self):
        # from LaSSI.structures.internal_graph.from_raw_json_graph import group_nodes
        self.atts_global.group_nodes(self.raw_json_graph, self.simplistic, self.parmenides, self.stanza_row)
        self.atts_global.checkForNegation(self.raw_json_graph)
        rejected_edges = self.parmenides.getRejectedVerbs()
        non_verbs = self.parmenides.getNonVerbs()
        self._internal_graph = self.atts_global.constructIntermediateGraph(self.raw_json_graph, rejected_edges, non_verbs)
        return self._internal_graph

    def sentence(self):
        from LaSSI.structures.kernels.Sentence import create_sentence_obj
        transitive_verbs = self.parmenides.getTransitiveVerbs()
        self._sentence = self.atts_global.constructSentence(transitive_verbs)
        return self._sentence