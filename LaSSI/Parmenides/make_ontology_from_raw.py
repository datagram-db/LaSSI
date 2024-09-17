__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"
import urllib
import rdflib
from rdflib.graph import Graph, ConjunctiveGraph
from rdflib import Graph, URIRef, BNode, Literal, XSD
from rdflib import Namespace
from rdflib.namespace import OWL, RDF, RDFS, FOAF

def literal(s:str):
    return Literal(s, datatype=XSD.string)

def boolean(s:bool):
    return Literal(s, datatype=XSD.boolean)

def onta(ns, s:str):
    return URIRef(ns[urllib.parse.quote_plus(s)])

class ParmenidesBuild():
    parmenides_ns = Namespace("https://lo-ds.github.io/parmenides#")

    def __init__(self):
        self.names = dict()
        self.g = Graph()
        self.classes = dict()
        self.g.bind("parmenides", ParmenidesBuild.parmenides_ns)
        self.g.bind("rdfs", RDF)
        hasAdjective = URIRef(ParmenidesBuild.parmenides_ns["hasAdjective"])
        self.g.add((hasAdjective, RDF.type, OWL.ObjectProperty))
        subject = URIRef(ParmenidesBuild.parmenides_ns["subject"])
        self.g.add((subject, RDF.type, OWL.ObjectProperty))
        d_object = URIRef(ParmenidesBuild.parmenides_ns["d_object"])
        self.g.add((d_object, RDF.type, OWL.ObjectProperty))
        composite_form_with = URIRef(ParmenidesBuild.parmenides_ns["composite_form_with"])
        self.g.add((composite_form_with, RDF.type, OWL.ObjectProperty))
        self.relationships = dict()
        self.relationships["composite_form_with"] = composite_form_with
        self.relationships["hasAdjective"] = hasAdjective
        self.relationships["subject"] = subject
        self.relationships["d_object"] = d_object
        self.relationships["hasProperty"] = URIRef(ParmenidesBuild.parmenides_ns["hasProperty"])
        self.relationships["formOf"] = URIRef(ParmenidesBuild.parmenides_ns["formOf"])
        self.relationships["entryPoint"] = URIRef(ParmenidesBuild.parmenides_ns["entryPoint"])
        self.relationships["partOf"] = URIRef(ParmenidesBuild.parmenides_ns["partOf"])
        self.relationships["isA"] = URIRef(ParmenidesBuild.parmenides_ns["isA"])
        self.relationships["relatedTo"] = URIRef(ParmenidesBuild.parmenides_ns["relatedTo"])
        self.relationships["capableOf"] = URIRef(ParmenidesBuild.parmenides_ns["capableOf"])
        self.relationships["adjectivalForm"] = URIRef(ParmenidesBuild.parmenides_ns["adjectivalForm"])
        self.relationships["adverbialForm"] = URIRef(ParmenidesBuild.parmenides_ns["adverbialForm"])
        self.relationships["eqTo"] = URIRef(ParmenidesBuild.parmenides_ns["eqTo"])
        self.relationships["neqTo"] = URIRef(ParmenidesBuild.parmenides_ns["neqTo"])

    def create_concept(self, full_name, type,
                             hasAdjective = None,
                             entryPoint = None,
                             subject = None,
                             d_object = None,
                       entity_name=None,
                       composite_with=None):
        if entity_name == None:
            entity_name = full_name
        ref = self.create_entity(full_name, type, label=entity_name)
        if entryPoint == None:
            entryPoint = ref
        else:
            assert entryPoint in self.names
            entryPoint = self.names[entryPoint]
        self.g.add((ref, self.relationships["entryPoint"], entryPoint))
        from collections.abc import Iterable
        if (hasAdjective != None) and (isinstance(hasAdjective, Iterable)):
            assert hasAdjective in self.names
            self.g.add((ref, self.relationships["hasAdjective"], self.names[hasAdjective]))
        if (d_object is not None):
            assert subject is not None
        if composite_with is not None:
            assert isinstance(composite_with, list)
            for composite in composite_with:
                assert composite in self.names
                self.g.add((ref, self.relationships["composite_form_with"], self.names[composite]))
        if subject is not None:
            assert subject in self.names
            self.g.add((ref, self.relationships["subject"], self.names[subject]))
            if d_object is not None:
                self.g.add((ref, self.relationships["d_object"], self.names[d_object]))
        return ref

    def create_relationship(self, src:str, rel:str, dst:str, refl=False):
        assert src in self.names
        assert rel in self.relationships
        assert dst in self.names
        self.g.add((self.names[src], self.relationships[rel], self.names[dst]))
        if refl:
            self.g.add((self.names[dst], self.relationships[rel], self.names[src]))

    def create_entity(self, name:str, clazzL=None, label=None):
        if label is None:
            label = name
        if name not in self.names:
            self.names[name] = onta(ParmenidesBuild.parmenides_ns, name)
            if (clazzL is not None):
                if isinstance(clazzL, list):
                    for clazz in clazzL:
                        assert clazz in self.classes
                        clazz = self.classes[clazz]
                        self.g.add((self.names[name], RDF.type, clazz))
                elif isinstance(clazzL, str):
                    clazz = self.classes[clazzL]
                    self.g.add((self.names[name], RDF.type, clazz))
                self.g.add((self.names[name], RDFS.label, literal(label)))
        return self.names[name]

    def create_class(self, name, subclazzOf=None):
        if name not in self.classes:
            clazz = onta(ParmenidesBuild.parmenides_ns, name)
            self.g.add((clazz, RDF.type, OWL.Class))
            if (subclazzOf is not None):
                if isinstance(subclazzOf, str):
                    subclazzOf = self.create_class(subclazzOf)
                    self.g.add((clazz, RDFS.subClassOf, subclazzOf))
                elif isinstance(subclazzOf, list):
                    for x in subclazzOf:
                        x = self.create_class(x)
                        self.g.add((clazz, RDFS.subClassOf, x))
            self.classes[name] = clazz
        return self.classes[name]


    def serialize(self, filename):
        self.g.serialize(destination=filename)
def make_ontology_from_raw():
    p = ParmenidesBuild()
    _T = p.create_class("Dimensions")
    LOC_T = p.create_class("LOC", "Dimensions")
    GPE_T = p.create_class("GPE",  ["Dimensions", "LOC"])
    gp_T = p.create_class("GraphParse")
    reject_T = p.create_class("Rejectable", "GraphParse")
    meta_T = p.create_class("MetaGrammaticalFunction")
    dep_T = p.create_class("dependency", "MetaGrammaticalFunction")
    gr_obj_T = p.create_class("GrammaticalFunction", "MetaGrammaticalFunction")
    verb_T = p.create_class("Verb", "GrammaticalFunction")
    noun_T = p.create_class("Noun", "GrammaticalFunction")
    adj_T = p.create_class("Adjective", "GrammaticalFunction")
    adj_T = p.create_class("Adverb", "GrammaticalFunction")
    adj_T = p.create_class("CompoundForm", "GrammaticalFunction")
    tverb_T = p.create_class("TransitiveVerb", "Verb")
    iverb_T = p.create_class("IntransitiveVerb", "Verb")
    to_reject = set()
    with open("../../raw_data/rejected_edge_types.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            to_reject.add(line)
    with open("../../raw_data/non_verb_types.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["dependency"]
            if line in to_reject:
                classes.append("Rejectable")
            p.create_entity(line, classes)
    with open("../../raw_data/transitive_verbs.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["TransitiveVerb"]
            if line in to_reject:
                classes.append("Rejectable")
            p.create_entity(line, classes)
    p.create_concept("center", "Adjective")
    p.create_concept("centre", "Adjective")
    p.create_relationship("center", "eqTo", "centre", True)
    p.create_concept("busy", "Adjective")
    p.create_concept("crowded", "Adjective")
    p.create_concept("fast", "Adjective")
    p.create_concept("busy", "Adjective")
    p.create_concept("slow#adj", "Adjective", entity_name="slow")
    p.create_concept("slow#v", "Verb", entity_name="slow")
    p.create_relationship("slow#v", "adjectivalForm", "slow#adj")
    p.create_relationship("slow#adj", "neqTo", "fast", True)
    p.create_concept("crowd#n", "Noun", entity_name="crowd")
    p.create_concept("crowd#v", "Verb", entity_name="crowd")
    p.create_relationship("busy", "relatedTo", "crowd#n", True)
    p.create_relationship("crowd#n", "relatedTo", "crowded", True)
    p.create_relationship("crowd#v", "relatedTo", "crowded", True)
    p.create_concept("city", "LOC")
    p.create_concept("Newcastle", "GPE")
    p.create_relationship("Newcastle", "isA", "city")
    p.create_concept("traffic#v", "Verb", entity_name="traffic")
    p.create_concept("traffic#n", "Noun", entity_name="traffic")
    p.create_concept("flow in", "Verb")
    p.create_concept("flow#v", "Verb", entity_name="flow")
    p.create_relationship("flow#v", "adjectivalForm", "fast")
    p.create_concept("flow#n", "Noun", entity_name="flow")
    p.create_concept("congestion", ["Noun"])
    p.create_concept("jam", ["Noun"])
    p.create_concept("busy city", ["Noun"], hasAdjective="busy", entryPoint="city")
    p.create_concept("traffic jam", ["Noun"], composite_with=["traffic#n", "jam"], entryPoint="traffic#n")
    p.create_concept("traffic congestion", ["Noun"], composite_with=["traffic#n", "congestion"], entryPoint="traffic#n")
    p.create_relationship("traffic jam", "eqTo", "traffic congestion", True)
    p.create_relationship("traffic jam", "eqTo", "traffic congestion", True)
    p.create_concept("flow fast", "CompoundForm", hasAdjective="fast", entryPoint="flow#v")
    p.create_relationship("flow in", "eqTo", "flow#v", True)
    p.create_concept("traffic jam can slow traffic", "CompoundForm", entryPoint="slow", subject="traffic jam", d_object="traffic#n")
    p.create_concept("city centers", "LOC", hasAdjective="center", entryPoint="city")
    p.create_concept("city centres", "LOC", hasAdjective="centre", entryPoint="city")
    p.create_concept("city center", "LOC", hasAdjective="center", entryPoint="city")
    p.create_concept("city centre", "LOC", hasAdjective="centre", entryPoint="city")
    p.create_relationship("city", "hasProperty", "busy", True)
    p.create_relationship("city center", "partOf", "city")
    p.create_relationship("city center", "eqTo", "city centre", True)
    p.create_relationship("city centre", "partOf", "city")
    p.create_relationship("city centers", "partOf", "city")
    p.create_relationship("city centres", "partOf", "city")
    p.create_relationship("city center", "eqTo", "city centers", True)
    p.create_relationship("city center", "eqTo", "city centres", True)
    p.create_relationship("city centre", "eqTo", "city centers", True)
    p.create_relationship("city centre", "eqTo", "city centres", True)
    p.create_relationship("city centers", "eqTo", "city centres", True)
    p.create_relationship("busy", "relatedTo", "crowd#n", True)
    p.create_relationship("congestion", "relatedTo", "traffic congestion", True)
    p.create_relationship("crowd#n", "relatedTo", "congestion", True)
    p.create_relationship("busy city", "relatedTo", "crowd#n", True)
    p.create_relationship("traffic jam", "capableOf", "traffic jam can slow traffic")
    p.serialize("turtle.ttl")

if __name__ == "__main__":
    make_ontology_from_raw()