__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import dataclasses
import urllib
import rdflib
from rdflib.graph import Graph, ConjunctiveGraph
from rdflib import Graph, URIRef, BNode, Literal, XSD
from rdflib import Namespace
from rdflib.namespace import OWL, RDF, RDFS, FOAF

from LaSSI.Parmenides import Prepositions, SentenceStructure
from LaSSI.structures.extended_fol.Formulae import FUnaryPredicate


def literal(s: str):
    return Literal(s, datatype=XSD.string)


def boolean(s: bool):
    return Literal(s, datatype=XSD.boolean)

def integer(s: bool):
    return Literal(s, datatype=XSD.integer)

def double(s: float):
    return Literal(s, datatype=XSD.double)

def onta(ns, s: str):
    return URIRef(ns[urllib.parse.quote_plus(s)])


class ParmenidesBuild():
    parmenides_ns = Namespace("https://logds.github.io/parmenides#")

    def create_property(self, name, comment=None):
        if name not in self.relationships:
            d_object = URIRef(ParmenidesBuild.parmenides_ns[name])
            self.g.add((d_object, RDF.type, OWL.ObjectProperty))
            self.relationships[name] = d_object
            if comment is not None:
                self.g.add((d_object, RDFS.comment, Literal(comment)))
        return self.relationships[name]

    def create_relationship(self, name, comment=None):
        if name not in self.relationships:
            self.relationships[name] = URIRef(ParmenidesBuild.parmenides_ns[name])
            if comment is not None:
                self.g.add((self.relationships[name], RDFS.comment, Literal(comment)))
        return self.relationships[name]

    def __init__(self):
        self.names = dict()
        self.relationships = dict()
        self.g = Graph()
        self.classes = dict()
        self.g.bind("parmenides", ParmenidesBuild.parmenides_ns)
        self.g.bind("rdfs", RDF)
        self.create_property("hasAdjective")
        self.create_property("subject")
        self.create_property("d_object")
        self.create_property("composite_form_with")
        self.create_property("attachTo")
        self.create_property("argument")
        self.create_property("logicalConstructProperty")
        self.create_property("logicalConstructName")
        self.create_relationship("hasProperty")
        self.create_relationship("formOf")
        self.create_relationship("entryPoint")
        self.create_relationship("partOf")
        self.create_relationship("isA")
        self.create_relationship("relatedTo")
        self.create_relationship("capableOf")
        self.create_relationship("adjectivalForm")
        self.create_relationship("adverbialForm")
        self.create_relationship("eqTo")
        self.create_relationship("neqTo")

    def create_concept(self, full_name, type,
                       hasAdjective=None,
                       entryPoint=None,
                       subject=None,
                       d_object=None,
                       entity_name=None,
                       composite_with=None,comment=None,
                       **kwargs):
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
        for k, val in kwargs.items():
            if k not in self.relationships:
                d_object = URIRef(ParmenidesBuild.parmenides_ns[k])
                self.g.add((d_object, RDF.type, OWL.ObjectProperty))
            result = literal(str(val))
            if isinstance(val, bool):
                result = boolean(val)
            elif isinstance(val, str):
                result = literal(val)
            elif isinstance(val, float):
                result = double(val)
            self.g.add((ref, self.relationships[k], result))
        if comment is not None:
            self.g.add((ref, RDFS.comment, Literal(comment)))
        return ref

    def create_relationship_instance(self, src: str, rel: str, dst: str, refl=False):
        assert src in self.names
        assert dst in self.names
        rel = self.create_relationship(rel)
        self.g.add((self.names[src], rel, self.names[dst]))
        if refl:
            self.g.add((self.names[dst], rel, self.names[src]))

    def create_entity(self, name: str, clazzL=None, label=None, comment=None,
                       **kwargs):
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
        self.extract_properties(self.names[name], kwargs)
        if comment is not None:
            self.g.add((self.names[name], RDFS.comment, Literal(comment)))
        return self.names[name]

    def extract_properties(self, obj_src, kwargs):
        for k, val in kwargs.items():
            if val is not None:
                if k not in self.relationships:
                    rel = URIRef(ParmenidesBuild.parmenides_ns[k])
                    self.g.add((rel, RDF.type, OWL.ObjectProperty))
                    self.relationships[k] = rel
                result = literal(str(val))
                if isinstance(val, dict):
                    src_bnode = BNode()
                    self.g.add((obj_src, self.relationships[k], src_bnode))
                    self.extract_properties(src_bnode, val)
                elif isinstance(val, list) or isinstance(val, tuple):
                    for x in val:
                        if isinstance(x, bool):
                            result = boolean(x)
                        elif isinstance(x, str):
                            result = literal(x)
                        elif isinstance(x, int):
                            result = integer(x)
                        elif isinstance(x, float):
                            result = double(x)
                        self.g.add((obj_src, self.relationships[k], result))
                elif isinstance(val, bool):
                    result = boolean(val)
                    self.g.add((obj_src, self.relationships[k], result))
                elif isinstance(val, str):
                    result = literal(val)
                    self.g.add((obj_src, self.relationships[k], result))
                elif isinstance(val, int):
                    result = integer(val)
                    self.g.add((obj_src, self.relationships[k], result))
                elif isinstance(val, float):
                    result = double(val)
                    self.g.add((obj_src, self.relationships[k], result))

    def create_class(self, name, subclazzOf=None, comment=None):
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
        if comment is not None:
            self.g.add((self.classes[name], RDFS.comment, Literal(comment)))
        return self.classes[name]

    def serialize(self, filename):
        self.g.serialize(destination=filename)


def make_ontology_from_raw():
    p = ParmenidesBuild()
    _T = p.create_class("Dimensions")
    LOC_T = p.create_class("LOC", "Dimensions")
    GPE_T = p.create_class("GPE", ["Dimensions", "LOC"])
    gp_T = p.create_class("GraphParse")
    reject_T = p.create_class("Rejectable", "GraphParse", comment="Whether the edge shall be rejected in the internal rewriting pipeline")
    meta_T = p.create_class("MetaGrammaticalFunction")
    dep_T = p.create_class("dependency", "MetaGrammaticalFunction")
    log_f_T = p.create_class("LogicalFunction", "MetaGrammaticalFunction", comment="The sentence constructs at the logical level, similarly to English' Adverbial Phrases and Indirect Objects (https://it.wikipedia.org/wiki/Analisi_logica_della_proposizione vs. https://en.wikipedia.org/wiki/Adverbial_phrase)")
    log_f_T = p.create_class("LogicalRewritingRule", "MetaGrammaticalFunction", comment="Defines how to capture the elements within the sentence structure and rewriting them in the most appropriate way as properties of the kernel/singleton they refer to")
    gr_obj_T = p.create_class("GrammaticalFunction", "MetaGrammaticalFunction")
    verb_T = p.create_class("Measure", "GrammaticalFunction")  # TODO: Is this a grammatical function
    verb_T = p.create_class("Concept", "GrammaticalFunction")  # TODO: Is this a grammatical function
    verb_T = p.create_class("Verb", "GrammaticalFunction")
    verb_T = p.create_class("Preposition", "GrammaticalFunction")
    noun_T = p.create_class("Noun", "GrammaticalFunction")
    adj_T = p.create_class("Adjective", "GrammaticalFunction")
    adj_T = p.create_class("Adverb", "GrammaticalFunction")
    adj_T = p.create_class("CompoundForm", "GrammaticalFunction")
    tverb_T = p.create_class("TransitiveVerb", "Verb")
    iverb_T = p.create_class("IntransitiveVerb", "Verb")
    causverb_T = p.create_class("CausativeVerb", "Verb")
    moveverb_T = p.create_class("MovementVerb", "Verb")
    meansverb_T = p.create_class("MeansVerb", "Verb")
    stateverb_T = p.create_class("StateVerb", "Verb")
    matverb_T = p.create_class("MaterialisationVerb", "Verb")
    semimodalverb_T = p.create_class("SemiModalVerb", "Verb")
    proto_Prop = p.create_class("PrototypicalPreposition", "Preposition")
    dep_Prop = p.create_class("DependantPreposition", "Preposition")
    idio_Prop = p.create_class("IdiomaticPreposition", "Preposition")
    complex_Prop = p.create_class("ComplexPreposition", "Preposition")
    pronoun = p.create_class("Pronoun")
    pronoun_per = p.create_class("PersonalPronoun", "Pronoun")
    pronoun_dem = p.create_class("DemonstrativePronoun", "Pronoun")
    pronoun_rel = p.create_class("RelativePronoun", "Pronoun")
    pronoun_indef = p.create_class("IndefinitePronoun", "Pronoun")
    pronoun_interro = p.create_class("InterrogativePronoun", "Pronoun")
    unit_of_measure = p.create_class("UnitOfMeasure", "Measure")
    abstract_concept = p.create_class("AbstractEntity", "Concept")

    p.create_concept("television", ["Noun"])
    p.create_concept("home", ["Noun"])
    p.create_concept("house", ["Noun"])
    p.create_concept("home entertainment", ["Noun"])
    p.create_concept("entertainment", ["Noun"])
    p.create_concept("pasttime", ["Noun"])
    p.create_relationship_instance("television", "isA", "home entertainment")
    p.create_relationship_instance("home entertainment", "isA", "entertainment")
    p.create_relationship_instance("home entertainment", "isA", "entertainment")
    p.create_relationship_instance("home entertainment", "isA", "pasttime")
    p.create_relationship_instance("home", "eqTo", "house", refl=True)
    p.create_relationship_instance("home entertainment", "locatedIn", "home", refl=True)
    p.serialize("franco_parmenides.ttl")

def parmenides_db_write():
    from LaSSI.Parmenides.Parmenides import ParmenidesSingleton
    ParmenidesSingleton.instance()
    ParmenidesSingleton.init("/home/giacomo/projects/LaSSI/cache", "giacomo", "omocaig",
                             "localhost", 5432, False, "franco_parmenides.ttl")
    from FunctionalMatch.language.LanguageMainPoint import parse_query
    queries = parse_query("/home/giacomo/projects/LaSSI/query_franco.txt")
    from LaSSI.structures.extended_fol.TBoxReasoning import KnowledgeExpansion
    ke = KnowledgeExpansion("/home/giacomo/projects/LaSSI/_kexp.pickle")

    from LaSSI.structures.extended_fol.Formulae import FVariable
    var = FVariable("?1", "existential", None, None, 1)
    cable = FVariable("cable", "ENTITY", None, None, 1)
    tv = FVariable("television", "ENTITY", None, None, 1)
    it = FVariable("it", "ENTITY", None, None, 1)
    he = FVariable("equipment", "ENTITY", "home entertainment", None, 1)
    from LaSSI.structures.extended_fol.Formulae import FBinaryPredicate
    q1 = FBinaryPredicate("isA", var, he, -1, frozenset())
    q2 = FBinaryPredicate("require", var, cable, -1, frozenset())
    q3 = FBinaryPredicate("isA", var, he, -1, frozenset())
    # what_home_entertainment = print(make_and([q1, q2]))

    baggage = FVariable("baggage", "ENTITY", None, None, 1)
    where = FVariable("?2", "ENTITY", None, None, 1)
    she = FVariable("she", "ENTITY", None, None, 1)
    headTo = FUnaryPredicate("head", she, 1, frozenset((("TOGETHERNESS", (baggage,)), ("SPACE", (where,)))))
    print(headTo)

    ke.pruned_expansion(headTo, queries, "rule")

    # from FunctionalMatch.example.LaSSI.eFOLsemantics.TBoxReasoning import knowledge_expansion
    for k,v in ke.fullGraph().items():
        for idx, element in v:
            print(element)

    ## STOP PARMENIDES
    ParmenidesSingleton.stop()


if __name__ == "__main__":
    make_ontology_from_raw()
    parmenides_db_write()
