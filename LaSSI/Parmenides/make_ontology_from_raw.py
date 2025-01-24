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
    to_reject = set()
    with open("../../raw_data/rejected_edge_types.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            to_reject.add(line.lower())
    with open("../../raw_data/non_verb_types.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["dependency"]
            if line in to_reject:
                classes.append("Rejectable")
            p.create_entity(line, classes)
    with open("../../raw_data/verbs/causative_verbs.txt", "r") as dep:
        # , \
        #     open("../../raw_data/transitive_verbs.txt", "r") as transitive_verbs_file, \
        #         open("../../raw_data/materialisation_verbs.txt", "r") as materialisation_verbs_file):
        # transitive_verbs = set(line.strip() for line in transitive_verbs_file)
        # materialisation_verbs = set(line.strip() for line in materialisation_verbs_file)
        for line in dep:
            line = line.strip()
            classes = ["CausativeVerb"]
            if line in to_reject:
                classes.append("Rejectable")  # TODO: Check
            # if line in transitive_verbs:
            #     classes.append("TransitiveVerb")
            # if line in materialisation_verbs:
            #     classes.append("MaterialisationVerb")
            p.create_entity(line, classes)
    with open("../../raw_data/verbs/transitive_verbs.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["TransitiveVerb"]
            if line in to_reject:
                classes.append("Rejectable")
            p.create_entity(line, classes)
    with open("../../raw_data/units_of_measure.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["UnitOfMeasure"]
            if line in to_reject:
                classes.append("Rejectable")  # TODO: Check
            p.create_entity(line, classes)
    with open("../../raw_data/abstract_entity_concepts.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["AbstractEntity"]
            if line in to_reject:
                classes.append("Rejectable")  # TODO: Check
            p.create_entity(line, classes)
    with open("../../raw_data/verbs/stative_verbs.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["CausativeVerb"]  # TODO: Should these be causative instead of "stative"?
            if line in to_reject:
                classes.append("Rejectable")  # TODO: Check
            p.create_entity(line, classes)
    with open("../../raw_data/verbs/state_verbs.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["StateVerb"]
            if line in to_reject:
                classes.append("Rejectable")  # TODO: Check
            p.create_entity(line, classes)
    with open("../../raw_data/verbs/movement_verbs.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["MovementVerb"]
            if line in to_reject:
                classes.append("Rejectable")  # TODO: Check
            p.create_entity(line, classes)
    with open("../../raw_data/verbs/materialisation_verbs.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["MaterialisationVerb"]
            if line in to_reject:
                classes.append("Rejectable")  # TODO: Check
            p.create_entity(line, classes)
    for preposition in Prepositions.load_prepositions("../../raw_data/prepositions.json"):
        classes = preposition.generate_classes(preposition.name.lower() in to_reject)
        p.create_entity(preposition.name, classes, **preposition.as_properties())
    log_defs, log_rewr_rules = SentenceStructure.load_logical_analysis("../../raw_data/logical_analysis.json")
    for name, v in log_defs.items():
        for x in v.specs:
            d = dataclasses.asdict(x)
            if "property" in d and d["property"] is None:
                d.pop("property")
            else:
                d["logicalConstructProperty"] = d.pop("property")
            d["logicalConstructName"] = name
            entity_name = f"log/{name}/{d['logicalConstructProperty']}" if "logicalConstructProperty" in d else f"log/{name}"
            p.create_entity(entity_name, "LogicalFunction", entity_name, **d)
    ruleid = 1
    for rule in log_rewr_rules:
        for result in rule.classification:
            dres = dataclasses.asdict(result)
            if "property" in dres and dres["property"] is None:
                dres.pop("property")
            else:
                dres["logicalConstructProperty"] = dres.pop("property")
            dres["logicalConstructName"] = dres.pop("type")
            dres["rule_order"] = ruleid
            dres.update(rule.premise)
            p.create_entity(f"logrule/{ruleid}","LogicalRewritingRule", **dres)
            ruleid += 1

    with open("../../raw_data/pronouns/personal_pronouns.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["Pronoun", "PersonalPronoun"]
            if line in to_reject:
                classes.append("Rejectable")
            p.create_entity(line, classes)
    with open("../../raw_data/pronouns/demonstrative_pronouns.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["Pronoun", "DemonstrativePronoun"]
            if line in to_reject:
                classes.append("Rejectable")
            p.create_entity(line, classes)
    with open("../../raw_data/pronouns/relative_pronouns.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["Pronoun", "RelativePronoun"]
            if line in to_reject:
                classes.append("Rejectable")
            p.create_entity(line, classes)
    with open("../../raw_data/pronouns/indefinite_pronouns.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["Pronoun", "IndefinitePronoun"]
            if line in to_reject:
                classes.append("Rejectable")
            p.create_entity(line, classes)
    with open("../../raw_data/pronouns/interrogative_pronouns.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["Pronoun", "InterrogativePronoun"]
            if line in to_reject:
                classes.append("Rejectable")
            p.create_entity(line, classes)
    with open("../../raw_data/verbs/semi_modal_verbs.txt", "r") as dep:
        for line in dep:
            line = line.strip()
            classes = ["Verb", "SemiModalVerb"]
            if line in to_reject:
                classes.append("Rejectable")
            p.create_entity(line, classes)
    p.create_concept("world", ["Noun"])
    p.create_concept("picture", ["Noun"])
    p.create_concept("hectic", "Adjective")
    p.create_concept("beautiful", "Adjective")
    p.create_concept("fabulous", "Adjective")
    p.create_concept("center", "Adjective")
    p.create_concept("centre", "Adjective")
    p.create_relationship_instance("center", "eqTo", "centre", True)
    p.create_concept("busy", "Adjective")
    p.create_concept("crowded", "Adjective")
    p.create_concept("fast", "Adjective")
    p.create_concept("busy", "Adjective")
    p.create_concept("slow#adj", "Adjective", entity_name="slow")
    p.create_concept("slow#v", "Verb", entity_name="slow")
    p.create_relationship_instance("slow#v", "adjectivalForm", "slow#adj")
    p.create_relationship_instance("slow#adj", "neqTo", "fast", True)
    p.create_concept("crowd#n", "Noun", entity_name="crowd")
    p.create_concept("crowd#v", "Verb", entity_name="crowd")
    p.create_relationship_instance("busy", "relatedTo", "crowd#n", True)
    p.create_relationship_instance("crowd#n", "relatedTo", "crowded", True)
    p.create_relationship_instance("crowd#v", "relatedTo", "crowded", True)
    p.create_concept("city", "LOC")
    p.create_concept("Newcastle", "GPE")
    p.create_relationship_instance("Newcastle", "isA", "city")
    p.create_concept("come back#v", "Verb", entity_name="come back")
    p.create_concept("traffic#v", "Verb", entity_name="traffic")
    p.create_concept("traffic#n", "Noun", entity_name="traffic")
    p.create_concept("flow in", "Verb")
    p.create_concept("flow#v", "Verb", entity_name="flow")
    p.create_relationship_instance("flow#v", "adjectivalForm", "fast")
    p.create_concept("flow#n", "Noun", entity_name="flow")
    p.create_concept("congestion", ["Noun"])
    p.create_concept("jam", ["Noun"])
    p.create_concept("busy city", ["Noun"], hasAdjective="busy", entryPoint="city")
    p.create_concept("traffic jam", ["Noun"], composite_with=["traffic#n", "jam"], entryPoint="traffic#n")
    p.create_concept("traffic congestion", ["Noun"], composite_with=["traffic#n", "congestion"], entryPoint="traffic#n")
    p.create_relationship_instance("traffic jam", "eqTo", "traffic congestion", True)
    p.create_relationship_instance("traffic jam", "eqTo", "traffic congestion", True)
    p.create_concept("flow fast", "CompoundForm", hasAdjective="fast", entryPoint="flow#v")
    p.create_relationship_instance("flow in", "eqTo", "flow#v", True)
    p.create_concept("traffic jam can slow traffic", "CompoundForm", entryPoint="slow", subject="traffic jam", d_object="traffic#n")
    p.create_concept("city centers", "LOC", hasAdjective="center", entryPoint="city")
    p.create_concept("city centres", "LOC", hasAdjective="centre", entryPoint="city")
    p.create_concept("city center", "LOC", hasAdjective="center", entryPoint="city")
    p.create_concept("city centre", "LOC", hasAdjective="centre", entryPoint="city")
    p.create_relationship_instance("city", "hasProperty", "busy", True)
    p.create_relationship_instance("city center", "partOf", "city")
    p.create_relationship_instance("city center", "eqTo", "city centre", True)
    p.create_relationship_instance("city centre", "partOf", "city")
    p.create_relationship_instance("city centers", "partOf", "city")
    p.create_relationship_instance("city centres", "partOf", "city")
    p.create_relationship_instance("city center", "eqTo", "city centers", True)
    p.create_relationship_instance("city center", "eqTo", "city centres", True)
    p.create_relationship_instance("city centre", "eqTo", "city centers", True)
    p.create_relationship_instance("city centre", "eqTo", "city centres", True)
    p.create_relationship_instance("city centers", "eqTo", "city centres", True)
    p.create_relationship_instance("busy", "relatedTo", "crowd#n", True)
    p.create_relationship_instance("congestion", "relatedTo", "traffic congestion", True)
    p.create_relationship_instance("crowd#n", "relatedTo", "congestion", True)
    p.create_relationship_instance("busy city", "relatedTo", "crowd#n", True)
    p.create_relationship_instance("traffic jam", "capableOf", "traffic jam can slow traffic")
    p.create_relationship_instance("hectic", "hasProperty", "traffic#n", True)
    p.create_relationship_instance("hectic", "eqTo", "busy", True)

    ## This is a tumor: single entity match in post-processing
    p.create_concept("embryoma_of_the_kidney#n", "Noun", entity_name="embryoma of the kidney")

    ## These are the examples leading to swapped specification properties via nmod, so, not specification(front,letter) as per nmod, but swapped specification(letter,front)
    p.create_concept("letter#n", "Noun", entity_name="letter")
    p.create_concept("front#n", "Noun", entity_name="front")
    p.create_relationship_instance("letter#n", "hasProperty", "front#n") ## Example of inverse occurrence of the relationship if compared to the nmod (isSymmetricalIfComparedToNMod,hasNModPartOf)
    p.create_concept("street#n", "Noun", entity_name="street")
    p.create_concept("corner#n", "Noun", entity_name="corner")
    p.create_relationship_instance("street#n", "hasProperty", "corner#n")
    p.create_concept("cell_division#n", "Noun", entity_name="cell division")
    p.create_concept("phase#n", "Noun", entity_name="phase")
    p.create_relationship_instance("cell_division#n", "hasProperty", "phase#n")
    p.create_concept("object#n", "Noun", entity_name="object")
    p.create_concept("surface#n", "Noun", entity_name="surface")
    p.create_relationship_instance("object#n", "hasProperty", "surface#n")
    p.create_concept("game#n", "Noun", entity_name="game")
    p.create_concept("chess#n", "Noun", entity_name="chess")
    p.create_relationship_instance("chess#n", "isA", "game#n") #(isSymmetricalIfComparedToNMod,hasNModIsA)

    p.serialize("turtle.ttl")


if __name__ == "__main__":
    make_ontology_from_raw()
