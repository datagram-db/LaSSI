__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import copy
import io
import os.path
import pickle
import re
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from functools import reduce, lru_cache
from string import Template
from typing import List, Optional

import dacite
import pandas
import rdflib
from rdflib.graph import Graph, ConjunctiveGraph
from rdflib import Graph, URIRef, BNode, Literal, XSD
from rdflib import Namespace
from rdflib.namespace import OWL, RDF, RDFS, FOAF


def instantiateWithMap(s: str, m: dict):
    return Template(s.replace("@", "$")).safe_substitute(m)


# def combine_or_cross(x,y):
#     if len(set(x.columns).intersection(y.columns)) == 0:
#

class Parmenides():
    parmenides_ns = Namespace("https://lo-ds.github.io/parmenides#")

    def __init__(self, filename=None):
        if filename is None:
            import pkg_resources
            filename = pkg_resources.resource_filename("LaSSI.resources", "turtle.ttl")
        self.g = rdflib.Graph()
        self.g.parse(filename)
        self.trcl = defaultdict(set)
        self.syn = defaultdict(set)
        self.st = defaultdict(set)
        self.semi_modal_verbs = set(self.get_semi_modal_verbs())
        self.pronouns = set(self.get_pronouns())
        self.prototypical_prepositions = set(self.get_prototypical_prepositions())
        self.transitive_verbs = set(self.get_transitive_verbs())
        self.causative_verbs = set(self.get_causative_verbs())
        self.movement_verbs = set(self.get_movement_verbs())
        self.means_verbs = set(self.get_means_verbs())
        self.state_verbs = set(self.get_state_verbs())
        self.materialisation_verbs = set(self.get_materialisation_verbs())
        self.units_of_measure = set(self.get_units_of_measure())
        self.abstract_entities = set(self.get_abstract_entities())
        self.rejected_edges = set(self.get_rejected_edges())
        self.non_verbs = set(self.get_universal_dependencies())
        self.logical_rewriting_rules = self.get_logical_rewriting_rules()
        self.nouns_with_properties = set(self.get_nouns_with_properties())
        self.nouns_with_a = set(self.get_nouns_with_a())
        self.prepositions = None

    def most_specific_type(self, types):
        types = list(map(lambda x: str(x).lower(), types))
        ### TODO: within .ttl and type inference
        if any(map(lambda x: "verb" in x, types)):
            return "VERB"
        elif "gpe" in types:
            return "GPE"
        elif "loc" in types:
            return "LOC"
        elif "org" in types:
            return "ORG"
        elif "noun" in types:
            return "noun"
        elif "entity" in types:
            return "ENTITY"
        elif "adjective" in types:
            return "JJ"
        else:
            return "None"

    # TODO: MAKE THIS BETTER EVENTUALLY
    def most_general_type(self, types):
        types = list(map(lambda x: str(x).upper(), types))

        tree = {
            "ENTITY": {
                "NOUN": {"ORG": {}, "PERSON": {}, "LOC": {"GPE": {}}},
                "JJ": {}
            }
        }

        if not types or not tree:
            return None

        if len(types) == 1:
            return types[0]

        def find_path(root, target_type, path):
            if root == target_type:
                return path
            if isinstance(root, dict):
                for key, value in root.items():
                    new_path = find_path(value, target_type, path + [key])
                    if new_path:
                        return new_path
            return None

        paths = []
        for type_ in types:
            path = find_path(tree, type_, [])
            if not path:
                return None  # Type not found in the tree
            paths.append(path)

        min_len = min(len(path) for path in paths)
        lca = None
        for i in range(min_len):
            if all(path[i] == paths[0][i] for path in paths):
                lca = paths[0][i]
            else:
                break

        return lca

    def getNounsWithProperties(self):
        return self.nouns_with_properties

    def getNounsWithA(self):
        return self.nouns_with_a

    def getSemiModalVerbs(self):
        return self.semi_modal_verbs

    def getPronouns(self):
        return self.pronouns

    def getPrototypicalPrepositions(self):
        return self.prototypical_prepositions

    def getTransitiveVerbs(self):
        return self.transitive_verbs

    def getRejectedVerbs(self):
        return self.rejected_edges

    def getNonVerbs(self):
        return self.non_verbs

    def getLogicalRewritingRules(self):
        return self.logical_rewriting_rules

    def getCausativeVerbs(self):
        return self.causative_verbs

    def getMovementVerbs(self):
        return self.movement_verbs

    def getUnitsOfMeasure(self):
        return self.units_of_measure

    def getMeansVerbs(self):
        return self.means_verbs

    def getAbstractEntities(self):
        return self.abstract_entities

    def getStateVerbs(self):
        return self.state_verbs

    def getMaterialisationVerbs(self):
        return self.materialisation_verbs

    def extractPureHierarchy(self, t, flip=False):
        ye = list(self.single_edge("^src", t, "^dst"))
        if len(ye) == 0:
            return set()
        elif flip:
            return {(x["dst"], x["src"]) for x in ye}
        else:
            return {(x["src"], x["dst"]) for x in ye}

    def getAllEntitiesBuyImmediateType(self, t):
        ye = list(self.isA("^x", str(t)))
        if len(ye) == 0:
            return set()
        else:
            return {x["x"] for x in ye}

    def getSynonymy(self, k):
        if len(self.syn) == 0:
            if os.path.exists("syn.pickle"):
                with open("syn.pickle", "rb") as f:
                    self.syn = pickle.load(f)
            else:
                from LaSSI.Parmenides.TBox.CrossMatch import transitive_closure
                self.syn = self.extractPureHierarchy("eqTo", True) | self.extractPureHierarchy("eqTo", False)
                self.syn = transitive_closure(self.syn)
                tmp = defaultdict(set)
                for (x, y) in self.syn:
                    tmp[x].add(y)
                    tmp[y].add(x)
                self.syn = tmp
                with open("syn.pickle", "wb") as f:
                    pickle.dump(self.syn, f, protocol=pickle.HIGHEST_PROTOCOL)
        if k in self.syn:
            return self.syn[k]
        else:
            return {k}

    def getTransitiveClosureHier(self, t):
        from LaSSI.Parmenides.TBox.CrossMatch import transitive_closure
        if os.path.exists("hier.pickle"):
            with open("hier.pickle", "rb") as f:
                self.trcl = pickle.load(f)
        if len(self.trcl) == 0:
            s = self.extractPureHierarchy("isA", True) | self.extractPureHierarchy("partOf", False)
            self.trcl = transitive_closure(s)
            with open("hier.pickle", "wb") as f:
                pickle.dump(self.trcl, f, protocol=pickle.HIGHEST_PROTOCOL)
        return self.trcl

    def name_eq(self, src, dst):
        from LaSSI.Parmenides.TBox.ExpandConstituents import CasusHappening
        if (src == dst):
            return CasusHappening.EQUIVALENT
        elif (src is None) or len(src) == 0:
            return CasusHappening.MISSING_1ST_IMPLICATION
        elif (dst is None) or len(dst) == 0:
            return CasusHappening.INDIFFERENT
        else:
            resolveTypeFromOntologyLHS = set(self.getSuperTypes(src))
            resolveTypeFromOntologyRHS = set(self.getSuperTypes(dst))
            isect = resolveTypeFromOntologyLHS.intersection(resolveTypeFromOntologyRHS)
            if len(resolveTypeFromOntologyLHS) == 0:
                return CasusHappening.INDIFFERENT
            elif len(resolveTypeFromOntologyRHS) == 0:
                return CasusHappening.INDIFFERENT
            elif len(isect) == 0:
                return CasusHappening.INDIFFERENT
            else:
                for k in isect:
                    if len(list(self.single_edge(src, "neqTo", dst))) > 0:
                        return CasusHappening.EXCLUSIVES
                    srcS = self.getSynonymy(src)
                    dstS = self.getSynonymy(dst)
                    if len(set(srcS).intersection(set(dstS))) > 0:
                        return CasusHappening.EQUIVALENT
                    for lhs in self.getSynonymy(src):
                        for rhs in self.getSynonymy(dst):
                            if (lhs, rhs) in self.getTransitiveClosureHier(k):
                                return CasusHappening.GENERAL_IMPLICATION
                return CasusHappening.INDIFFERENT

    def _single_unary_query(self, knows_query, f):
        qres = self.g.query(knows_query)
        for row in qres:
            yield f(row)

    @lru_cache(maxsize=128)
    def get_semi_modal_verbs(self):
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a parmenides:SemiModalVerb.
             ?a rdfs:label ?c .
         }"""
        return self._single_unary_query(knows_query, lambda x: str(x.c))

    @lru_cache(maxsize=128)
    def get_transitive_verbs(self):
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a parmenides:TransitiveVerb.
             ?a rdfs:label ?c .
         }"""
        return self._single_unary_query(knows_query, lambda x: str(x.c))

    @lru_cache(maxsize=128)
    def get_prototypical_prepositions(self):
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a parmenides:PrototypicalPreposition.
             ?a rdfs:label ?c .
         }"""
        return self._single_unary_query(knows_query, lambda x: str(x.c))

    @lru_cache(maxsize=128)
    def get_pronouns(self):
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a parmenides:Pronoun.
             ?a rdfs:label ?c .
         }"""
        return self._single_unary_query(knows_query, lambda x: str(x.c))

    @lru_cache(maxsize=128)
    def get_nouns_with_properties(self):
        knows_query = """
         SELECT DISTINCT ?hasProperty ?label
         WHERE {
             ?a a parmenides:Noun.
             ?a rdfs:label ?label .
             ?a parmenides:hasProperty ?hasProperty .
         }"""
        return self._single_unary_query(knows_query, lambda x: x)

    @lru_cache(maxsize=128)
    def get_nouns_with_a(self):
        knows_query = """
         SELECT DISTINCT ?isA ?label
         WHERE {
             ?a a parmenides:Noun.
             ?a rdfs:label ?label .
             ?a parmenides:isA ?isA .
         }"""
        return self._single_unary_query(knows_query, lambda x: x)

    @lru_cache(maxsize=128)
    def get_universal_dependencies(self):
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a parmenides:dependency.
             ?a rdfs:label ?c .
         }"""
        return self._single_unary_query(knows_query, lambda x: str(x.c))

    def getSuperTypes(self, src):
        if src in self.st:
            return self.st[src]
        s = list(self.typeOf(src))
        visited = set()
        while len(s) > 0:
            x = s.pop()
            if x not in visited:
                visited.add(x)
                for y in self.typeOf2(Parmenides.parmenides_ns[str(x)[len(Parmenides.parmenides_ns):]]):
                    s.append(y)
        self.st[src] = visited
        return visited

    def typeOf2(self, src):
        knows_query = """
         SELECT DISTINCT ?dst 
         WHERE {
             ?src rdfs:subClassOf ?dst.
         }"""
        qres = self.g.query(knows_query, initBindings={"src": src})
        s = set()
        for x in qres:
            s.add(str(x.dst))
        return s

    def typeOf(self, src):
        knows_query = """
         SELECT DISTINCT ?dst 
         WHERE {
             ?src a ?dst.
             ?src rdfs:label ?src_label.
         }"""
        qres = self.g.query(knows_query, initBindings={"src_label": Literal(src, datatype=XSD.string)})
        s = set()
        for x in qres:
            s.add(str(x.dst))
        return s

    def getTypedObjects(self):
        typing = """
         SELECT DISTINCT ?s ?src_label ?dst
WHERE {
   ?s a ?dst .
   ?s rdfs:label ?src_label.
} """
        qres = self.g.query(typing)
        d = defaultdict(set)
        for x in qres:
            d[str(x.src_label)].add(str(x.dst)[len(Parmenides.parmenides_ns):])
        return d

    def dumpTypedObjectsToTAB(self, filename: str | io.IOBase):
        l = self.getTypedObjects()
        n = len(l)
        f = None
        if isinstance(filename, io.IOBase):
            f = filename
        else:
            f = open(str(filename), "w")
        count = 1
        for k, v in l.items():
            k, t = k, self.most_specific_type(v)
            f.write(f"{count}\t{k}\t{k}\t{t}")
            count += 1
            if count <= n:
                f.write(os.linesep)
        return f

    def isA(self, src, type):
        knows_query = """
         SELECT DISTINCT ?src ?dst
         WHERE {
             ?src a ?dst.
             ?src rdfs:label ?src_label.
         }"""
        bindings = {}
        srcBool = False
        dstBool = False
        if not src.startswith("^"):
            bindings["src_label"] = Literal(src, datatype=XSD.string)
        else:
            srcBool = True
        if not type.startswith("^"):
            bindings["dst"] = URIRef(Parmenides.parmenides_ns[type])
        else:
            dstBool = True
        qres = self.g.query(knows_query, initBindings=bindings)
        for x in qres:
            d = x.asdict()
            k = dict()
            k["@^hasResult"] = True
            if srcBool:
                k[src[1:]] = str(d.get("src_label"))
            if dstBool:
                k[type[1:]] = str(d.get("dst"))[len(Parmenides.parmenides_ns):]
            yield k

    def _run_custom_sparql_query(self, query, bindings=None):
        ## You can test custom SPARQL queries in https://atomgraph.github.io/SPARQL-Playground/
        qres = self.g.query(query, initBindings=bindings)
        for x in qres:
            yield x.asdict()

    def collect_prepositions(self):
        ## This is the way to have explicit caching and control
        if self.prepositions is not None:
            return self.prepositions
        query = """
SELECT *
WHERE {
    { ?src a <https://logds.github.io/parmenides#Preposition>. }
    UNION
{ ?s a ?t. ?t rdfs:subClassOf  <https://logds.github.io/parmenides#Preposition>. }
 ?src rdfs:label ?src_label.
}"""
        from LaSSI.Parmenides.Prepositions import Preposition
        result = defaultdict(dict)
        for d in self._run_custom_sparql_query(query):
            rel = str(d.get("t", ""))[len(Parmenides.parmenides_ns):]
            # rel = str(d.get("rel", ""))[len(Parmenides.parmenides_ns):]
            label = str(d["src_label"])
            local = result[label]
            # print(label)
            result[label] = Preposition.update_with_label(local, rel)
        query = """
        SELECT *
        WHERE {
         ?src ?prop ?value.
         ?prop a owl:ObjectProperty.
         ?src rdfs:label ?label.
        }"""
        for k in result.keys():
            binding = {"label": Literal(k, datatype=XSD.string)}
            for d in self._run_custom_sparql_query(query, bindings=binding):
                rel = str(d.get("prop", ""))[len(Parmenides.parmenides_ns):]
                if isinstance(d["value"], Literal):
                    # print(rel)
                    result[k][rel] = d["value"].value

        self.prepositions = dict()
        for k, v in result.items():
            v["name"] = k
            self.prepositions[k] =  dacite.from_dict(Preposition, v)
        return self.prepositions

    def single_edge_dst_unary_capability(self, src, edge_type, verb, subj):
        knows_query = """
         SELECT DISTINCT ?src ?edge_type ?dst ?src_label ?verb ?subj
         WHERE {
             ?src ?edge_type ?dst.
             ?dst parmenides:entryPoint ?verb_e.
             ?verb_e rdfs:label ?verb. 
             ?dst parmenides:subject ?subj_e.
             ?subj_e rdfs:label ?subj. 
             ?src rdfs:label ?src_label.
         }"""
        bindings = {}
        srcBool = False
        edgeBool = False
        verbBool = False
        subjBool = False
        objBool = False
        if not src.startswith("^"):
            bindings["src_label"] = Literal(src, datatype=XSD.string)
        else:
            srcBool = True
        if not edge_type.startswith("^"):
            bindings["edge_type"] = URIRef(Parmenides.parmenides_ns[edge_type])
        else:
            edgeBool = True
        if not verb.startswith("^"):
            bindings["verb"] = Literal(verb, datatype=XSD.string)
        else:
            verbBool = True
        if not subj.startswith("^"):
            bindings["subj"] = Literal(subj, datatype=XSD.string)
        else:
            subjBool = True
        qres = self.g.query(knows_query, initBindings=bindings)
        for x in qres:
            d = x.asdict()
            k = dict()
            k["@^hasResult"] = True
            if srcBool:
                k[src[1:]] = str(d.get("src_label"))
            if subjBool:
                k[subj[1:]] = str(d.get("subj"))
            if verbBool:
                k[verb[1:]] = str(d.get("verb"))
            if edgeBool:
                k[edge_type[1:]] = str(d.get("edge_type"))[len(Parmenides.parmenides_ns):]
            yield k

    def single_edge_dst_binary_capability(self, src, edge_type, verb, subj, obj):
        knows_query = """
         SELECT DISTINCT ?src ?edge_type ?dst ?src_label ?verb ?subj ?obj
         WHERE {
             ?src ?edge_type ?dst.
             ?dst parmenides:entryPoint ?verb_e.
             ?verb_e rdfs:label ?verb. 
             ?dst parmenides:subject ?subj_e.
             ?subj_e rdfs:label ?subj. 
             ?dst parmenides:d_object ?obj_e.
             ?obj_e rdfs:label ?obj. 
             ?src rdfs:label ?src_label.
         }"""
        bindings = {}
        srcBool = False
        edgeBool = False
        verbBool = False
        subjBool = False
        objBool = False
        if not src.startswith("^"):
            bindings["src_label"] = Literal(src, datatype=XSD.string)
        else:
            srcBool = True
        if not edge_type.startswith("^"):
            bindings["edge_type"] = URIRef(Parmenides.parmenides_ns[edge_type])
        else:
            edgeBool = True
        if not verb.startswith("^"):
            bindings["verb"] = Literal(verb, datatype=XSD.string)
        else:
            verbBool = True
        if not subj.startswith("^"):
            bindings["subj"] = Literal(subj, datatype=XSD.string)
        else:
            subjBool = True
        if not obj.startswith("^"):
            bindings["obj"] = Literal(obj, datatype=XSD.string)
        else:
            objBool = True
        qres = self.g.query(knows_query, initBindings=bindings)
        for x in qres:
            d = x.asdict()
            k = dict()
            k["@^hasResult"] = True
            if srcBool:
                k[src[1:]] = str(d.get("src_label"))
            if subjBool:
                k[subj[1:]] = str(d.get("subj"))
            if objBool:
                k[obj[1:]] = str(d.get("obj"))
            if verbBool:
                k[verb[1:]] = str(d.get("verb"))
            if edgeBool:
                k[edge_type[1:]] = str(d.get("edge_type"))[len(Parmenides.parmenides_ns):]
            yield k

    def single_edge_src_multipoint(self, src, src_spec, edge_type, dst):
        knows_query = """
         SELECT DISTINCT ?src ?edge_type ?dst ?src_label ?src_spec ?dst_label
         WHERE {
             ?src ?edge_type ?dst.
             ?src parmenides:entryPoint ?src_entry.
             ?src_entry rdfs:label ?src_label.
             ?src_multi parmenides:hasAdjective ?src_spec_node.
             ?src_spec_node rdfs:label ?src_spec.
             ?dst rdfs:label ?dst_label .
         }"""
        bindings = {}
        srcBool = False
        srcSpecBool = False
        edgeBool = False
        dstBool = False
        if not src.startswith("^"):
            bindings["src_label"] = Literal(src, datatype=XSD.string)
        else:
            srcBool = True
        if not src_spec.startswith("^"):
            bindings["src_spec"] = Literal(src_spec, datatype=XSD.string)
        else:
            srcSpecBool = True
        if not edge_type.startswith("^"):
            bindings["edge_type"] = URIRef(Parmenides.parmenides_ns[edge_type])
        else:
            edgeBool = True
        if not dst.startswith("^"):
            bindings["dst_label"] = Literal(dst, datatype=XSD.string)
        else:
            dstBool = True
        qres = self.g.query(knows_query, initBindings=bindings)
        for x in qres:
            d = x.asdict()
            k = dict()
            k["@^hasResult"] = True
            if srcBool:
                k[src[1:]] = str(d.get("src_label"))
            if srcSpecBool:
                k[src_spec[1:]] = str(d.get("src_spec"))
            if dstBool:
                k[dst[1:]] = str(d.get("dst_label"))
            if edgeBool:
                k[edge_type[1:]] = str(d.get("edge_type"))[len(Parmenides.parmenides_ns):]
            yield k

    def single_edge(self, src, edge_type, dst):
        m = re.match(r"(?P<main>[^\[]+)\[(?P<spec>[^\]]+)\]", src)
        if m:
            yield from self.single_edge_src_multipoint(m.group('main'), m.group('spec'), edge_type, dst)
            return
        m = re.match(r"(?P<main>[^\(]+)\((?P<subj>[^\,)]+),(?P<obj>[^\)]+)\)", dst)
        if m:
            yield from self.single_edge_dst_binary_capability(src, edge_type, m.group('main'), m.group('subj'),
                                                              m.group('obj'))
            return
        m = re.match(r"(?P<main>[^\(]+)\((?P<subj>[^\)]+)\)", dst)
        if m:
            yield from self.single_edge_dst_unary_capability(src, edge_type, m.group('main'), m.group('subj'))
            return
        knows_query = """
         SELECT DISTINCT ?src ?edge_type ?dst ?src_label ?dst_label
         WHERE {
             ?src ?edge_type ?dst.
             ?src rdfs:label ?src_label.
             ?dst rdfs:label ?dst_label .
         }"""
        bindings = {}
        srcBool = False
        edgeBool = False
        dstBool = False
        if not src.startswith("^"):
            bindings["src_label"] = Literal(src, datatype=XSD.string)
        else:
            srcBool = True
        if not edge_type.startswith("^"):
            bindings["edge_type"] = URIRef(Parmenides.parmenides_ns[edge_type])
        else:
            edgeBool = True
        if not dst.startswith("^"):
            bindings["dst_label"] = Literal(dst, datatype=XSD.string)
        else:
            dstBool = True
        qres = self.g.query(knows_query, initBindings=bindings)
        for x in qres:
            d = x.asdict()
            k = dict()
            k["@^hasResult"] = True
            if srcBool:
                k[src[1:]] = str(d.get("src_label"))
            if dstBool:
                k[dst[1:]] = str(d.get("dst_label"))
            if edgeBool:
                k[edge_type[1:]] = str(d.get("edge_type"))[len(Parmenides.parmenides_ns):]
            yield k

    @staticmethod
    def instantiate_query_with_map(Q, m):

        if Q is None:
            return None
        elif isinstance(Q, list):
            if len(Q) == 2 or len(Q) == 3:
                return list(map(lambda x: instantiateWithMap(x, m), Q))
            else:
                raise ValueError("Len 2 are IsA queries, while the rest are Len 3")
        elif isinstance(Q, tuple) and len(Q) == 2:
            return tuple([Q[0], list(map(lambda x: Parmenides.instantiate_query_with_map(x, m), Q[1]))])
        else:
            raise ValueError(
                "Cases error: a list will identify base queries, while a tuple will identify compound constructions")

    def old_multiple_queries(self, Q):
        if Q is None:
            return pandas.DataFrame()
        elif isinstance(Q, list):
            if len(Q) == 2:
                return pandas.DataFrame(self.isA(Q[0], Q[1]))
            elif len(Q) == 3:
                return pandas.DataFrame(self.single_edge(Q[0], Q[1], Q[2]))
        elif isinstance(Q, tuple):
            assert len(Q) == 2
            if Q[0].lower() == "and":
                M = list(map(self.old_multiple_queries, Q[1]))
                if any(map(lambda x: len(x) == 0, M)):
                    return pandas.DataFrame()
                return reduce(lambda x, y: x.merge(y), M)
            else:
                raise ValueError(Q[0] + " is unexpected")
        else:
            raise ValueError(
                "Cases error: a list will identify base queries, while a tuple will identify compound constructions")

    def multiple_queries(self, Q):
        result = self.old_multiple_queries(Q)
        if result is not None:
            return result.to_dict('records')
        else:
            return []

    @lru_cache(maxsize=128)
    def get_rejected_edges(self):
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a parmenides:Rejectable.
             ?a rdfs:label ?c .
         }"""
        return self._single_unary_query(knows_query, lambda x: str(x.c))

    @lru_cache(maxsize=128)
    def get_causative_verbs(self):
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a parmenides:CausativeVerb.
             ?a rdfs:label ?c .
         }"""
        return self._single_unary_query(knows_query, lambda x: str(x.c))

    @lru_cache(maxsize=128)
    def get_movement_verbs(self):
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a parmenides:MovementVerb.
             ?a rdfs:label ?c .
         }"""
        return self._single_unary_query(knows_query, lambda x: str(x.c))

    @lru_cache(maxsize=128)
    def get_units_of_measure(self):
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a parmenides:UnitOfMeasure.
             ?a rdfs:label ?c .
         }"""
        return self._single_unary_query(knows_query, lambda x: str(x.c))

    @lru_cache(maxsize=128)
    def get_means_verbs(self):
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a parmenides:MeansVerb.
             ?a rdfs:label ?c .
         }"""
        return self._single_unary_query(knows_query, lambda x: str(x.c))

    @lru_cache(maxsize=128)
    def get_abstract_entities(self):
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a parmenides:AbstractEntity.
             ?a rdfs:label ?c .
         }"""
        return self._single_unary_query(knows_query, lambda x: str(x.c))

    @lru_cache(maxsize=128)
    def get_state_verbs(self):
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a parmenides:StateVerb.
             ?a rdfs:label ?c .
         }"""
        return self._single_unary_query(knows_query, lambda x: str(x.c))

    @lru_cache(maxsize=128)
    def get_materialisation_verbs(self):
        knows_query = """
         SELECT DISTINCT ?c
         WHERE {
             ?a a parmenides:MaterialisationVerb.
             ?a rdfs:label ?c .
         }"""
        return self._single_unary_query(knows_query, lambda x: str(x.c))

    @lru_cache(maxsize=128)
    def get_logical_rewriting_rules(self):
        knows_query = """
         SELECT DISTINCT ?label ?rule_order ?preposition ?logicalConstructName ?logicalConstructProperty ?verb_of_motion ?SingletonHasBeenMatchedBy ?not ?abstract_entity ?hasNMod ?hasNModPartOf ?hasNModIsA ?isSymmetricalIfComparedToNMod ?hasNumber ?hasUnitOfMeasure ?verb_of_aims ?verb_of_state ?verb_of_means ?causative_verb ?isMaterializationVerb
         WHERE {
             ?a a parmenides:LogicalRewritingRule.
             ?a rdfs:label ?label .
             ?a parmenides:logicalConstructName ?logicalConstructName .
             ?a parmenides:rule_order ?rule_order .
             OPTIONAL { ?a parmenides:preposition ?preposition }
             OPTIONAL { ?a parmenides:logicalConstructProperty ?logicalConstructProperty }
             OPTIONAL { ?a parmenides:SingletonHasBeenMatchedBy ?SingletonHasBeenMatchedBy }
             OPTIONAL { ?a parmenides:not ?not }
             OPTIONAL { ?a parmenides:abstract_entity ?abstract_entity }
             OPTIONAL { ?a parmenides:hasNMod ?hasNMod }
             OPTIONAL { ?a parmenides:hasNModPartOf ?hasNModPartOf }
             OPTIONAL { ?a parmenides:hasNModIsA ?hasNModIsA }
             OPTIONAL { ?a parmenides:isSymmetricalIfComparedToNMod ?isSymmetricalIfComparedToNMod }
             OPTIONAL { ?a parmenides:causative_verb ?causative_verb }
             OPTIONAL { ?a parmenides:hasNumber ?hasNumber }
             OPTIONAL { ?a parmenides:hasUnitOfMeasure ?hasUnitOfMeasure }
             OPTIONAL { ?a parmenides:verb_of_motion ?verb_of_motion }
             OPTIONAL { ?a parmenides:verb_of_aims ?verb_of_aims }
             OPTIONAL { ?a parmenides:verb_of_state ?verb_of_state }
             OPTIONAL { ?a parmenides:verb_of_means ?verb_of_means }
             OPTIONAL { ?a parmenides:isMaterializationVerb ?isMaterializationVerb }
         }"""
        # return self._single_unary_query(knows_query, lambda x: x)

        not_query = """
         SELECT DISTINCT ?preposition ?verb_of_motion ?SingletonHasBeenMatchedBy ?abstract_entity ?hasNMod ?hasNModPartOf ?hasNModIsA ?isSymmetricalIfComparedToNMod ?hasNumber ?hasUnitOfMeasure ?verb_of_aims ?verb_of_state ?verb_of_means ?causative_verb ?isMaterializationVerb
         WHERE {
             OPTIONAL { ?a parmenides:preposition ?preposition }
             OPTIONAL { ?a parmenides:SingletonHasBeenMatchedBy ?SingletonHasBeenMatchedBy }
             OPTIONAL { ?a parmenides:abstract_entity ?abstract_entity }
             OPTIONAL { ?a parmenides:hasNMod ?hasNMod }
             OPTIONAL { ?a parmenides:hasNModPartOf ?hasNModPartOf }
             OPTIONAL { ?a parmenides:hasNModIsA ?hasNModIsA }
             OPTIONAL { ?a parmenides:isSymmetricalIfComparedToNMod ?isSymmetricalIfComparedToNMod }
             OPTIONAL { ?a parmenides:causative_verb ?causative_verb }
             OPTIONAL { ?a parmenides:hasNumber ?hasNumber }
             OPTIONAL { ?a parmenides:hasUnitOfMeasure ?hasUnitOfMeasure }
             OPTIONAL { ?a parmenides:verb_of_motion ?verb_of_motion }
             OPTIONAL { ?a parmenides:verb_of_aims ?verb_of_aims }
             OPTIONAL { ?a parmenides:verb_of_state ?verb_of_state }
             OPTIONAL { ?a parmenides:verb_of_means ?verb_of_means }
             OPTIONAL { ?a parmenides:isMaterializationVerb ?isMaterializationVerb }
         }"""

        str_premises = {"preposition", "verb_of_motion", "SingletonHasBeenMatchedBy", "abstract_entity", "hasNMod", "hasNModPartOf", "hasNModIsA", "isSymmetricalIfComparedToNMod", "hasNumber", "hasUnitOfMeasure", "verb_of_aims", "verb_of_state", "verb_of_means", "causative_verb", "isMaterializationVerb"}

        rules = defaultdict()

        query_rules = list(self._single_unary_query(knows_query, lambda x: x))
        query_rules.sort(key=lambda rule: rule.rule_order)  # So list is ordered numerically by rule_order
        grouped_rules = defaultdict(list)
        for rule in query_rules:
            grouped_rules[rule.rule_order].append(rule)

        for gr_key, grouped_rule in grouped_rules.items():
            premises = defaultdict(list)
            not_premises = defaultdict(list)
            logical_construct_name = None
            logical_construct_property = None
            for rule in grouped_rule:
                logical_construct_name = rule.logicalConstructName.value if logical_construct_name is None else logical_construct_name
                logical_construct_property = rule.logicalConstructProperty.value if logical_construct_property is None and rule.logicalConstructProperty is not None else logical_construct_property
                for str_premise in str_premises:
                    if hasattr(rule, str_premise) and getattr(rule, str_premise) is not None:
                        premises[str_premise].append(getattr(rule, str_premise).value)
                if hasattr(rule, "not") and getattr(rule, "not") is not None:
                    not_query_premises = self.g.query(not_query, initBindings={'a': getattr(rule, "not")})
                    for not_query_premise in not_query_premises:
                        for str_premise in str_premises:
                            if hasattr(not_query_premise, str_premise) and getattr(not_query_premise, str_premise) is not None:
                                not_premises[str_premise].append(getattr(not_query_premise, str_premise).value)

            con_premises = list()
            for p_key, premise in premises.items():
                con_premises.append(Condition(
                    name=p_key,
                    values=list(set(premise))
                ))

            neg_con_premises = list()
            for np_key, not_premise in not_premises.items():
                neg_con_premises.append(Condition(
                    name=np_key,
                    values=list(set(not_premise))
                ))

            rules[gr_key.value] = Rule(
                id=gr_key.value,
                premises=con_premises,
                not_premises=neg_con_premises,
                logicalConstructName=logical_construct_name,
                logicalConstructProperty=logical_construct_property
            )

        return rules

    @lru_cache(maxsize=128)
    def get_logical_functions(self, logical_construct_name, logical_construct_property):
        knows_query = """
         SELECT DISTINCT ?label ?attachTo ?argument
         WHERE {
             ?a a parmenides:LogicalFunction.
             ?a rdfs:label ?label .
             ?a parmenides:logicalConstructName ?logicalConstructName .
             OPTIONAL { ?a parmenides:logicalConstructProperty ?logicalConstructProperty }
             ?a parmenides:attachTo ?attachTo .
             ?a parmenides:argument ?argument .
         }"""

        logical_functions = list()
        query_functions = self.g.query(knows_query, initBindings={'logicalConstructName': Literal(logical_construct_name, datatype=XSD.string), 'logicalConstructProperty': Literal(logical_construct_property, datatype=XSD.string)})

        for function in query_functions:
            logical_functions.append(LogicalRewritingRule(
                label=function.label.value,
                attachTo=function.attachTo.value,
                logicalConstructName=logical_construct_name,
                logicalConstructProperty=logical_construct_property
            ))

        return logical_functions

@dataclass()
class LogicalRewritingRule:
    label: str
    attachTo: str
    logicalConstructName: str
    logicalConstructProperty: Optional[str]

@dataclass()
class Condition:
    name: str
    values: list

@dataclass()
class Rule:
    id: int
    premises: List[Condition]
    not_premises: List[Condition]
    logicalConstructName: str
    logicalConstructProperty: Optional[str]


if __name__ == "__main__":
    g = Parmenides(filename="/home/fox/PycharmProjects/LaSSI-python/LaSSI/resources/turtle.ttl")

    # test_arr = sorted([x for x in g.logical_rewriting_rules if str(x.preposition) in {"at"}], key=lambda x: int(x.rule_order))
    # print(test_arr)

    print([x for x in g.getNounsWithProperties() if "street" == str(x.label) and "corner" in str(x.hasProperty)])

    # print({str(x)[len(Parmenides.parmenides_ns):] for x in g.typeOf("fabulous")})
    # L = list(g.collect_prepositions())
    # print(L)
    # l = g.dumpTypedObjectsToTAB("tabby.tab")
    # count = 1
    # for k, v in l.items():
    #     k, t = k, g.most_specific_type(v)
    #     print(f"{count}\t{k}\t{t}")
    #     count += 1
    # # knows_query = """
    # # SELECT DISTINCT ?c
    # # WHERE {
    # #     ?a a parmenides:Rejectable.
    # #     ?a rdfs:label ?c .
    # # }"""
    # # qres = g.query(knows_query)
    # w = g.old_multiple_queries(tuple(["and", [["^x", "^y", "^z"], ["slow", "Adjective"]]]))
    # for hasEdge in g.single_edge("city center", "partOf", "^var"):
    #     print(hasEdge)
    # for outcome in g.isA("flow", "^t"):
    #     print(outcome)
    # for outcome in g.isA("busy", "Adjective"):
    #     print(outcome)
    # for outcome in g.single_edge("city[busy]", "relatedTo", "^d"):
    #     print(outcome)
    # for outcome in g.single_edge("traffic jam", "capableOf", "^v(^s,^o)"):
    #     print(outcome)
    # print(g.typeOf("Newcastle"))
    # print(g.typeOf("city"))
    # print(g.typeOf("flow"))
