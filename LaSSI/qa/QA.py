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
#--- rdflib 6.3.2, dependencies in pyproject.toml --- make a new project with a new env, install rdflib there
import rdflib
from FunctionalMatch.language.LanguageMainPoint import parse_query
from rdflib.graph import Graph, ConjunctiveGraph
from rdflib import Graph, URIRef, BNode, Literal, XSD
from rdflib import Namespace
from rdflib.namespace import OWL, RDF, RDFS, FOAF

from LaSSI.Parmenides.Parmenides import ParmenidesSingleton
from LaSSI.Parmenides.make_franco_from_scratch import make_ontology_from_raw
from LaSSI.structures.extended_fol.Formulae import FVariable, FUnaryPredicate, FBinaryPredicate, FNot
from LaSSI.structures.extended_fol.TBoxReasoning import KnowledgeExpansion


#from LaSSI.Parmenides import Prepositions, SentenceStructure
# --> import below from formula.py
# from LaSSI.structures.extended_fol.Formulae import FUnaryPredicate
# from src.Formulae import FVariable, FUnaryPredicate, FBinaryPredicate, FNot # had to add "src." because then the type checks wont work cuz the rules txt imports with src
# from FunctionalMatch.language.LanguageMainPoint import parse_query
# from TBoxReasoning import KnowledgeExpansion
# from Parmenides import ParmenidesSingleton
# from LaSSI.structures.extended_fol.TBoxReasoning import KnowledgeExpansion   #### tboxreasoning.py


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
        self.create_relationship("adjectivalForm")
        self.create_relationship("adverbialForm")
        self.create_relationship("entryPoint")

        # self.create_relationship("hasProperty")
        # self.create_relationship("formOf")
        # self.create_relationship("partOf")
        # self.create_relationship("isa")
        # self.create_relationship("relatedTo")
        # self.create_relationship("capableOf")
        # self.create_relationship("eq")
        # self.create_relationship("noteq")

        self.create_relationship("AtLocation")
        self.create_relationship("CapableOf")
        self.create_relationship("Causes")
        self.create_relationship("CausesDesire")
        self.create_relationship("CreatedBy")
        self.create_relationship("DefinedAs")
        self.create_relationship("DerivedFrom")
        self.create_relationship("Desires")
        self.create_relationship("notisa")  # from "DistinctFrom"
        self.create_relationship("Entails")
        self.create_relationship("etymology")
        self.create_relationship("related_to")
        self.create_relationship("form_of")
        self.create_relationship("HasA")
        self.create_relationship("HasContext")
        self.create_relationship("HasFirstSubevent")
        self.create_relationship("HasLastSubevent")
        self.create_relationship("HasPrerequisite")
        self.create_relationship("HasProperty")
        self.create_relationship("HasSubevent")
        self.create_relationship("instance_of")
        self.create_relationship("isa")
        self.create_relationship("LocatedNear")
        self.create_relationship("MadeOf")
        self.create_relationship("MannerOf")
        self.create_relationship("MotivatedByGoal")
        self.create_relationship("notCapableOf")
        self.create_relationship("notDesires")
        self.create_relationship("notHasProperty")
        self.create_relationship("notUsedFor")
        self.create_relationship("ObstructedBy")
        self.create_relationship("part_of")
        self.create_relationship("ReceivesAction")
        self.create_relationship("shares_isa_with")
        self.create_relationship("SymbolOf")
        self.create_relationship("eq")
        self.create_relationship("noteq")
        self.create_relationship("UsedFor")
        self.create_relationship("capital")
        self.create_relationship("field")
        self.create_relationship("genre")
        self.create_relationship("genus")
        self.create_relationship("influencedBy")
        self.create_relationship("knownFor")
        self.create_relationship("language")
        self.create_relationship("leader")
        self.create_relationship("occupation")
        self.create_relationship("product")
        self.create_relationship("SubwordOf")
        self.create_relationship("class")
        self.create_relationship("with_sense")

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

def parmenides_db_write():
    from LaSSI.Parmenides.Parmenides import ParmenidesSingleton
    ParmenidesSingleton.instance()
    ParmenidesSingleton.init("/home/parallels/PycharmProjects/LaSSI/cache", "lassi", "drowssap",
                             "localhost", 5432, False, "franco_parmenides.ttl") # folder where intermediate computations will be kept, change to whatever

    #### do this:
    # pip install git+https://github.com/LogDS/FunctionalMatch.git

    from FunctionalMatch.language.LanguageMainPoint import parse_query
    queries = parse_query("/home/parallels/PycharmProjects/LaSSI/query_franco.txt") # change

    ke = KnowledgeExpansion("/home/parallels/PycharmProjects/LaSSI/LaSSI/_kexp.pickle") # change to whatever

    ### writing out the questions as logical rep as if other lassi module did it
    #from LaSSI.structures.extended_fol.Formulae import FVariable



    # 2nd arg: for stuff like ?1, make it existential ('there exists')
    var = FVariable("?1", "existential", None, None, 1)
    cable = FVariable("cable", "ENTITY", None, None, 1)
    tv = FVariable("television", "ENTITY", None, None, 1)
    it = FVariable("it", "ENTITY", None, None, 1)
    he = FVariable("equipment", "ENTITY", "home entertainment", None, 1)
    #from LaSSI.structures.extended_fol.Formulae import FBinaryPredicate
    q1 = FBinaryPredicate("isa", var, he, -1, frozenset())
    q2 = FBinaryPredicate("require", var, cable, -1, frozenset())
    q3 = FBinaryPredicate("isa", var, he, -1, frozenset())
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
    # ParmenidesSingleton.stop()

def q1():
    # Sammy wanted to go to where the people were.  Where might he go?
    sammy = FVariable("sammy", "person", None, None, 1)
    people = FVariable("people", "ENTITY", None, None, 1)
    where = FVariable("?1", "ENTITY", None, None, 1)

    people_be = FUnaryPredicate("be", people, 1, frozenset( {("SPACE", where)})  )
    go = FUnaryPredicate("go", sammy, 1, frozenset({("SPACE", people)})  )                                           ## maybe change [space: where] to [space: people]

    ## should there be an extra predicate so the system knows sammy is a person who can do person things

    return go, people_be # FAnd((go, people_be))

def q2():
    # To locate a choker not located in a jewelry box or boutique where would you go?
    choker = FVariable("choker", "ENTITY", None, None, 1)
    jewelry_box = FVariable("jewelry box", "ENTITY", None, None, 1)
    boutique = FVariable("boutique", "ENTITY", None, None, 1)
    you = FVariable("you", "person", None, None, 1)
    where = FVariable("?1", "ENTITY", None, None, 1)

    ## would is a verb but im skipping over it

    #FNot(FUnaryPredicate("locate", choker, 1, frozenset(("SPACE", (jewelry_box,)))))


    #locate_not_box = FUnaryPredicate("locate", choker, 1, frozenset( ("SPACE", (FNot(jewelry_box),)) ) )
    #locate_not_bout = FUnaryPredicate("locate", choker, 1, frozenset(("SPACE", (FNot(boutique),))))

    locate_not_box = FNot(FUnaryPredicate("locate", choker, 1, frozenset({("SPACE", jewelry_box)})))
    locate_not_bout = FNot(FUnaryPredicate("locate", choker, 1, frozenset({("SPACE", boutique)})))
    locate = FUnaryPredicate("locate", choker, 1, frozenset({("SPACE", where)}))
    go = FUnaryPredicate("go", you, 1, frozenset({("SPACE", where)}))

    return locate_not_box, locate_not_bout, locate, go

## Parmenides is currently holding ABox (look at wikipedia page of it)
# queries_old.txt is the TBox
def q3():
    # What home entertainment equipment requires cable?
    cable = FVariable("cable", "ENTITY", None, None, 1)
    home_enter = FVariable("equipment", "ENTITY", "home entertainment", None, 1)
    what = FVariable("?1", "existential", None, None, 1)

    # home enter equip is not a specification of 'what', it should be empty

    is_a = FBinaryPredicate("isa", what, home_enter, -1, frozenset()) # Assume LaSSI does this already
    require = FBinaryPredicate("require", what, cable, -1, frozenset())
    # require pred is very specific so there needs to be a specific rule that changes this to something that works with the ontology

    ## isa is a rel in the ontology, find all whats
    ## the expansion will give a rel in the ontology, from the require pred.
    ## then find all the whats from that new pred and then finally do intersection between what sets

    # Dissert can mention using a language similar to natural language
    # and future work could be processing natural language to provide it in a suitable form

    return is_a, require #FAnd((is_a, require))

def q4():
    # The only baggage the woman checked was a drawstring bag, where was she heading with it?
    # The only drawstring baggage the woman checked, where was she heading with it?
    baggage = FVariable("baggage", "ENTITY", "drawstring bag", None, 1) ## remove spec
    # FVariable("bag", "ENTITY", "drawstring", None, 1) ## this is also an option and u can use it in the is_a
    where = FVariable("?1", "ENTITY", None, None, 1)
    woman = FVariable("woman", "person", None, None, 1)
    she = FVariable("she", "ENTITY", None, None, 1)

    # baggage isa drawstringbag
    check = FBinaryPredicate("check", woman, baggage, -1, frozenset()) ## baggage used for traverelling
    headTo = FUnaryPredicate("head", woman, 1, frozenset((("TOGETHERNESS", (baggage,)), ("SPACE", (where,))))) ## in the past, "woman" was "she" but i assume that this is done

    ## should handle "check" by an atlocation for non ontology rel predicates that applies atlocation rule?

    return check, headTo
def q5():
    # ENTITIES are nouns, all nouns are entities

    # The forgotten leftovers had gotten quite old, he found it covered in mold in the back of his what?
    leftovers = FVariable("leftovers", "ENTITY", "forgotten", None, 1) ## u can remove spec here
    he = FVariable("he", "person", None, None, 1)
    it = FVariable("it", "ENTITY", None, None, 1) ### LaSSI automatically replaces it with leftovers
    mold = FVariable("mold", "ENTITY", None, None, 1)
    old = FVariable("old", "ADJECTIVE", None, None, 1)
    x = FVariable("?2", "ENTITY", None, None, 1)
    what = FVariable("?1", "ENTITY", "back", None, 1) ## could be a specification but wont help resolving it

    got = FBinaryPredicate("got", leftovers, old, -1, frozenset()) ## property for forgotten
    # someone/something (x) (missing information) covered it in mold, this is usually indicated by a passive verb (covered).
    # the SPACE property is attributed to the cover predicate since it was the last verb
    cover = FBinaryPredicate("cover", x, leftovers, -1, frozenset((("INSTRUMENT", (mold,)), ("SPACE", (what,)))))
    #cover = FUnaryPredicate("cover", it)
    find = FBinaryPredicate("find", he, leftovers, -1, frozenset())

    # how to handle ?2? should it be in the final intersection or should it be there on the side without interfering

    return got, cover, find



def q6():
    # What do people use to absorb extra ink from a fountain pen?

    # FVariables can also have properties
    people = FVariable("people", "ENTITY", None, None, 1)
    ink = FVariable("ink", "ENTITY", "extra", None, 1)
    pen = FVariable("fountain pen", "ENTITY", None, None, 1)
    what = FVariable("?1", "ENTITY", None, None, 1)

    absorb = FBinaryPredicate("absorb", what, ink, 1, frozenset({("PROVENANCE", pen)}))
    #absorb = FUnaryPredicate("absorb", ink, 1, frozenset(("PROVENANCE", (pen,))))

    # at this point, LaSSi is not dealing with subsentences:
    # e.g. Im drinking water **because** [subsentence reason]
    # so although absorb might actually be the objective, it will boil down to conjuction in the end.
    # so for these types of questions, reasonings and subsentence connections can just be conjuctions, so objective is irrelevant
    #use = FBinaryPredicate("use", people, what, -1, frozenset(("OBJECTIVE", (absorb,)))) ## u can remove objective prop out of simplicity
    use = FBinaryPredicate("use", people, what, -1, frozenset())

    # maybe for use predicate, find a way to change people to person and then do the atlocation rule, what is at location of person
    # for absorb, use the atlocation rule cuz fountain pen is at blotter

    # and it could be just worded as
    return use, absorb

def q7():
    # Where do you put your grapes just before checking out?
    you = FVariable("you", "person", None, None, 1)
    grapes = FVariable("grapes", "ENTITY", None, None, 1)
    where = FVariable("?1", "ENTITY", None, None, 1)

    put = FBinaryPredicate("put", you, grapes, -1, frozenset(("SPACE", (where,))))
    check_out = FUnaryPredicate("check out", you, -1, frozenset(("SPACE", (where,))))
    # how to include "checking out": it would be an "AND"

    # apply atlocation on put nodes: you, grapes. grapes would give shopping cart as a location (and store)
    # check_out. going to store has last subservent check out. idk how but try to connect that and u can use atlocation since both processes are at location store
    # then the things that are nearby location include the shopping cart.

    ## ask what i should do with nodes that are phrases rather than words.
    # ask what to do with nodes like "you"

    return put, check_out # put and check_out

def q8():
    # Johnny sat on a bench and relaxed after doing a lot of work on his hobby.  Where is he?
    johnny = FVariable("johnny", "person", None, None, 1)
    bench = FVariable("bench", "ENTITY", None, None, 1)
    hobby = FVariable("hobby", "ENTITY", None, None, 1)
    where = FVariable("?1", "ENTITY", None, None, 1)
    work = FVariable("work", "ENTITY", "lot", None, 1)

    ##### IMPORTANT frozenset is a collection of pairs, even if u have 1 pair, it should be in a tuple
    sit = FUnaryPredicate("sit", johnny, -1, frozenset((("SPACE", (bench,)))))
    relax = FUnaryPredicate("relax", johnny, -1, frozenset(("SPACE", (where,)))) ## make rule: isUsedFor(SPACE, verb)
    # work = FUnaryPredicate("work", johnny, -1, frozenset())
    # so do pred could go without the SPACE property since the work might not be in the same place as he is in now.
    # vvvvv PARTITIVE because a lot of work was done on his hobby, a lot being a part.
    do = FBinaryPredicate("do", johnny, work, -1, frozenset((("SPACE", (where,)), ("PARTITIVE", (hobby,))))) ## idk if i should keep space prop here
    be = FUnaryPredicate("be", johnny, -1, frozenset(("SPACE", (where,)))) ## this is for the 'Where is he?'

    # do predicate can ahve rule isUsedFor(SPACE, dst)


    return sit, relax, do, be

def q9():
    # What is it called when you slowly cook using a grill?
    you = FVariable("you", "person", None, None, 1)
    grill = FVariable("grill", "ENTITY", None, None, 1)
    what = FVariable("?1", "ENTITY", None, None, 1)
    some_food = FVariable("?2", "ENTITY", None, None, 1)
    slowly = FVariable("slowly", "ADVERB", None, None, 1)

    # call(?1, cook(you, ?2))
    use = FBinaryPredicate("use", you, grill, -1, frozenset())
    # u can remove slowly prop
    cook = FBinaryPredicate("cook", you, some_food, -1, frozenset((("OBJECTIVE", (use,)), ("MODE", (slowly,)))))
    call = FBinaryPredicate("call", what, cook, -1, frozenset())

    # ?1 is related to cook/grill
    # preds can be made for this and to make it simpler, they could be added here: FAnd(use, cook, call, rel_to_grill, rel_to_cook)



    return use, cook, call, rel_to_grill, rel_to_cook



questions = [q1, q2, q3, q4, q5, q6, q7, q8, q9]

answer_sets = [
    {"race track", "populated areas", "the desert", "apartment", "roadblock"},
    {"jewelry store", "neck", "jewelry box", "boutique"},
    {"radio shack", "substation", "cabinet", "television", "desk"},
    {"garbage can", "military",  "jewelry store", "safe", "airport"},
    {"carpet", "refrigerator", "breadbox", "fridge", "coach"},
    {"Shirt pocket", "calligrapher’s hand",  "inkwell", "desk drawer", "blotter"}
]

# for printing a predicate in non-latex form
def clean_print(pred):
    if isinstance(pred, FNot):
        pred = pred.arg
        if isinstance(pred, FBinaryPredicate):
            print(pred.src.name, pred.rel, pred.dst.name, pred.properties)
        else:
            print(pred.rel, pred.arg.name, pred.properties)
    elif isinstance(pred, FBinaryPredicate):
        print(pred.src.name, pred.rel, pred.dst.name, pred.properties)
    else:
        print(pred.rel, pred.arg.name, pred.properties)

def knowledge_expand(question, answer_set):
    ke = KnowledgeExpansion(r".\_kexp.pickle")
    # ke.pruned_expansion(question, rules, "rules")
    impl_rules = parse_query(r"../../query_impl.txt")

    import collections

    is_pred_idx_fnot = list()
    variable_result_sets = collections.defaultdict(list)
    # variable_result_sets = {
    # [1] : [answers, answers, answers,...]  # for ?1
    # [2] : [answers, answers,...]  # for ?2

    # later the intersections are done within each list:
    # [1] : [intersected_answers]
    # [2] : ...

    result_sets = list()
    for i, original_pred in enumerate(question):
        print("NEW PRED ROUND ", end="")
        clean_print(original_pred)

        isNot = type(original_pred) == FNot
        is_pred_idx_fnot.append(isNot)

        ke.pruned_expansion(original_pred, impl_rules, "rule")
        predicates = ke.get_full_expansion(original_pred, "rule")
        predicates.add(original_pred) # i add the original pred because it can be useful if it's something like isa(apple, ?1)

        # for pred in predicates:
        #     print("rule id and info:", ke.constituents.add_with_wasPresent(pred)[0], ke.Graph[ke.constituents.add_with_wasPresent(pred)[0]])
        #     clean_print(pred)
        #     print()

        def DFS(current, target, visited, path):
            if current == target:
                return path

            visited.add(current)
            #print("hi:", ke.Graph[current])
            for (_, rule_idx), neighbor in ke.Graph[current]:
                if neighbor not in visited:
                    new_path = path + [(current, rule_idx)]
                    result = DFS(neighbor, target, visited, new_path)
                    if result is not None:
                        return result

            return None

        def subpath_check(path, potential_subpath):
            if len(potential_subpath) > len(path): return False
            for i in range(len(path) - len(potential_subpath) + 1):
                if path[i : i + len(potential_subpath)] == potential_subpath:
                    return True
            return True

        # this is for debugging purposes.
        # it prints out the traces/paths of predicate derivations so u can see which rules were applied on what predicate to get another predicate
        existing_paths = collections.defaultdict(list)
        for pred in predicates:
            start_idx, _ = ke.constituents.add_with_wasPresent(pred)
            for pred in predicates:
                target_idx, _ = ke.constituents.add_with_wasPresent(pred)
                #print(ke.constituents.add_with_wasPresent(pred)[1])
                path = DFS(start_idx, target_idx, set(), [])
                if not path: continue
                path.reverse()

                # sometimes a path is a subpath of a previous path so the subpath must be dropped
                # because otherwise it will be printed and it will clutter the output
                is_subpath = False
                other_subpaths = []
                for other_path in existing_paths[target_idx]:
                    if subpath_check(other_path, path):
                        is_subpath = True
                        break
                    elif subpath_check(path, other_path):
                        other_subpaths.append(other_path)

                if is_subpath: continue

                for subpath in other_subpaths:
                    existing_paths[target_idx].remove(subpath)
                existing_paths[target_idx].append(path)

                print("new trace:")
                clean_print(pred)

                for (current_idx, rule_idx) in path:
                    print("\t↑")
                    print("rule id", rule_idx, )
                    print("applied to: ", end="")
                    clean_print(ke.constituents.fromId(current_idx))
                print()
                print()
        print()
        print()

        #predicates = ke.fullGraph().keys()

        # https: // www.geeksforgeeks.org / defaultdict - in -python /
        # https: // stackoverflow.com / questions / 5228158 / cartesian - product - of - a - dictionary - of - lists
        #
        # https: // www.geeksforgeeks.org / python - itertools - product /
        # Itertools.Product() – Python | GeeksforGeeks
        #
        # d = defaultdict(set)
        # ...
        # itertools.product(d.values())

        unknown_var_id = 0
        unknown_var = None # this will be found below and it will be set to the same thing multiple times but it's fine
        answers = set()
        for pred in predicates:
            #print(pred, type(pred), type(pred) == FNot)

            # FNots are skipped
            if isinstance(pred, FNot): continue # this doesnt account for predicates with unknown vars and wrapped in FNot but for now none of the questions are like that
            # unary predicates are skipped since answer retrieval cant be done on it, only on binary predicates
            if type(pred) == FUnaryPredicate: continue
            if pred.src.name[0] != "?" and pred.dst.name[0] != "?": continue # preventing answer retrieval when there is no answer to retrieve

            print(str(pred.src.name), pred.rel, pred.dst.name)
            if pred.src.name[0] == "?":
                results = ParmenidesSingleton.get().getIngoingNodes(pred.dst.name, pred.rel)
                unknown_var_id = int(pred.src.name[1])
                unknown_var = pred.src.name
            else:
                results = ParmenidesSingleton.get().getOutgoingNodes(pred.src.name, pred.rel) ## getoutgoing
                unknown_var_id = int(pred.dst.name[1])
                unknown_var = pred.dst.name
            print("nodes: ", results)
            #total_results.append(results)
            answers = answers.union(results) # idk about the effectiveness of unioning
            print()

        if isNot:
            # then the original pred and its expansions are FNots so those relations must be removed from ontology.
            # As they are literally removed from the ontology (and not placed back in once this QA is done, at least in this version),
            # if u need to do QA on another question, the program must be reran.
            # An alternative is to not remove these relations from the ontology and instead get the answers from FNot predicates
            # and do a "difference" operation to remove them from the answers.
            # I was previously doing this but by removing these relations from the ontology entirely,
            # the relations are severed so other predicates cannot be generated by going through those relations and
            # answers that are from those relations cannot be acquired.
            p = ParmenidesSingleton.get()
            n = p.names
            for pred in predicates: ### these r FNot predicates with known variables, this is only for question 2
                pred = pred.arg
                if isinstance(pred, FUnaryPredicate): continue
                p.graph.remove((n[pred.src.name], p.relationships.get(pred.rel), n[pred.dst.name]))

        #result_sets.append(answers)
        if unknown_var:
            variable_result_sets[unknown_var].append(answers)
            print("nodes union: ", answers)
        # if isNot and not final_answers is None:
        #     final_answers = final_answers.difference(answers)
        #     continue

        # final_answers = answers if final_answers is None else final_answers.intersection(answers)
        # final_answers = total_results[1]
        # for i in range(1, len(total_results)):
        #     final_answers.intersection(total_results[i])
        # print(final_answers)
        # print("final answers:", final_answers)

    ## should give a list of unions where unions of FNots are at the end of the list
    # result_sets = sorted(result_sets, key=lambda x: is_pred_idx_fnot[result_sets.index(x)]) # using list.sort causes issues with .index cuz references r different i guess
    # print(result_sets)
    #
    # for i, s in enumerate(result_sets):
    #     if is_pred_idx_fnot[i]:
    #         final_answers = None if final_answers is None else final_answers.difference(s)
    #     else:
    #         final_answers = result_sets[0] if final_answers is None else final_answers.intersection(s)

    for unknown_var_name, sets in variable_result_sets.items():
        final_answers = sets[0]
        if len(sets) == 1: continue

        for j in range(1, len(sets)):
            final_answers = final_answers.intersection(sets[j])

        variable_result_sets[unknown_var_name] = [final_answers]

    #final_answers = final_answers if final_answers else set()
    print("final answer chosen from the answer set:", answer_set.intersection(variable_result_sets["?1"][0]))

    # how to visualize:
    # https: // github.com / LogDS / LaSSI / blob / giacomo / extra / as_reviewer.py
    # ignore yaml and for json provide a list of latex of this graph
    #
    # nvm, use franco.py and provide a list of equations to visualize the result of KE


def solve_questions():
    make_ontology_from_raw()

    # rules = parse_query(r".\src\queries_test.txt")#
    question_idx = 6 # set question number here (ranging from 1-9)


    question_idx -= 1
    question = [*questions[question_idx]()]
    knowledge_expand(question, answer_sets[question_idx])

    #### experiment for QA is f1, recall, accuracy
    ## the evaluation is how many answers are the right one
    # commonsenseQA questions have answers that i am particularly interested in.


if __name__ == "__main__":
    make_ontology_from_raw()
    parmenides_db_write()

    solve_questions()
