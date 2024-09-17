import copy
import json
from typing import List

from LaSSI.Parmenides.TBox.SimpleDataMatch import boolean_simple_data_match
from LaSSI.Parmenides.TBox.language.TBoxParse import UpdateOperation, parse_query
from LaSSI.Parmenides.paremenides import Parmenides
from LaSSI.structures.extended_fol.Sentences import formula_from_dict, FBinaryPredicate, make_variable, make_name, FUnaryPredicate, \
    PostProcessingOperations, RemovePropertiesFromResult, AddPropertyFromResult, InheritProperties, Formula, FNot


def object_magic(id):
    import ctypes
    return ctypes.cast(id, ctypes.py_object).value

def transitive_closure(a):
    closure = set(a)
    while True:
        new_relations = set((x,w) for x,y in closure for q,w in closure if q == y)
        closure_until_now = closure | new_relations
        if closure_until_now == closure:
            break
        closure = closure_until_now
    return closure

# def do_match_old(formula, qq, onto_query, p, replacement_map):
#     # Lz = []
#     from collections import defaultdict
#     from Parmenides.TBox.SentenceMatch import structure_dictionary
#     d = defaultdict(list)
#     fugitive = dict()
#     # for x in match(formula, q):
#     # print(formula)
#
#     ## 0. Preserving the original ids
#     ORIG = set(formula.collectIds())
#     ## 1. Matching the query with the data
#     rw_1 = formula.matchWith(qq, d, None, fugitive)
#     ## 2. Within the data, I'm applying the substitution required by the matching and rewriting semantics
#     rw_2 = rw_1.replaceWith(replacement_map, d=d, fugitive=fugitive)
#
#
#     ## 3. Defining the correspondences across the same variable object and across matchings
#     dd = structure_dictionary(d)
#     print(dd)
#     N = len(dd)
#     for i in range(N):
#         ## 4. For each replacement record m
#         m = dd.to_dict('records')[i]
#         ## 5. Using this to instantiate the parametrized ontology query using the matched data
#         QInst = Parmenides.instantiate_query_with_map(onto_query, m)
#         ## 6. Running the ontology query over the instantiated query
#         Ly = list(p.multiple_queries(QInst))
#         if len(Ly) > 0:  ## Determining the match only if we have one element
#             for z in Ly:
#                 m = m | z
#             # 7. Determining the rewriting transformation for each element of interest
#             ddd = dict()
#             for k,v in m.items():
#                 if k != "obj":
#                     ddd["@"+str(k)] = v
#                 else:
#                     ddd[k] = v
#             # 8. Partially instantiating the matched data to be rewritten
#             rw_2 = rw_2.replaceWith(ddd, True, d, fugitive=fugitive)
#             dd = structure_dictionary(d)
#
#     ## 9. Determining the unresolved objects
#     TCL = transitive_closure(fugitive.items())
#     DST = set(rw_2.collectIds())
#     DST = {x[0] for x in TCL}.intersection(DST)
#     ORIG = {x[1] for x in TCL}.intersection(ORIG)
#     object_magic(list(DST)[1]).isUnresolved()
#     finalReplacement = dict({object_magic(x[0]):object_magic(x[1]) for x in TCL if (x[0] in DST and object_magic(x[0]).isUnresolved()) and x[1] in ORIG})
#     if len(finalReplacement) > 0:
#         rw_2 = rw_2.replaceWith(finalReplacement)
#     return rw_2 ## Returning the variable instantiation for both the ontology match as well as the node match


# def jjj(fugitive_init, rw_2, self_ORIG, self_inherit_properties_from):
#     TCL = transitive_closure(fugitive_init.items())
#     ## Determining actually all the sub-expressions associated to the final rewritten object
#     DST = set(rw_2.collectIds())
#     ## Determining which objects from the origin and the end actually appear in the matches thorugh
#     ## the transitive closure and matching
#     DST = {x[0] for x in TCL}.intersection(DST)
#     ORIG = {x[1] for x in TCL}.intersection(self_ORIG)
#     finalReplacement = dict({object_magic(x[0]): object_magic(x[1]) for x in TCL if
#                              (x[0] in DST and object_magic(x[0]).isUnresolved()) and x[1] in ORIG})
#     forAllProperties = None
#     if self_inherit_properties_from:
#         forAllProperties = dict({object_magic(x[0]): object_magic(x[1]) for x in TCL if
#                                  (x[0] in DST and (not object_magic(x[0]).isUnresolved())) and x[1] in ORIG})
#         if len(forAllProperties) == 0:
#             forAllProperties = None
#     return (TCL, DST, ORIG, finalReplacement, forAllProperties)

class DoMatchRec:
    def __init__(self, onto_query, p, ORIG, rewritings, original):
        self.onto_query = onto_query
        self.p = p
        self.ORIG = ORIG
        self.result = []
        self.original = original
        if rewritings is not None:
            self.inherit_properties_from = len(list(filter(lambda x : isinstance(x, InheritProperties), rewritings)))>0
            self.del_rewritings = set(map(lambda x: x.ofField, filter(lambda x : isinstance(x, RemovePropertiesFromResult), rewritings)))
        else:
            self.del_rewritings = set()
            self.inherit_properties_from = False


    def do_expansion_match_iterative(self, N, struct_dict_dd, expansion):
        for i in range(N):
            data_match_morphism = struct_dict_dd.to_dict('records')[i]
            ## 5. Using this to instantiate the parametrized ontology query using the matched data
            QInst = Parmenides.instantiate_query_with_map(self.onto_query, data_match_morphism)
            Ly = list(self.p.multiple_queries(QInst))
            if len(Ly) > 0:
                for ontology_navigation_morphism in Ly:
                    merged_morphism = data_match_morphism | ontology_navigation_morphism
                    merged_morphism_ddd = dict()
                    for k, v in merged_morphism.items():
                        if k != "obj":
                            merged_morphism_ddd["@" + str(k)] = v
                    forAllProperties = None if not self.inherit_properties_from else {expansion.arg if isinstance(expansion, FNot) else expansion:self.original}
                    rw_2_dis = expansion.replaceWith(merged_morphism_ddd, True, forAllProperties=forAllProperties)
                    if not rw_2_dis.isUnresolved():
                        if len(self.del_rewritings) > 0:
                            res = rw_2_dis.removePropertiesFrom(self.del_rewritings, True)
                        else:
                            res = rw_2_dis
                        self.result.append(res)
            else:
                merged_morphism_ddd = dict()
                for k, v in data_match_morphism.items():
                    if k != "obj":
                        merged_morphism_ddd["@" + str(k)] = v
                forAllProperties = None if not self.inherit_properties_from else {expansion.arg if isinstance(expansion, FNot) else expansion: self.original}
                rw_2_dis = expansion.replaceWith(merged_morphism_ddd, True, forAllProperties=forAllProperties)
                if not rw_2_dis.isUnresolved():
                    if len(self.del_rewritings) > 0:
                        res = rw_2_dis.removePropertiesFrom(self.del_rewritings, True)
                    else:
                        res = rw_2_dis
                    self.result.append(res)
                    # expansion.replaceWith(merged_morphism_ddd, True, forAllProperties=forAllProperties)

    def do_replacement_match_rec(self, i, d, fugitive_init, rw_2):
        """
        This method implements the rewriting step by replacing the terms
        of a previously matched component by substituting some of the
        variables

        :param i:               Index associated to the morphisms to be used for instantiation
        :param d: dictionary    Dictionary containing the variable instantiation also associated with a
                                specific matched object, through which we are going to determine
                                how to possibly rewrite the thing given the things were matched
        :param fugitive_init:   Keeping track across mutual rewritings of which components of the original
                                expression ended up in representing the final rewriting
        :param rw_2:            Expression that needs to be rewritten after expansion
        """
        if i < 0: # If I applied all the morphisms, then I can attempt at rewriting
            ## 9. Determining the association between the original data-matched elements, and
            ## the same sub-expression after rewriting. This is required to fall-back to the non-
            ## matched part under the circumstantces that the update over certain sub-expressions
            ## was not effective, thus requiring to fall-back to the original data without
            ## any kind of rewriting
            TCL = transitive_closure(fugitive_init.items())
            ## Determining actually all the sub-expressions associated to the final rewritten object
            DST = set(rw_2.collectIds())
            ## Determining which objects from the origin and the end actually appear in the matches thorugh
            ## the transitive closure and matching
            DST = {x[0] for x in TCL}.intersection(DST)
            ORIG = {x[1] for x in TCL}.intersection(self.ORIG)
            finalReplacement = dict({object_magic(x[0]): object_magic(x[1]) for x in TCL if
                                     (x[0] in DST and object_magic(x[0]).isUnresolved()) and x[1] in ORIG})
            forAllProperties = None
            if self.inherit_properties_from:
                forAllProperties = dict({object_magic(x[0]): object_magic(x[1]) for x in TCL if
                                         (x[0] in DST and (not object_magic(x[0]).isUnresolved())) and x[1] in ORIG})
                if len(forAllProperties) == 0:
                    forAllProperties = None
            if len(finalReplacement) > 0:
                rw_2 = rw_2.replaceWith(finalReplacement, forAllProperties=forAllProperties)
            if not rw_2.isOntoUnmatched():
                if len(self.del_rewritings)>0:
                    res = rw_2.removePropertiesFrom(self.del_rewritings, True)
                else:
                    res = rw_2
                self.result.append(res)
        else:
            from LaSSI.Parmenides.TBox.SentenceMatch import structure_dictionary
            dd = structure_dictionary(d)
            ## 4. For each replacement record m
            data_match_morphism = dd.to_dict('records')[i]
            ## 5. Using this to instantiate the parametrized ontology query using the matched data
            QInst = Parmenides.instantiate_query_with_map(self.onto_query, data_match_morphism)
            Ly = list(self.p.multiple_queries(QInst))
            if len(Ly) > 0:  ## Determining the match only if we have one element
                for ontology_navigation_morphism in Ly:
                    merged_morphism = data_match_morphism | ontology_navigation_morphism
                    # 7. Determining the rewriting transformation for each element of interest
                    merged_morphism_ddd = dict()
                    d_local = copy.deepcopy(d)
                    local_fugitive = copy.deepcopy(fugitive_init)

                    for k, v in merged_morphism.items():
                        if k != "obj":
                            merged_morphism_ddd["@" + str(k)] = v
                        else:
                            merged_morphism_ddd[k] = v
                    # 8. Partially instantiating the matched data to be rewritten
                    rw_2_dis = rw_2.replaceWith(merged_morphism_ddd, True, d_local, fugitive=local_fugitive)
                    self.do_replacement_match_rec(i - 1, d_local, local_fugitive, rw_2_dis)
            else:
                self.do_replacement_match_rec(i - 1, d, fugitive_init, rw_2)

def do_match(datum, toMatch, onto_query, p, replacement_map,
             value_invention=None, sentence_transformations:List[PostProcessingOperations]=None, where_cond=None):
    from collections import defaultdict
    from LaSSI.Parmenides.TBox.SentenceMatch import structure_dictionary
    if sentence_transformations is None:
        sentence_transformations = []
    # else:
    #     if value_invention is not None:
    #         raise ValueError("ERROR: if I have a value invention, then it makes no sense to also have a sentence transformation")
    d_orig = defaultdict(list)
    fugitive_init = dict()
    # for x in match(formula, q):
    # print(formula)

    ## Preparation:
    if (value_invention is None and
            len(sentence_transformations)>0 and
            any(map(lambda x: isinstance(x, AddPropertyFromResult), sentence_transformations))):
        ## Generation. Still, if I want to then add data, I might still perform
        ## value invention in some sense. So, I might boild down this to the latter
        value_invention = copy.deepcopy(datum)

    if value_invention is not None:
        ## Update
        properties = dict(datum.getProperties())
        for sentence in sentence_transformations:
            if isinstance(sentence, AddPropertyFromResult):
                if sentence.ofField in sentence:
                    d_orig[sentence.ofField] = d_orig[sentence.ofField] + (sentence.toAddInFields,)
                else:
                    d_orig[sentence.ofField] = tuple([sentence.toAddInFields])

    ## 0. Preserving the original ids
    ORIG = set(datum.collectIds())
    ## 1. Matching the query with the data
    rw_1 = datum.matchWith(toMatch, d_orig, None, fugitive_init)
    if not rw_1.matched:
        return []
    matched = list(map(object_magic, set(fugitive_init.values())))
    if len(matched) == 0:
        return []
    if value_invention is None:
        # assert len(d_orig) == 0 ## ASSERZIONE: QQQ
        ## 2. Within the data, I'm applying the substitution required by the matching and rewriting semantics
        dtr = copy.deepcopy(d_orig) ## DA RIAGGIUNGERE SE NON VALE QQQ
        # dtr = None ## DA RIMUOVERE SE NON VALE QQQ
        rw_init = rw_1.replaceWith(replacement_map, d=dtr, fugitive=fugitive_init)
        struct_dict_dd = structure_dictionary(dtr)
        # matched = filter(lambda x: x.matched, map(object_magic, rw_init.collectIds()))
    else:
        struct_dict_dd = structure_dictionary(d_orig)
        # matched = filter(lambda x: x.matched, map(object_magic, rw_1.collectIds()))
    # matched = list(matched)

    if (where_cond is not None) and len(where_cond) > 0:
        # matched = list(map(object_magic, set(fugitive_init.values())))
        if len(matched) == 0:
            return []
        for obj in matched:
            for matchQ in where_cond:
                if not boolean_simple_data_match(obj, matchQ):
                    return []

    ## 3. Defining the correspondences across the same variable object and across matchings
    N = len(struct_dict_dd)
    qRec = DoMatchRec(onto_query, p, ORIG, sentence_transformations, datum)

    if value_invention is None:
        qRec.do_replacement_match_rec(N - 1, dtr, fugitive_init, rw_init)
    else:
        qRec.do_expansion_match_iterative(N, struct_dict_dd, value_invention)
    return qRec.result

def do_actual_match(datum, Q, g:Parmenides):
    if Q is None:
        print("WARNING: Q IS NONE!")
        return datum
    elif datum is None:
        return datum
    elif isinstance(Q, str):
        return do_actual_match(datum, parse_query(Q), g)
    else:
        assert hasattr(Q, "cc")
        assert Q.cc is not None
        from LaSSI.Parmenides.TBox.language.TBoxParse import RewriteOperation
        if isinstance(Q, RewriteOperation):
            return do_match(datum, Q.match, Q.cc.ontology_query, g, Q.cc.replacement_pair, Q.rewrite, Q.cc.operations, Q.cc.where_theta)
        elif isinstance(Q, UpdateOperation):
            return do_match(datum, Q.formula, Q.cc.ontology_query, g, Q.cc.replacement_pair, None, Q.cc.operations, Q.cc.where_theta)
        else:
            raise ValueError("ERROR: Unexpected value of Q, "+str(Q))


class DoExpand:
    def __init__(self, parmenides_file:str, tbox_file_impl:str, tbox_file_eq:str):
        from LaSSI.Parmenides.TBox.language.TBoxParse import load_tbox_rules
        print("Loading TBox Rules...")
        self.q_list_impl = load_tbox_rules(tbox_file_impl)
        self.q_list_eq = load_tbox_rules(tbox_file_eq)
        print("Loading ABox Data...")
        self.g = Parmenides(parmenides_file)

    def __call__(self, formula:Formula, isImpl:bool):
        stack = [formula]
        visited = set()
        while len(stack)>0:
            f = stack.pop()
            if f in visited:
                continue
            visited.add(f)
            for Q in (self.q_list_impl if isImpl else self.q_list_eq):
                for f2 in do_actual_match(f, Q, self.g):
                    if f2 not in visited:
                        stack.append(f2)
        # if len(visited)>1:
        visited.remove(formula)
        return visited




q1 = """        {
        "args": [{
        "rel": "have",
        "src": {
            "name": "city center",
            "type": "None",
            "specification": "",
            "cop": null,
            "meta": "FVariable"
        },
        "dst": {
            "name": "XXXXX",
            "type": "JJ",
            "specification": null,
            "cop": null,
            "meta": "FVariable"
        },
        "score": 1.0,
        "properties": {
            "DATE": [
                {
                    "name": "Saturdays",
                    "type": "DATE",
                    "specification": null,
                    "cop": null,
                    "meta": "FVariable"
                }
            ]
        },
        "meta": "FBinaryPredicate"
        },{
        "rel": "have",
        "src": {
            "name": "city center",
            "type": "None",
            "specification": "",
            "cop": null,
            "meta": "FVariable"
        },
        "dst": {
            "name": "YYYYY",
            "type": "JJ",
            "specification": null,
            "cop": null,
            "meta": "FVariable"
        },
        "score": 1.0,
        "properties": {
            "DATE": [
                {
                    "name": "Saturdays",
                    "type": "DATE",
                    "specification": null,
                    "cop": null,
                    "meta": "FVariable"
                }
            ]
        },
        "meta": "FBinaryPredicate"
        }],
        "meta": "FAnd"
    }"""

q1bis = """        {
        "args": [{
        "rel": "have",
        "src": {
            "name": "city center",
            "type": "None",
            "specification": "",
            "cop": null,
            "meta": "FVariable"
        },
        "dst": {
            "name": "busy",
            "type": "JJ",
            "specification": null,
            "cop": null,
            "meta": "FVariable"
        },
        "score": 1.0,
        "properties": {
            "DATE": [
                {
                    "name": "Saturdays",
                    "type": "DATE",
                    "specification": null,
                    "cop": null,
                    "meta": "FVariable"
                }
            ]
        },
        "meta": "FBinaryPredicate"
        },{
        "rel": "have",
        "src": {
            "name": "city",
            "type": "None",
            "specification": "",
            "cop": null,
            "meta": "FVariable"
        },
        "dst": {
            "name": "cicicicia",
            "type": "JJ",
            "specification": null,
            "cop": null,
            "meta": "FVariable"
        },
        "score": 1.0,
        "properties": {
            "DATE": [
                {
                    "name": "Saturdays",
                    "type": "DATE",
                    "specification": null,
                    "cop": null,
                    "meta": "FVariable"
                }
            ]
        },
        "meta": "FBinaryPredicate"
        }],
        "meta": "FAnd"
    }"""

def query1(qstr):
        # with open("/home/giacomo/Scaricati/newcastle_sentences.txt_logical_rewriting.json", "r") as f:
        g = Parmenides("/home/giacomo/projects/similarity-pipeline/submodules/news-crawler/Parmenides/turtle.ttl")
        list_json = json.loads(qstr)
        formula = formula_from_dict(list_json)
        qq = FBinaryPredicate("have", make_variable("x"), make_variable("y"), 1.0, frozenset())
        onto_Q = ["@x", "partOf", "^z"]
        replacement_map = {"@y": "@z"}
        do_match(formula, qq, onto_Q, g, replacement_map)

def query2():
    g = Parmenides("/home/giacomo/projects/similarity-pipeline/submodules/news-crawler/Parmenides/turtle.ttl")
    formula = FBinaryPredicate("have", make_name("flow"), make_name("Chyraa-Khoor"), 1.0, frozenset())
    qq = FBinaryPredicate("have", make_variable("x"), make_variable("y"), 1.0, frozenset())
    onto_Q = ["@x", "^y"]
    replacement_map = dict()
    do_match(formula, qq, onto_Q, g, replacement_map)

def query3():
    g = Parmenides("/home/giacomo/projects/similarity-pipeline/submodules/news-crawler/Parmenides/turtle.ttl")
    formula = FBinaryPredicate("have", make_name("flow"), make_name("Chyraa-Khoor"), 1.0, frozenset())
    qq = FBinaryPredicate("have", make_variable("x"), make_variable("z"), 1.0, frozenset())
    rewriting = FUnaryPredicate("be", make_variable("y"), 1.0, frozenset())
    onto_Q = ["@x", "^y"]
    replacement_map = dict()
    for x in do_match(formula, qq, onto_Q, g, replacement_map, rewriting):
        print(x)

if __name__ == "__main__":
    de = DoExpand("/home/giacomo/projects/similarity-pipeline/submodules/news-crawler/Parmenides/turtle.ttl",
             "/home/giacomo/projects/similarity-pipeline/submodules/news-crawler/Parmenides/TBox/file.txt",
             "/home/giacomo/projects/similarity-pipeline/submodules/news-crawler/Parmenides/TBox/file_eq.txt")
    with open("/home/giacomo/projects/similarity-pipeline/submodules/news-crawler/sentences/newcastle_sentences2.txt_logical_rewriting.json", "r") as f:
        list_json = json.load(f)
        list_json = formula_from_dict(list_json)
        s1 = list_json[3]
        # s1 = FUnaryPredicate("be", FVariable("busy", "JJ", None, None), 1.0, frozenset({"GPE": tuple([FVariable("Newcastle",None,None)])}.items()))
        # for x in list_json:
        s1 = de(s1, True)
        print(list(map(str, s1)))
        # s2 = de(list_json[7])
        # print(list_json[7])
    #query1(q1bis) #q1bis
