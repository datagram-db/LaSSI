__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import copy
from collections import defaultdict
from copy import deepcopy

from LaSSI.ner.node_functions import create_props_for_singleton

# from scipy._lib.array_api_compat.array_api_compat import numpy

from LaSSI.structures.internal_graph.EntityRelationship import NodeEntryPoint, Singleton, SetOfSingletons, Grouping
from LaSSI.structures.extended_fol.Sentences import FNot, FOr, FAnd, FUnaryPredicate, FVariable, FBinaryPredicate, Formula, \
    prune_from_cop

bogus_dst = FVariable(name="there", type="non_verb", specification=None, cop=None, id=-1)
bogus_src = {"it"}
discard_properties = {"end", "lemma", "begin", "kernel", "expl", "pos", "root", "common", "number", "adv"}
relative_pronouns = {"which","that", "who", "whom" }
interrogative_pronouns = {"what", "which", "who", "whom", "whose"}
demonstrative_pronouns = {"this", "these", "that", "those"}

def property_write(key, val: NodeEntryPoint) -> str:
    value = '""'
    # if isinstance(val, Singleton):
    return f'{key} : {value}'


def make_cop(entity) -> FVariable:
    if entity is None:
        return None
    elif isinstance(entity, str):
        return FVariable(name=entity, type="JJ", specification=None, cop=None, id=-1)
    else:
        return make_arg(entity[0])  # TODO: Will we ever have more than one cop for a given entity?


def make_arg(entity):
    if entity is None:
        return None
    elif isinstance(entity, FVariable) or isinstance(entity, FBinaryPredicate) or isinstance(entity, FUnaryPredicate) or isinstance(entity, FNot) or isinstance(entity, FAnd) or isinstance(entity, FOr):
        return entity
    elif hasattr(entity, "kernel") and entity.kernel is not None:
        return rewrite_kernels(entity)
    props = entity if isinstance(entity, dict) else entity.get_props()
    specifiaction = None
    if "extra" in props and props["extra"] is not None and (
                (not isinstance(props["extra"], tuple)) or len(props["extra"]) == 1):
        specifiaction = make_arg(props.pop("extra")[0])
    coplist = []
    cop =  None
    if "cop" in props:
        cop = make_cop(props.pop("cop"))
    if cop is None:
        if "JJ" in props:
            cop = make_cop(props.pop("JJ"))
        for k in props:
            if k.endswith("mod"):
                coplist.append(make_cop(props[k]))
    if len(coplist) == 1:
        cop = coplist[0]
    elif len(coplist) > 1:
        cop = tuple(coplist)
    named_entity = props.pop("named_entity", None) if isinstance(entity, dict) else entity.get_name()  # TODO: Is this okay for getting the name of SetOfSingletons?
    type = props.pop("type", None) if isinstance(entity, dict) else entity.type
    if type != "GPE":
        named_entity = named_entity.lower()
    props2 = dict()
    for k, v in props.items():
        if k not in discard_properties and k not in {} and ((not isinstance(v, str)) or len(v)==0):
            props2[k] = make_arg(v) if isinstance(v, Singleton) else v
    return FVariable(name=named_entity, type=type, specification=specifiaction, cop=cop, id=entity.id, properties=frozenset(props2.items()))


def make_and(entities):
    return FAnd(args=tuple(entities))


def make_or(entities):
    return FOr(args=tuple(entities))


def make_not(param):
    return FNot(arg=param)


def make_unary(rel, dst, score, prop):
    if rel == "be":  # TODO: generalise
        if dst is not None and dst.cop is not None:  # TODO: generalise
            return make_binary("have", prune_from_cop(dst), dst.cop, score, prop)
        if dst is not None and (
                dst.type == "DATE" or dst.type == "GPE" or dst.type == "LOC") and dst.cop is not None:  # TODO: generalise
            if dst.type not in prop:
                prop[dst.type] = []
            prop[dst.type].append(dst)
            return make_unary(rel, dst.cop, score, prop)
    if "non_verb" in prop:
        prop.pop("non_verb")
    s = set(prop.keys())
    for x in s:
        if isinstance(prop[x], list):
            if len(prop[x]) == 0:
                prop[x] = prop[x][0]
            else:
                prop[x] = tuple(prop[x])
    return FUnaryPredicate(rel=rel, arg=dst, score=score, properties=frozenset(prop.items()))


def make_binary(rel, src, dst, score, prop):
    if (rel == "be" and (dst is None or (not isinstance(dst, FBinaryPredicate) and not isinstance(dst, FUnaryPredicate) and dst.type == "existential"))) or dst is None:
        return make_unary(rel, src, score, prop)
    if rel == "have":  # TODO: generalise
        if src is not None and (
                src.type == "DATE" or src.type == "GPE" or src.type == "LOC") and src.cop is None:  # TODO: generalise
            if src.type not in prop:
                prop[src.type] = []
            prop[src.type].append(src)
            return make_unary("be", dst, score, prop)
    if "non_verb" in prop:
        prop.pop("non_verb")
    s = set(prop.keys())
    for x in s:
        if isinstance(prop[x], list):
            if len(prop[x]) == 0:
                prop[x] = prop[x][0]
            else:
                prop[x] = tuple(prop[x])
    return FBinaryPredicate(rel=rel, src=src, dst=dst, score=score, properties=frozenset(prop.items()))


def make_properties(p):
    result = defaultdict(set)
    if "not" in set(map(lambda x: x.lower(), p.keys())):
        for k in filter(lambda x: x.lower() == "not", p.keys()):
            for single_val in p[k]:
                if isinstance(single_val, SetOfSingletons):
                    assert len(single_val.entities) == 1
                    single_val = single_val.entities[0]
                assert isinstance(single_val, Singleton)
                type = single_val.type
                single_val = make_not(make_arg(single_val))
                result[type].add(single_val)

    for k, v in p.items():
        if str(k) == "\u2203" or str(k).lower() == "in" or str(k).lower() == "not":
            continue
        else:
            for single_val in v:
                tmp = make_arg(single_val)
                neg_tmp = make_not(tmp)
                if neg_tmp not in result[k]:
                    result[k].add(tmp)

    result2 = dict()
    for k, v in result.items():
        result2[k] = list(v)
    return result2  # dict(result2.items())


def make_prop(src, rel, negated, score, properties, dst):
    if (dst is not None):
        if isinstance(dst, SetOfSingletons):
            if dst.type == Grouping.AND:
                return make_and(map(lambda x: make_prop(src, rel, negated, score, properties, x), dst.entities))
            elif dst.type == Grouping.OR:
                return make_or(map(lambda x: make_prop(src, rel, negated, score, properties, x), dst.entities))
            elif dst.type == Grouping.NOT:
                return make_not(make_prop(src, rel, negated, score, properties, list(dst.entities)[0]))
            else:
                n = dst.type.name
                raise RuntimeError(f"Unknown source type: {n}")
        else:
            if negated:
                return make_not(make_prop(src, rel, False, score, properties, dst))
            else:
                result = None
                p = dict()
                foundSingleton = Grouping.NONE
                argument = None
                forKey = None
                for k, v in properties.items():
                    if k not in discard_properties:
                        if isinstance(v, list) or isinstance(v, tuple):
                            j = []
                            for x in v:
                                if (foundSingleton == Grouping.NONE) and isinstance(x, SetOfSingletons) and (
                                        x.type != Grouping.NOT):
                                    assert len(x.entities) == 1
                                    foundSingleton = x.type
                                    argument = x.entities[0]
                                    forKey = k
                                elif isinstance(x, str) or isinstance(x, Formula):
                                    j.append(x)
                                elif isinstance(x, Singleton):
                                    j.append(make_arg(x))
                                elif isinstance(x, SetOfSingletons):
                                    if x.type == Grouping.NOT:
                                        if isinstance(x.entities[0], Singleton):
                                            j.append(make_not(make_arg(x.entities[0])))
                                        else:
                                            raise RuntimeError(f"Unknown argument type: {x}")
                                    else:
                                        raise RuntimeError(f"Unknown argument type: {x}")
                                else:
                                    raise RuntimeError(f"Unknown argument type: {x}")
                            if len(j) > 0:
                                p[k] = tuple(j)
                        else:
                            p[k] = v
                src = make_arg(src)
                p["src"] = []
                p["dst"] = []
                if src is not None:
                    p["src"].append(src)
                dst = make_arg(dst)
                if dst is not None:
                    p["dst"].append(dst)
                prop = make_properties(p)
                del prop["src"]
                del prop["dst"]
                if src.name.lower() in bogus_src and rel.lower() == "be":
                    if src.cop is None:
                        result = make_unary(rel, dst, score, prop)
                    else:
                        src = src.cop
                if result is None:
                    if dst == bogus_dst:
                        result = make_unary(rel, src, score, prop)
                    else:
                        result = make_binary(rel, src, dst, score, prop)
                if foundSingleton == Grouping.NONE:
                    return result
                else:
                    dcp = copy.deepcopy(prop)
                    dcp[forKey] = (argument,)
                    if foundSingleton == Grouping.AND:
                        return make_and([result, make_prop(src, rel, negated, score, dcp, dst)])
                    elif foundSingleton == Grouping.OR:
                        return make_or([result, make_prop(src, rel, negated, score, dcp, dst)])
                    else:
                        n = dst.type.name
                        raise RuntimeError(f"Unknown source type: {n}")
    else:
        if negated:
            return make_not(make_prop(src, rel, False, score, properties, None))
        else:
            p = dict()
            for k, v in properties.items():
                if k not in discard_properties:
                    p[k] = v
            p["src"] = []
            p["dst"] = []
            if src is not None:
                p["src"].append(src)
            prop = make_properties(p)
            del prop["src"]
            if "dst" in prop:
                del prop["dst"]
            return make_unary(rel, make_arg(src), score, prop)


def src_make_prop(src, rel, negated, score, properties, dst):
    if src is not None:
        if isinstance(src, SetOfSingletons):
            if src.type == Grouping.AND:
                return make_and(map(lambda x: src_make_prop(x, rel, negated, score, properties, dst), src.entities))
            elif src.type == Grouping.OR:
                return make_or(map(lambda x: src_make_prop(x, rel, negated, score, properties, dst), src.entities))
            elif src.type == Grouping.NOT:
                return make_not(src_make_prop(list(src.entities)[0], rel, negated, score, properties, dst))
            elif src.type == Grouping.NEITHER:
                return make_not(make_and(map(lambda x: src_make_prop(x, rel, negated, score, properties, dst), src.entities)))
            else:
                n = src.type.name
                raise RuntimeError(f"Unknown source type: {n}")
        else:
            return make_prop(src, rel, negated, score, properties, dst)
    else:
        return make_prop(None, rel, negated, score, properties, dst)


def rewrite_kernels(obj: Singleton) -> Formula:
    main_pop = obj.kernel
    properties = obj.properties
    rel = main_pop.edgeLabel.named_entity if main_pop.edgeLabel is not None else "None"  # TODO: Do we want string "None" or None? (e.g. when we have no verb?)
    negated = main_pop.isNegated

    score = main_pop.edgeLabel.confidence if main_pop.edgeLabel is not None else 1
    if main_pop.source is not None:
        score *= main_pop.source.confidence
    if main_pop.target is not None:
        score *= main_pop.target.confidence

    return src_make_prop(main_pop.source, rel, negated, score, dict(properties), main_pop.target)
