__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2024, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import copy
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple, Dict, Union



class Formula:

    def __str__(self):
        return "Formula(?)"

    def isOntoUnmatched(self):
        return False

    def matchWith(self, f, d, ancestor, fugitive):
        return self

    def isUnresolved(self):
        return False

    def collectIds(self):
        yield id(self)

    def replaceWith(self,map, onObjectId=False, d=None, fugitive=None, forAllProperties=None):
        return self

    def getFlattenedProperties(self):
        return frozenset(set())

    def getProperties(self):
        return frozenset(set())

    def updateWithProperties(self, toFrozenSet):
        return self

    def removePropertiesFrom(self, coll, onMatch=False):
        return self

def replace_string(s:str, map:dict):
    if s is None or map is None or len(map) == 0:
        return s
    elif s.startswith("@") and s in map:
        return map[s]
    else:
        return s

def replace_formula(s:Formula, map:dict, onObjectId, d, fugitive, forAllProperties):
    if s is None or map is None or len(map) == 0:
        return s
    else:
        return s.replaceWith(map, onObjectId, d, fugitive, forAllProperties)

def replace_frozenset(fs: frozenset, map: dict, onObjectId, d, fugitive, forAllProperties):
    if fs is None or map is None or len(map) == 0:
        return fs
    else:
        result = dict()
        for x,y in fs:
            result[x] = tuple([query_replace(z, map, onObjectId, d, fugitive, forAllProperties) for z in y])
        return frozenset(result.items())

def query_replace(x, map: dict, id1, d, fugitive, forAllProperties):
    if map is None or len(map) == 0:
        return x
    elif x in map:
        return map[x]
    elif isinstance(x, str):
        return replace_string(x, map)
    elif isinstance(x, Formula):
        return replace_formula(x, map, id1, d, fugitive, forAllProperties)
    elif isinstance(x, frozenset):
        return replace_frozenset(x, map, id1, d, fugitive, forAllProperties)
    else:
        raise ValueError("Unsupported type for: "+str(x))

def match_field(self_name, other_name, d, orig):
    if other_name is None:
        return True, self_name
    elif other_name.startswith("@"):
        d[other_name[1:]].append((self_name, orig))
        return True, other_name##+":{"+self_name+"}"
    else:
        return self_name == other_name, self_name

def match_frozen_set(self_fs, match_fs, d, ancestor, fugitive):
    final_result = dict()
    if self_fs is None or match_fs is None:
        return self_fs
    if isinstance(match_fs, frozenset):
        if len(match_fs) == 0:
            return self_fs
        tmp_matchfs = dict(match_fs)
        for x,y in self_fs:
            if x not in tmp_matchfs:
                ## If no specification is provided for the match, I just keep it as such
                final_result[x] = y
            else:
                ## Otherwise, I need to guarantee that the match has one single possible interpretation
                assert len(tmp_matchfs[x]) == 1
                final_result[x] = tuple(map(lambda z: z.matchWith(list(tmp_matchfs[x])[0], d, ancestor, fugitive), y))
    elif isinstance(match_fs, Formula):
        for x, y in self_fs:
            final_result[x] = tuple(map(lambda z: z.matchWith(match_fs, d, ancestor, fugitive), y))
    return frozenset(final_result.items())

def match_formula(formula, match_fs, d, ancestor, fugitive):
    if formula is None or match_fs is None:
        return formula
    else:
        return formula.matchWith(match_fs, d, ancestor, fugitive)


def retaliate_dd(dd, prev, curr, fugitive):
    if dd is None:
        return
    for key in dd:
        L = []
        for (k,v) in dd[key]:
            if v == prev or v == id(prev):
                if fugitive is not None:
                    i = prev if isinstance(prev, int) else id(prev)
                    fugitive[id(curr)] = i
                L.append((k, id(curr)))
            else:
                L.append((k, v))
        dd[key] = L

def is_string_unresolved(s):
    return s is not None and isinstance(s, str) and (s.startswith("@"))

def is_formula_unresolved(f):
    return f is not None and f.isUnresolved()

def is_frozenset_unresolved(fs):
    if fs is None :
        return False
    else:
        for k, v in fs:
            for x in v:
                if is_formula_unresolved(x): return True
        return False

def isStringUnresolved(s):
    if s is None:
        return False
    elif s.startswith("@"):
        return True
    else:
        return False

@dataclass(order=True, frozen=True, eq=True)
class FVariable(Formula):
    name: str
    type: str
    specification: str #extra
    cop: Formula
    meta: str = field(default_factory=lambda : "FVariable")
    matched: bool = field(default_factory=lambda : False)

    def __str__(self):
        s = self.name
        if s is None:
            s = "?"
        if self.specification is not None and len(self.specification) > 0:
            s += (" [of] "+ self.specification)
        if self.cop is not None:
            s += (" JJ:" + str(self.cop))
        return s

    def isOntoUnmatched(self):
        if isStringUnresolved(self.name): return True
        if isStringUnresolved(self.type): return True
        if isStringUnresolved(self.specification): return True
        if self.cop is not None and self.cop.isOntoUnmatched(): return True
        return False

    def getFlattenedProperties(self):
        if self.cop is not None:
            return self.cop.getFlattenedProperties()
        return super().getFlattenedProperties()

    def removePropertiesFrom(self, coll, onMatch=False):
        return FVariable(self.name, self.type, self.specification, self.cop.removePropertiesFrom(coll, onMatch) if self.cop is not None else None, matched=self.matched)

    def isUnresolved(self):
        if self.matched: return False
        if is_string_unresolved(self.name): return True
        if is_string_unresolved(self.type): return True
        if is_string_unresolved(self.specification): return True
        if is_formula_unresolved(self.cop): return True
        return False

    def collectIds(self):
        yield id(self)
        if self.cop is not None:
            yield from self.cop.collectIds()

    def matchWith(self, f, d, ancestor, fugitive):
        isAncestorNone = ancestor is None
        if self.meta == f.meta:
            if ancestor is None:
                ancestor = self
            dd = copy.deepcopy(d)
            name_test, name_value = match_field(self.name, f.name, dd, ancestor)
            type_test, type_value = match_field(self.type, f.type, dd, ancestor)
            specification, spec_value = match_field(self.specification, f.specification, dd, ancestor)
            if name_test and type_test and specification:
                for k,y in dd.items():
                    d[k] = y
                cop = match_formula(self.cop, f.cop, d, ancestor, fugitive)
                var = FVariable(name_value, type_value, spec_value, cop=cop, matched=True)
                if isAncestorNone:
                    retaliate_dd(d, self, var, fugitive)
                return var
        if isAncestorNone:
            ancestor = None
        # cop = match_formula(self.cop, f, d, ancestor, fugitive)
        return FVariable(self.name, self.type, self.specification, cop=self.cop)

    def updateWithProperties(self, toFrozenSet):
        return self

    def replaceWith(self, map, onObjectId=False, d=None, fugitive=None, forAllProperties=None):
        if map is None or len(map) == 0:
            return self
        if self in map:
            return map[self]
        if onObjectId and "obj" in map and id(self) != map["obj"]:
            name = self.name
            type = self.type
        else:
            onObjectId = False  # Forcing always the rewrite from now on
            name = replace_string(self.name, map)
            type = replace_string(self.type, map)
        specification = replace_string(self.specification, map)
        cop = replace_formula(self.cop, map, onObjectId, d, fugitive, forAllProperties)
        var = FVariable(name=name, type=type, specification=specification, cop=cop, matched=self.matched)
        retaliate_dd(d, self, var, fugitive)
        return var

    def strippedByType(self):
        return FVariable(name=self.name, type="TODO", specification=self.specification, cop=stripArg(self.cop), matched=self.matched)

def make_variable(var, type=None, spec=None, cop:Formula=None):
    return FVariable(name="@"+str(var), type=type, specification=spec, cop=cop)

def make_name(var, type=None, spec=None, cop:Formula=None):
    return FVariable(name=str(var) if var is not None else None, type=type, specification=spec, cop=cop)

def stripArg(x):
    if x is None:
        return x
    else:
        return x.strippedByType()

def prune_from_cop(var:FVariable):
    return FVariable(name=var.name, type=var.type, specification=var.specification, cop=None)

def stripFrozenProperties(coll):
    d = dict()
    for x,y in coll:
        if isinstance(y,Formula):
            d[x] = y.strippedByType()
        else:
            d[x] = y
    return frozenset(d.items())

def remove_properties_from(coll, var):
        return frozenset({x:y for x, y in coll if x not in var}.items())


class PostProcessingOperations():
    pass

@dataclass(order=True, frozen=True, eq=True)
class RemovePropertiesFromResult(PostProcessingOperations):
    ofField: str
    meta_op: str = field(default_factory=lambda: "RemovePropertiesFromResult")

@dataclass(order=True, frozen=True, eq=True)
class AddPropertyFromResult(PostProcessingOperations):
    ofField: str
    toAddInFields: Formula
    meta_op: str = field(default_factory=lambda: "RemovePropertiesFromResult")

@dataclass(order=True, frozen=True, eq=True)
class InheritProperties(PostProcessingOperations):
    meta_op: str = field(default_factory=lambda: "InheritProperties")

def applyOperation(formula:Formula, operation:PostProcessingOperations):
    if not isinstance(operation, PostProcessingOperations):
        raise ValueError("ERROR: operation shouild be a PostProcessingOperations")
    if isinstance(operation, RemovePropertiesFromResult):
        return formula.removePropertiesFrom(operation.ofField)
    elif isinstance(operation, AddPropertyFromResult):
        properties = frozenset()
        if hasattr(formula, "properties"):
            properties = dict()
            wasFound = False
            for x,y in formula.properties:
                if x == operation.ofField:
                    wasFound = True
                    tmp = set()
                    tmp.add(operation.toAddInFields)
                    properties[x] = tuple(tmp)
                else:
                    properties[x] = y
            if not wasFound:
                properties[operation.ofField] = tuple([operation.toAddInFields])
            properties = frozenset(properties.items())
        return formula.updateWithProperties(properties)
    else:
        raise ValueError("ERROR: unexpected operation case!")



@dataclass(order=True, frozen=True, eq=True)
class FUnaryPredicate(Formula):
    rel: str
    arg: Formula
    score: float
    properties: frozenset
    meta: str = field(default_factory=lambda : "FUnaryPredicate")
    matched: bool = field(default_factory=lambda : False)

    def __str__(self):
        s = self.rel
        if s is None:
            s = "?"
        if self.arg is not None:
            s += ("(" + str(self.arg) + ")")
        if self.properties is not None:
            s += json.dumps({k:[str(x) for x in v] for k,v in self.properties})
        return s

    def isOntoUnmatched(self):
        if isStringUnresolved(self.rel): return True
        if self.arg is not None and self.arg.isOntoUnmatched(): return True
        for x,y in self.properties:
            for z in y:
                if z is not None and z.isOntoUnmatched(): return True
        return False

    def getFlattenedProperties(self):
        props = defaultdict(set)
        if self.arg is not None:
            for x, y in self.arg.getFlattenedProperties():
                props[x] = set(y)
        if self.properties is not None:
            for x, y in self.properties:
                props[x] = props[x].union(set(y))
        return frozenset({x: tuple(y) for x,y in props.items()}.items())

    def getProperties(self):
        return self.properties

    def removePropertiesFrom(self, coll, onMatch=False):
        if (not onMatch) or (self.matched):
            return FUnaryPredicate(self.rel, self.arg.removePropertiesFrom(coll, onMatch) if self.arg is not None else None, self.score, remove_properties_from(self.properties, coll), matched=self.matched)
        else:
            return FUnaryPredicate(self.rel, self.arg.removePropertiesFrom(coll, onMatch) if self.arg is not None else None, self.score, self.properties, matched=self.matched)


    def isUnresolved(self):
        if self.matched: return False
        if is_formula_unresolved(self.arg): return True
        if is_frozenset_unresolved(self.properties): return True
        return is_string_unresolved(self.rel)

    def collectIds(self):
        yield id(self)
        if self.arg is not None:
            yield from self.arg.collectIds()
        if self.properties is not None:
            for k,v in self.properties:
                for y in v:
                    if (y is not None) and (isinstance(y, Formula)):
                        yield from y.collectIds()

    def matchWith(self, f, d, ancestor, fugitive):
        isAncestorNone = ancestor is None
        if self.meta == f.meta:
            if ancestor is None:
                ancestor = self
            dd = copy.deepcopy(d)
            rel_test, rel_value = match_field(self.rel, f.rel, dd, ancestor)
            score = self.score
            if rel_test:
                for k,y in dd.items():
                    d[k] = y
                arg = match_formula(self.arg, f.arg, d, ancestor, fugitive)
                if arg is None or arg.matched:
                    var = FUnaryPredicate(rel_value, arg, score, match_frozen_set(self.properties, f.properties, d, ancestor, fugitive), matched=True)
                    if isAncestorNone:
                        retaliate_dd(d, self, var, fugitive)
                    return var
        if isAncestorNone:
            ancestor = None
        arg = match_formula(self.arg, f, d, ancestor, fugitive)
        return FUnaryPredicate(self.rel, arg, self.score, match_frozen_set(self.properties, f, d, ancestor, fugitive))

    def updateWithProperties(self, toFrozenSet):
        r = {k:tuple(v) for k,v in toFrozenSet.items()}
        return FUnaryPredicate(rel=self.rel, arg=self.arg, score=self.score, properties=frozenset(r.items()), matched=self.matched)

    def replaceWith(self, map, onObjectId=False, d=None, fugitive=None, forAllProperties=None):
        if map is None or len(map) == 0:
            return self
        if self in map:
            return map[self]
        if onObjectId and "obj" in map and id(self) != map["obj"]:
            rel = self.rel
        else:
            onObjectId = False  # Forcing always the rewrite from now on
            rel = replace_string(self.rel, map)
        arg = replace_formula(self.arg, map, onObjectId, d, fugitive, forAllProperties)
        properties = replace_frozenset(self.properties, map, onObjectId, d, fugitive, forAllProperties)
        if forAllProperties is not None and self in forAllProperties:
            properties = {x:set(y) for x,y in properties}
            for x, y in forAllProperties[self].getFlattenedProperties():
                if x in properties:
                    properties[x] = properties[x]#.union(set(y))
                else:
                    properties[x] = set(y)
            properties = frozenset({x:tuple(y) for x,y in properties.items()}.items())
        var = FUnaryPredicate(rel=rel, arg=arg, score=self.score, properties=properties, matched=self.matched)
        retaliate_dd(d, self, var, fugitive)
        return var

    def semantic(self, d:Dict[Formula, bool]):
        assert self in d
        return d[self]

    def getAtoms(self):
        s =  {self}
        return s

    def strippedByType(self):
        return FUnaryPredicate(rel=self.rel, arg=stripArg(self.arg), score=self.score, properties=stripFrozenProperties(self.properties))


@dataclass(order=True, frozen=True, eq=True)
class FBinaryPredicate(Formula):
    rel:str
    src: Formula
    dst: Formula
    score: float
    properties: frozenset
    meta: str = field(default_factory=lambda : "FBinaryPredicate")
    matched: bool = field(default_factory=lambda : False)

    def __str__(self):
        s = self.rel
        if s is None:
            s = "?("
        else:
            s += "("
        if self.src is not None:
            s += (str(self.src) + ",")
        else:
            s += "?,"
        if self.dst is not None:
            s += (str(self.dst) + ")")
        else:
            s += "?)"
        if self.properties is not None:
            s += json.dumps({k:[str(x) for x in v] for k,v in self.properties})
        return s

    def isOntoUnmatched(self):
        if isStringUnresolved(self.rel): return True
        if self.src is not None and self.src.isOntoUnmatched(): return True
        if self.dst is not None and self.dst.isOntoUnmatched(): return True
        for x,y in self.properties:
            for z in y:
                if z is not None and z.isOntoUnmatched(): return True
        return False

    def removePropertiesFrom(self, coll, onMatch=False):
        if (not onMatch) or (self.matched):
            return FBinaryPredicate(self.rel,
                                   self.src.removePropertiesFrom(coll, onMatch) if self.src is not None else None,
                                   self.dst.removePropertiesFrom(coll, onMatch) if self.dst is not None else None,
                                   self.score, remove_properties_from(self.properties, coll), matched=self.matched)
        else:
            return FBinaryPredicate(self.rel,
                                   self.src.removePropertiesFrom(coll, onMatch) if self.src is not None else None,
                                   self.dst.removePropertiesFrom(coll, onMatch) if self.dst is not None else None,
                                   self.score, self.properties, matched=self.matched)

    def getFlattenedProperties(self):
        flattenedSuperSet = defaultdict(set)
        if self.src is not None:
            for x, y in self.src.getFlattenedProperties():
                if x not in flattenedSuperSet:
                    flattenedSuperSet[x] = set(y)
                else:
                    flattenedSuperSet[x] = flattenedSuperSet[x].union(set(y))
        if self.dst is not None:
            for x, y in self.dst.getFlattenedProperties():
                if x not in flattenedSuperSet:
                    flattenedSuperSet[x] = set(y)
                else:
                    flattenedSuperSet[x] = flattenedSuperSet[x].union(set(y))
        if self.properties is not None:
            for x, y in self.properties:
                flattenedSuperSet[x] = flattenedSuperSet[x].union(set(y))
        return frozenset({x:tuple(y) for x,y in flattenedSuperSet.items()}.items())


    def getProperties(self):
        return self.properties


    def isUnresolved(self):
        if self.matched: return False
        if is_formula_unresolved(self.src): return True
        if is_formula_unresolved(self.dst): return True
        if is_frozenset_unresolved(self.properties): return True
        return is_string_unresolved(self.rel)

    def collectIds(self):
        yield id(self)
        if self.src is not None:
            yield from self.src.collectIds()
        if self.dst is not None:
            yield from self.dst.collectIds()
        if self.properties is not None:
            for k,v in self.properties:
                for y in v:
                    if (y is not None) and (isinstance(y, Formula)):
                        yield from y.collectIds()

    def matchWith(self, f, d, ancestor, fugitive):
        isAncestorNone = ancestor is None
        if self.meta == f.meta:
            if ancestor is None:
                ancestor = self
            dd = copy.deepcopy(d)
            rel_test, rel_value = match_field(self.rel, f.rel, dd, ancestor)
            score = self.score
            if rel_test:
                for k,y in dd.items():
                    d[k] = y
                src = match_formula(self.src, f.src, d, ancestor, fugitive)
                dst = match_formula(self.dst, f.dst, d, ancestor, fugitive)
                if (src is None or src.matched) and (dst is None or dst.matched):
                    var = FBinaryPredicate(rel_value, src, dst, score, match_frozen_set(self.properties, f.properties, d, ancestor, fugitive), matched=True)
                    if isAncestorNone:
                        retaliate_dd(d, self, var, fugitive)
                    return var
        if isAncestorNone:
            ancestor = None
        src = match_formula(self.src, f, d, ancestor, fugitive)
        dst = match_formula(self.dst, f, d, ancestor, fugitive)
        return FBinaryPredicate(self.rel, src, dst, self.score, match_frozen_set(self.properties, f.properties, d, ancestor, fugitive))

    def updateWithProperties(self, toFrozenSet):
        r = {k:tuple(v) for k,v in toFrozenSet.items()}
        return FBinaryPredicate(rel=self.rel, src=self.src, dst=self.dst, score=self.score, properties=frozenset(r.items()), matched=self.matched)

    def replaceWith(self, map, onObjectId=False, d=None, fugitive=None, forAllProperties=None):
        if map is None or len(map) == 0:
            return self
        if self in map:
            return map[self]
        if onObjectId and "obj" in map and id(self) != map["obj"]:
            rel = self.rel
        else:
            onObjectId = False #Forcing always the rewrite from now on
            rel = replace_string(self.rel, map)
        src = replace_formula(self.src, map, onObjectId, d, fugitive, forAllProperties)
        dst = replace_formula(self.dst, map, onObjectId, d, fugitive, forAllProperties)
        properties = replace_frozenset(self.properties, map, onObjectId, d, fugitive, forAllProperties)
        if forAllProperties is not None and self in forAllProperties:
            properties = {x:set(y) for x,y in properties}
            for x, y in forAllProperties[self].getFlattenedProperties():
                if x in properties:
                    properties[x] = properties[x]#.union(set(y))
                else:
                    properties[x] = set(y)
            properties = frozenset({x:tuple(y) for x,y in properties.items()}.items())
        var = FBinaryPredicate(rel=rel, src=src, dst=dst, score=self.score, properties=properties, matched=self.matched)
        retaliate_dd(d, self, var, fugitive)
        return var

    def semantic(self, d:Dict[Formula, bool]):
        assert self in d
        return d[self]

    def getAtoms(self):
        s = {self}
        return s

    def strippedByType(self):
        return FBinaryPredicate(rel=self.rel, src=stripArg(self.src), dst=stripArg(self.dst), score=self.score, properties=stripFrozenProperties(self.properties))

@dataclass(order=True, frozen=True, eq=True)
class FAnd(Formula):
    args: Tuple[Formula]
    meta: str = field(default_factory=lambda : "FAnd")
    matched: bool = field(default_factory=lambda : False)

    def __str__(self):
        return " /\\ ".join(map(str, self.args))

    def isOntoUnmatched(self):
        return any(map(lambda x: x is not None and x.isOntoUnmatched(), self.args))

    def removePropertiesFrom(self, coll, onMatch=False):
        return FAnd(args=tuple([x.removePropertiesFrom(coll, onMatch) if x is not None else None for x in self.args]), matched=self.matched)

    def getFlattenedProperties(self):
        flattenedSuperSet = defaultdict(set)
        for arg in self.args:
            if arg is not None:
                for x,y in arg.getFlattenedProperties():
                    if x not in flattenedSuperSet:
                        flattenedSuperSet[x] = set(y)
                    else:
                        flattenedSuperSet[x] = flattenedSuperSet[x].union(set(y))
        return frozenset({x:tuple(y) for x,y in flattenedSuperSet.items()}.items())


    def updateWithProperties(self, toFrozenSet):
        return self

    def isUnresolved(self):
        if self.matched: return False
        return any(map(is_formula_unresolved, self.args))

    def collectIds(self):
        yield id(self)
        for arg in self.args:
            if arg is not None and isinstance(arg, Formula):
                yield from arg.collectIds()

    def matchWith(self, f, d, ancestor, fugitive):
        isAncestorNone = ancestor is None
        if self.meta == f.meta:
            if ancestor is None:
                ancestor = self
            L = []
            dd = copy.deepcopy(d)
            for x, y in zip(self.args, f.args):
                z = match_formula(x, y, dd, ancestor, fugitive)
                if z == x:
                    ancestor = None
                    break
                else:
                    L.append(z)
            if ancestor == self:
                for x, y in dd.items():
                    d[x] = y
                var = FAnd(args=tuple(L), matched=True)
                if isAncestorNone:
                    retaliate_dd(d, self, var, fugitive)
                return var
        if isAncestorNone:
            ancestor = None
        return FAnd(args=tuple(map(lambda x: match_formula(x, f, d, ancestor, fugitive), self.args)))

    def replaceWith(self, m, onObjectId=False, d=None, fugitive=None, forAllProperties=None):
        if m is None or len(m) == 0:
            return self
        if self in m:
            return m[self]
        if onObjectId and id(self) == m["obj"]:
            onObjectId = False
        var = FAnd(args=tuple(map(lambda x: replace_formula(x, m, onObjectId, d, fugitive, forAllProperties), self.args)), matched=self.matched)
        retaliate_dd(d, self, var, fugitive)
        return var

    def semantic(self, d:Dict[Formula, bool]):
        # assert self in d
        return min(map(lambda x: x.semantic(d), self.args))

    def getAtoms(self):
        s = set()
        for x in self.args:
            s = s.union(x.getAtoms())
        return s

    def strippedByType(self):
        return FAnd(args=tuple(map(lambda x: x.strippedByType(), self.args)))

@dataclass(order=True, frozen=True, eq=True)
class FOr(Formula):
    args: Tuple[Formula]
    meta: str = field(default_factory=lambda : "FOr")
    matched: bool = field(default_factory=lambda : False)


    def __str__(self):
        return " \// ".join(map(str, self.args))

    def isOntoUnmatched(self):
        return any(map(lambda x: x is not None and x.isOntoUnmatched(), self.args))

    def removePropertiesFrom(self, coll, onMatch=False):
        return FOr(args=tuple([x.removePropertiesFrom(coll, onMatch) if x is not None else None for x in self.args]), matched=self.matched)


    def getFlattenedProperties(self):
        flattenedSuperSet = defaultdict(set)
        keys = set()
        for arg in self.args:
            if arg is not None:
                for x,y in arg.getFlattenedProperties():
                    if x not in flattenedSuperSet:
                        flattenedSuperSet[x] = set(y)
                    else:
                        flattenedSuperSet[x] = flattenedSuperSet[x].union(set(y))
        return frozenset({x:tuple(y) for x,y in flattenedSuperSet.items()}.items())

    def updateWithProperties(self, toFrozenSet):
        return self

    def isUnresolved(self):
        if self.matched: return False
        return any(map(is_formula_unresolved, self.args))

    def collectIds(self):
        yield id(self)
        for arg in self.args:
            if arg is not None and isinstance(arg, Formula):
                yield from arg.collectIds()

    def matchWith(self, f, d, ancestor, fugitive):
        isAncestorNone = ancestor is None
        if self.meta == f.meta:
            if ancestor is None:
                ancestor = self
            L = []
            dd = copy.deepcopy(d)
            for x, y in zip(self.args, f.args):
                z = match_formula(x, y, dd, ancestor, fugitive)
                if z == x:
                    ancestor = None
                    break
                else:
                    L.append(z)
            if ancestor == self:
                for x, y in dd.items():
                    d[x] = y
                var = FOr(args=tuple(L), matched=True)
                if isAncestorNone:
                    retaliate_dd(d, self, var, fugitive)
                return var
        if isAncestorNone:
            ancestor = None
        return FOr(args=tuple(map(lambda x: match_formula(x, f, d, ancestor, fugitive), self.args)))

    def replaceWith(self, m, onObjectId=False, d=None, fugitive=None, forAllProperties=None):
        if m is None or len(m) == 0:
            return self
        if self in m:
            return m[self]
        if onObjectId and id(self) == m["obj"]:
            onObjectId = False
        var= FOr(args=tuple(map(lambda x: replace_formula(x, m, onObjectId,d, fugitive,forAllProperties), self.args)), matched=self.matched)
        retaliate_dd(d, self, var, fugitive)
        return var

    def semantic(self, d:Dict[Formula, bool]):
        return max(map(lambda x: x.semantic(d), self.args))

    def getAtoms(self):
        s = set()
        for x in self.args:
            s = s.union(x.getAtoms())
        return s

    def strippedByType(self):
        return FOr(args=tuple(map(lambda x: x.strippedByType(), self.args)))

@dataclass(order=True, frozen=True, eq=True)
class FNot(Formula):
    arg: Formula
    meta: str = field(default_factory=lambda : "FNot")
    matched: bool = field(default_factory=lambda : False)

    def __str__(self):
        return " ~ "+str(self.arg)

    def isOntoUnmatched(self):
        return self.arg is not None and self.arg.isOntoUnmatched()

    def removePropertiesFrom(self, coll, onMatch=False):
        return FNot(arg=self.arg.removePropertiesFrom(coll, onMatch) if self.arg is not None else None, matched=self.matched)

    def getFlattenedProperties(self):
        if self.arg is not None:
            return self.arg.getFlattenedProperties()
        return super().getFlattenedProperties()

    def updateWithProperties(self, toFrozenSet):
        return self

    def isUnresolved(self):
        if self.matched: return False
        return is_formula_unresolved(self.arg)

    def collectIds(self):
        yield id(self)
        if self.arg is not None and isinstance(self.arg, Formula):
                yield from self.arg.collectIds()

    def matchWith(self, f, d, ancestor, fugitive):
        isAncestorNone = ancestor is None
        if self.meta == f.meta:
            if ancestor is None:
                ancestor = self
            dd = copy.deepcopy(d)
            z = match_formula(self.arg, f.arg, dd, ancestor, fugitive)
            if z != self.arg:
                for x, y in dd.items():
                    d[x] = y
                var = FNot(arg=z, matched=True)
                if isAncestorNone:
                    retaliate_dd(d, self, var, fugitive)
                return var
            else:
                ancestor = None
        if isAncestorNone:
            ancestor = None
        return FNot(arg=match_formula(self.arg, f, d, ancestor, fugitive))

    def replaceWith(self, map, onObjectId=False, d=None, fugitive=None, forAllProperties=None):
        if map is None or len(map) == 0:
            return self
        if self in map:
            return map[self]
        if onObjectId and "obj" in map and id(self) == map["obj"]:
            onObjectId = False
        arg = replace_formula(self.arg, map, onObjectId, d, fugitive, forAllProperties)
        var= FNot(arg=arg, matched=self.matched)
        retaliate_dd(d, self, var, fugitive)
        return var

    def semantic(self, d:Dict[Formula, bool]):
        return 1-self.arg.semantic(d)

    def getAtoms(self):
        return self.arg.getAtoms()

    def strippedByType(self):
        return FNot(arg=self.arg.strippedByType())


def formula_from_dict(f:Union[dict,str]):
    """
    Loading a json file in its object-dictionary rerpesentation into formulaes.
    This is mainly used to run the pipeline from one point at a time.
    """
    if f is None:
        return None
    if isinstance(f, str):
        return f
    from collections.abc import Iterable
    if isinstance(f, Iterable) and not (isinstance(f, dict)):
        return list(map(formula_from_dict, f))
    assert "meta" in f
    meta = f["meta"]
    if meta == "FNot":
        return FNot(arg=formula_from_dict(f["arg"]))
    if meta == "FOr":
        return FOr(args=tuple(map(formula_from_dict, f["args"])))
    if meta == "FAnd":
        return FAnd(args=tuple(map(formula_from_dict, f["args"])))
    if meta == "FVariable":
        name = str(f["name"]) if "name" in f and f["name"] is not None else None
        type = str(f["type"]) if "name" in f and f["type"] is not None else None
        specification = str(f["specification"]) if "specification" in f and f["specification"] is not None else None
        cop = formula_from_dict(f["cop"]) if "cop" in f else None
        return FVariable(name=name, type=type, specification=specification, cop=cop)
    if meta == "FUnaryPredicate":
        rel = str(f["rel"]) if "rel" in f and f["rel"] is not None else ""
        arg = formula_from_dict(f["arg"]) if "arg" in f and f["arg"] is not None else None
        score  = float(f["score"]) if "score" in f else 1.0
        properties = defaultdict(list)
        if "properties" in f:
            for k,v in f["properties"].items():
                for x in v:
                    properties[k].append(formula_from_dict(x))
        properties = {k: tuple(v) for k,v in properties.items()}
        return FUnaryPredicate(rel, arg, score, frozenset(properties.items()))
    if meta == "FBinaryPredicate":
        rel = str(f["rel"]) if "rel" in f else ""
        src = formula_from_dict(f["src"]) if "src" in f else None
        dst = formula_from_dict(f["dst"]) if "dst" in f else None
        score  = float(f["score"]) if "score" in f else 1.0
        properties = defaultdict(list)
        if "properties" in f:
            for k, v in f["properties"].items():
                for x in v:
                    properties[k].append(formula_from_dict(x))
        properties = {k: tuple(v) for k, v in properties.items()}
        return FBinaryPredicate(rel, src, dst, score, frozenset(properties.items()))
