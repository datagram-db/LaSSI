import copy
import dataclasses
import itertools
import json
from collections.abc import Iterable
from typing import Tuple

from logical_repr.Sentences import formula_from_dict


@dataclasses.dataclass(order=True, frozen=True, eq=True)
class FieldMatch:
    action: str
    n: int  #argn
    fields: Tuple[str] #fields navigation
    attr: str
    with_value: str
    as_name: str

@dataclasses.dataclass(order=True, frozen=True, eq=True)
class SentenceMatch:
    relname: str
    n: int
    fields: Tuple[FieldMatch]
    as_name: str
    parents: int

def sanity_check2(query):
    if query is None:
        raise ValueError("ERROR: query cannot be null!")
    if query.action == "MATCH":
        raise ValueError("ERROR: cannot MATCH a rewriting!")

def sanity_check(query):
    if query is None:
        raise ValueError("ERROR: query cannot be null!")
    if query.action == "REPLACE":
        raise ValueError("ERROR: cannot replace at match!")
    if query.action == "REMOVE":
        raise ValueError("ERROR: cannot remove at match!")

def attr_match(self, query:FieldMatch, attr, q_val):
    result = None
    field = query if query.as_name is None else query.as_name
    if (attr is None) or len(attr) == 0:
        result = self
    elif not hasattr(self, attr):
        result = None
    else:
        val = getattr(self, attr)
        if (((q_val is None) and (val is not None)) or ((q_val is not None) and len(q_val) == 0)) or ((val == q_val)):
            result = val
        else:
            result = None
    if query is None:
        return result
    elif result is not None:
        return [{field:result}]
    else:
        return []

def set_attr_match(self, attr, q_val, m):
    if (attr is None) or len(attr) == 0 or not hasattr(self, attr):
        return self
    else:
        return self.replaceWith(attr, q_val, m)

# def field_update(self, query:FieldMatch, fields:list[str], m):
#     sanity_check2(query)
#     if (len(fields) == 0):
#         attr = query.attr
#         q_val = query.with_value
#         return set_attr_match(self, attr, q_val, m, q.as_name)
#     else:
#         curr = fields.pop(0)
#         if curr.startswith("@"):
#             curr = curr[1:]
#             if self.meta == curr:
#                 return field_update(self, query, copy.deepcopy(fields), m)
#             else:
#                 return []
#         elif (curr == "-1"):
#             if not hasattr(self, "properties"):
#                 return []
#             else:
#                 return properties_update(self.properties, query, copy.deepcopy(fields), m)
#         else:
#             from logical_repr.Sentences import FUnaryPredicate
#             from logical_repr.Sentences import FNot
#             from logical_repr.Sentences import FBinaryPredicate
#             if curr == "0":
#
#                 if isinstance(self, FUnaryPredicate) or isinstance(self, FNot):
#                     curr = "arg"
#                 elif isinstance(self, FBinaryPredicate):
#                     curr = "src"
#                 else:
#                     curr = None
#             elif curr == "1":
#                 if isinstance(self, FBinaryPredicate):
#                     curr = "dst"
#                 else:
#                     curr = None
#             if hasattr(self, curr):
#                 return field_update(getattr(self, curr), query, copy.deepcopy(fields), m)
#             else:
#                 if hasattr(self, "properties"):
#                     l = copy.deepcopy(fields)
#                     l.insert(0, curr)
#                     return properties_update(self.properties, query, l, m)
#                 else:
#                     return self
#     return self

def field_match(self, query:FieldMatch, fields:list[str]):
    sanity_check(query)
    if (len(fields) == 0):
        attr = query.attr
        q_val = query.with_value
        return attr_match(self, query, attr, q_val)
    else:
        curr = fields.pop(0)
        if curr.startswith("@"):
            curr = curr[1:]
            if self.meta == curr:
                return field_match(self, query, copy.deepcopy(fields))
            else:
                return []
        elif (curr == "-1"):
            if not hasattr(self, "properties"):
                return []
            else:
                return properties_match(self.properties, query, copy.deepcopy(fields))
        else:
            from logical_repr.Sentences import FUnaryPredicate
            from logical_repr.Sentences import FNot
            from logical_repr.Sentences import FBinaryPredicate
            if curr == "0":
                if isinstance(self, FUnaryPredicate) or isinstance(self, FNot):
                    curr = "arg"
                elif isinstance(self, FBinaryPredicate):
                    curr = "src"
                else:
                    curr = None
            elif curr == "1":
                if isinstance(self, FBinaryPredicate):
                    curr = "dst"
                else:
                    curr = None
            if hasattr(self, curr):
                return field_match(getattr(self, curr), query, copy.deepcopy(fields))
            else:
                if hasattr(self, "properties"):
                    l = copy.deepcopy(fields)
                    l.insert(0, curr)
                    return properties_match(self.properties, query, l)
                else:
                    return []
    return []


def properties_match(frozenProperties, query:FieldMatch, fields:list[str]):
    sanity_check(query)
    if frozenProperties is None:
        return []
    if isinstance(frozenProperties, frozenset):
        d = dict(frozenProperties)
        if len(fields) == 0:
            raise ValueError("ERROR: we are still in a frozen property!")
        else:
            curr = fields.pop(0)
            result = None
            if curr in d:
                for elements in d[curr]:
                    tmp = properties_match(elements, query, copy.deepcopy(fields))
                    if (len(tmp) == 0):
                        return []
                    if result == None:
                        result = tmp
                    else:
                        result = [a | b for a, b in itertools.product(result, tmp)]
                return result
            else:
                return []
    else:
        if len(fields) == 0:
            from logical_repr.Sentences import Formula
            if isinstance(frozenProperties, Formula):
                return field_match(frozenProperties, query, fields)
            else:
                R = []
                if isinstance(frozenProperties, str):
                    R.append({query:frozenProperties})
                elif isinstance(frozenProperties, Iterable):
                    for field in frozenProperties:
                        from logical_repr.Sentences import FVariable
                        if isinstance(field, str):
                            R.append({query: field})
                        elif isinstance(field, FVariable):
                            if (query.attr == "name"):
                                R.append({query: field.name})
                            elif (query.attr == "type"):
                                R.append({query: field.type})
                            elif (query.attr == "specification"):
                                R.append({query: field.specification})
                else:
                    raise ValueError(frozenProperties)
                return R
        else:
            curr = fields.pop(0)
            if (curr.isdigit()) or curr.startswith("@"):
                L = copy.deepcopy(fields)
                L.insert(0, curr)
                return field_match(frozenProperties, query, L)
            test = attr_match(frozenProperties, None, curr, None)
            if test is None:
                return []
            else:
                return properties_match(test, query, copy.deepcopy(fields))

def match_formula_no_fields(self, q:SentenceMatch):
    if self is None:
        yield from []
    else:
        if (self.meta == "FNot"):
            for x in match_formula_no_fields(self.arg, q):
                    L = copy.deepcopy(x)
                    L.append(self)
                    yield L
        elif (self.meta == "FAnd") or (self.meta == "FOr"):
            for x in match_formula_no_fields(self.args[0], q):
                    x.append(self)
                    yield x
            for x in  match_formula_no_fields(self.args[1], q):
                x.append(self)
                yield x
        else:
            if (q.relname is not None) and (len(q.relname) > 0):
                if (not hasattr(self, "rel")) or (self.rel != q.relname):
                    yield from []
                    return  # No match
            if (q.n == 1) and (self.meta == "FUnaryPredicate"):
                yield [self]
            elif (q.n == 2) and (self.meta == "FBinaryPredicate"):
                yield [self]
            yield from []

def sentence_match_formula(self, q:SentenceMatch):
    if self is None:
        return []
    else:
        zero_arg = None
        uno_arg = None
        props = None
        if (q.n == 1) and (self.meta == "FUnaryPredicate"):
            zero_arg = self.arg
            uno_arg = None
            props = self.properties
        elif (q.n == 2) and (self.meta == "FBinaryPredicate"):
            zero_arg = self.src
            uno_arg = self.dst
            props = self.properties
        result = None
        if len(q.fields) == 0:
            return [dict()]  # Match with no variable assignment
        for x in q.fields:
            tmp = []
            if (x.n is None):
                tmp = field_match(self, x, list(x.fields))
            if (x.n == -1):
                tmp = properties_match(props, x, list(x.fields))
            elif (x.n == 0):
                tmp = field_match(zero_arg, x, list(x.fields))
            elif (x.n == 1):
                tmp = field_match(uno_arg, x, list(x.fields))
            if (len(tmp) == 0):
                return []
            if result == None:
                result = tmp
            else:
                result = [a | b for a, b in itertools.product(result, tmp)]
        return result


def simple_data_match(formula, q):
    L = list(match_formula_no_fields(formula, q))
    for z in L:
        if len(z)>q.parents:
            x = z[q.parents]
            for y in sentence_match_formula(x, q):
                if (q.as_name is not None) and (len(q.as_name) > 0):
                    y[q.as_name] = x
                    yield y
                else:
                    yield y

def boolean_simple_data_match(formula, q):
    L = list(simple_data_match(formula, q))
    return len(L)>0


qs = [
    SentenceMatch("be",
                          1,
                          tuple([FieldMatch("MATCH", -1, tuple(["GPE"]), "name", None, "x"),
                                 FieldMatch("MATCH", -1, tuple(["GPE"]), "type", None, "t")]),
                          "obj_x",
                          0),
        SentenceMatch("be",
                          1,
                          tuple([FieldMatch("MATCH", -1, tuple(["GPE", "specification"]), None, None, "s")]),
                          "obj_y",
                           0),
         SentenceMatch("be",
                          1,
                          tuple([FieldMatch("MATCH", -1, tuple(["GPE"]), "meta", "FNot", "neg")]),
                          "obj_z",
                           0),
         SentenceMatch("be",
                          1,
                          tuple([FieldMatch("MATCH", -1, tuple(["GPE", "@FNot", "0"]), None, None, "negarg")]),
                          "obj_t",
                           0),
         SentenceMatch("be",
                          1,
                          tuple([FieldMatch("MATCH", None, tuple(["@FNot"]), None, None, "neg")]),
                          "obj_t",
                           1),
    SentenceMatch("have",
                  2,
                  tuple([FieldMatch("MATCH", 1, tuple([]), "type", "JJ", None)]),
                  "obj_t",
                  0),
]


def test_query(list_json, qs, i, n):
    print(f"{i}:")
    L = list(simple_data_match(list_json[i], qs[n]))
    for x in L:
        print(x)


if __name__ == "__main__":
    with open("/old_specifications/newcastle_sentences.txt_logical_rewriting.json", "r") as f:
        list_json = json.load(f)
        list_json = formula_from_dict(list_json)
        n, m = -1,0
        # test_query(list_json, qs, 7, n)
        for i in range(len(list_json)):
            test_query(list_json, qs, i, n)

