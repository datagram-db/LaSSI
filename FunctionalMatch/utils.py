import collections
from collections import defaultdict
import copy
from dataclasses import is_dataclass, fields, dataclass
from typing import Dict, List

from FunctionalMatch.functions.Reference import Reference


@dataclass
class ObjDepthDeterminerResult:
    depth_dict: Dict[Reference, int]
    layer_dict: Dict[int, List[object]]
    root_to_ancestors_ref: set
    root_to_ancestors: set

class ObjDepthDeterminer():
    def __init__(self, object):
        self.visited = set()
        self.rec_stack = set()
        self.object = object
        self.depth = defaultdict(int)
        self.parents = defaultdict(set)
        self.is_cyclic_util(self.object)

    def is_cyclic_util(self, u):
        if u is None:
            return False
        # If already present in the recursion call
        # stack, then there is a cycle
        if u in self.rec_stack:
            return True

        # Mark the current node as visited
        # and part of recursion stack
        self.visited.add(u)
        self.rec_stack.add(u)
        self.depth[u] = max(map(lambda x: self.depth[x], self.parents[u]), default=0)+1

        # Recur for all the vertices adjacent
        # to this vertex
        if is_dataclass(u):
            for attr in fields(u):
                x = getattr(u, attr.name)
                self.parents[x].add(u)
                if (x not in self.visited) and self.is_cyclic_util(x):
                    return True
                elif x in self.rec_stack:
                    return True

        # Remove the vertex from recursion stack
        self.rec_stack.remove(u)
        return False

    @staticmethod
    def get_depth_dictionary(obj):
        computer = ObjDepthDeterminer(obj)
        depth_dict = computer.depth
        layer_dict = defaultdict(list)
        for k, v in depth_dict.items():
            layer_dict[v].append(Reference.from_object(k))
        return ObjDepthDeterminerResult({Reference.from_object(k): v for k,v in depth_dict.items()}, dict(layer_dict), {Reference.from_object(x) for x in computer.visited if is_dataclass(x)}, computer.visited)

def depth_dictionary(obj):
    assert is_dataclass(obj)
    visited = set()
    q = []
    q.append(obj)
    while len(q) > 0:
        obj = q.pop(0)
        if obj in visited:
            continue
        visited.add(obj)

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


class CountingDictionary:
    def __init__(self):
        self.counter = dict()
        self.reverseConstituent = dict()

    def add(self, x):
        if x not in self.counter:
            self.reverseConstituent[len(self.counter)] = x
            self.counter[x] = len(self.counter)
        return self.counter[x]

    def add_with_wasPresent(self, x):
        wasPresent = True
        if x not in self.counter:
            self.reverseConstituent[len(self.counter)] = x
            self.counter[x] = len(self.counter)
            wasPresent = False
        return self.counter[x], wasPresent

    def getAllObjects(self):
        return list(map(self.fromId, range(len(self.counter))))

    def __len__(self):
        return len(self.counter)

    def fromId(self, x):
        if int(x) not in self.reverseConstituent:
            return None
        return self.reverseConstituent[int(x)]

    def contains(self, x):
        if x not in self.counter:
            return -1
        return self.counter[x]

    @staticmethod
    def load(ke_file):
        import os
        if os.path.exists(ke_file):
            with open(ke_file, "rb") as f:
                import pickle
                constituents = pickle.load(f)
                assert isinstance(constituents, CountingDictionary)
                return constituents
        else:
            return CountingDictionary()

    def dump(self, ke_file):
        with open(ke_file, "wb") as f:
            import pickle
            pickle.dump(self, f)


class FrozenDict(collections.abc.Mapping):
    """author: https://stackoverflow.com/a/2704866/1376095"""

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = None

    @staticmethod
    def from_dictionary(d):
        return FrozenDict(**d)

    def __dict__(self):
        return self._d

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def dict(self):
        return self._d

    def __contains__(self, key):
        return key in self._d

    def __iter__(self):
        return iter(self._d)

    def items(self):
        return self._d.items()

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def update(self, key, value):
        j = copy.deepcopy(self._d)
        if value is None:
            if key in j:
                del j[key]
        else:
            j[key] = value
        return FrozenDict(**j)

    def __contains__(self, item):
        return item in self._d

    def get(self, key, value=None):
        return self._d.get(key, value)

    def __hash__(self):
        # It would have been simpler and maybe more obvious to
        # use hash(tuple(sorted(self._d.iteritems()))) from this discussion
        # so far, but this solution is O(n). I don't know what kind of
        # n we are going to run into, but sometimes it's hard to resist the
        # urge to optimize when it will gain improved algorithmic performance.
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash
