from dataclasses import dataclass
from typing import List

from LaSSI.structures.internal_graph.EntityRelationship import Relationship
@dataclass(order=True, frozen=True, eq=True)
class Graph:
    edges: List[Relationship]  # A graph is defined as a collection of edges

    @classmethod
    def from_dict(cls, param):
        return cls(edges = [Relationship.from_dict(x) for x in param.get('edges')])
