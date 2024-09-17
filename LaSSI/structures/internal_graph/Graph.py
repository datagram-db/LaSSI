from dataclasses import dataclass
from typing import List

from LaSSI.structures.internal_graph.EntityRelationship import Relationship
@dataclass(order=True, frozen=True, eq=True)
class Graph:
    edges: List[Relationship]  # A graph is defined as a collection of edges
