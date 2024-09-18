import dataclasses
from dataclasses import dataclass
from typing import List

from LaSSI.structures.internal_graph.Graph import Graph
from LaSSI.structures.kernels.Sentence import Sentence

@dataclass
class InternalRepresentation:
    internal_graph: Graph
    sentences: List[Sentence] = dataclasses.field(default_factory=list)

    @classmethod
    def from_dict(cls, data):
        return cls(
            internal_graph=Graph.from_dict(data.get('internal_graph')),
            sentences=[Sentence.from_dict(x) for x in data.get('sentences')]
        )
