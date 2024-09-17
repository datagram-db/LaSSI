__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2024, Oliver R. Fox"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"
__email__ = "ollie.fox5@gmail.com"
__status__ = "Production"
from dataclasses import dataclass, field
from typing import List

from LaSSI.structures.internal_graph.EntityRelationship import Relationship, NodeEntryPoint

@dataclass(order=True, frozen=True, eq=True)
class Sentence:
    kernel: Relationship
    properties: dict = field(default_factory=lambda: {
        'time': List[NodeEntryPoint],
        'loc': List[NodeEntryPoint]
    })