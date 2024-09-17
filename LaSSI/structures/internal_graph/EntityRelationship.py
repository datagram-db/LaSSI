__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox, Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"


from dataclasses import dataclass
from enum import Enum
from typing import List


class Grouping(Enum):
    AND = 0
    OR = 1
    NEITHER = 2
    NOT = 3
    NONE = 4
    GROUPING = 5
    MULTIINDIRECT = 6


# class Properties(TypedDict):  # Key-Value association
#     property: str  # Key
#     value: Union[int, str, float, bool]  # Value


class NodeEntryPoint:
    pass


@dataclass(order=True, frozen=True, eq=True)
class Singleton(NodeEntryPoint):  # Graph node representing just one entity
    id: int
    named_entity: str  # String representation of the entity
    properties: frozenset  # Key-Value association for such entity
    min: int
    max: int
    type: str
    confidence: float


def replaceNamed(entity: Singleton, s: str) -> Singleton:
    return Singleton(id=entity.id,
                     named_entity=s,
                     properties=entity.properties,
                     min=entity.min,
                     max=entity.max,
                     type=entity.type,
                     confidence=entity.confidence)


@dataclass(order=True, frozen=True, eq=True)
class SetOfSingletons(NodeEntryPoint):  # Graph node representing conjunction/disjunction/exclusion between entities
    id: int
    type: Grouping  # Type of node grouping
    entities: List[NodeEntryPoint]  # A list of entity nodes
    min: int
    max: int
    confidence: float


@dataclass(order=True, frozen=True, eq=True)
class Relationship:  # Representation of an edge
    source: NodeEntryPoint  # Source node
    target: NodeEntryPoint  # Target node
    edgeLabel: Singleton  # Edge label, also represented as an entity with properties
    isNegated: bool = False  # Whether the edge expresses a negated action
