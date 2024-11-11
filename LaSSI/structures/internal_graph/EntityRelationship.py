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
class SingletonProperties():
    begin: str
    end: str
    pos: str
    specification: str = None
    number: str = None
    extra: str = None

def replaceNamed(entity: 'Singleton', s: str) -> 'Singleton':
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
    root: bool = False


def deserialize_NodeEntryPoint(data: dict) -> NodeEntryPoint:
    if data is None:
        return data
    if data['type'] == 'SetOfSingletons':
        return SetOfSingletons(id=int(data.get('id')),
                               type=Grouping(int(data.get('type'))),
                               entities =tuple(map(deserialize_NodeEntryPoint, data.get('entities'))),
            min=int(data.get('min')),
            max=int(data.get('max')),
            confidence=float(data.get('confidence')
                                             ))
    elif data['type'] == 'Singleton':
        return Singleton(
            id=int(data.get('id')),
            named_entity=data.get('named_entity'),
            properties=frozenset(data.get('properties').items()),
            min=int(data.get('min')),
            max=int(data.get('max')),
            confidence=float(data.get('confidence')),
                             type=data.get('type'),
                             )



@dataclass(order=True, frozen=True, eq=True)
class Relationship:  # Representation of an edge
    source: NodeEntryPoint  # Source node
    target: NodeEntryPoint  # Target node
    edgeLabel: 'Singleton'  # Edge label, also represented as an entity with properties
    isNegated: bool = False  # Whether the edge expresses a negated action

    @staticmethod
    def from_nodes(r, nodes):
        return Relationship(source=nodes[r.source.id] if r.source.id >= 0 else r.source, target=nodes[r.target.id] if r.target.id >= 0 else r.target, edgeLabel=r.edgeLabel, isNegated=r.isNegated)

    @classmethod
    def from_dict(cls, c):
        return cls(source=deserialize_NodeEntryPoint(c.get('source')),
                   target=deserialize_NodeEntryPoint(c.get('target')),
                   edgeLabel=deserialize_NodeEntryPoint(c.get('edgeLabel')),
                   isNegated=bool(c.get('Singleton', False))
                   )


@dataclass(order=True, frozen=True, eq=True)
class Singleton(NodeEntryPoint):  # Graph node representing just one entity
    id: int
    named_entity: str  # String representation of the entity
    properties: frozenset  # Key-Value association for such entity
    min: int
    max: int
    type: str
    confidence: float
    kernel: Relationship = None

    @staticmethod
    def get_props(node):
        return dict(node.properties) if node is not None and isinstance(node, Singleton) else None

    @staticmethod
    def update_node_props(node, node_props):
        return Singleton(
            id=node.id,
            named_entity=node.named_entity,
            properties=frozenset(node_props.items()),
            min=node.min,
            max=node.max,
            type=node.type,
            confidence=node.confidence,
        )

    @staticmethod
    def strip_root_properties(node):
        sing_props = dict(node.properties)
        if 'kernel' in sing_props:
            sing_props.pop('kernel')
        if 'root' in sing_props:
            sing_props.pop('root')

        return Singleton(
            id=node.id,
            named_entity=node.named_entity,
            properties=frozenset(sing_props.items()),
            min=node.min,
            max=node.max,
            type=node.type,
            confidence=node.confidence,
            kernel=node.kernel,
        )

    @classmethod
    def from_dict(cls, c):
        return cls(kernel=Relationship.from_dict(c.get('kernel')),
                   properties={k: [deserialize_NodeEntryPoint(x) for x in v] for k, v in c.get('properties').items()}
                   )

    # Rewrite Singleton(kernel) in form edgeLabel(source[props], target[props])[props]
    def to_string(self, node=None):
        def get_node_properties_string(node_to_use):
            if node_to_use is None or not isinstance(node_to_use, Singleton):
                return ''

            props_to_ignore = ['begin', 'pos', 'end', 'kernel', 'lemma', 'specification', 'number', 'root', 'expl']
            properties_list = []
            for key in dict(node_to_use.properties):
                if key not in props_to_ignore:
                    properties_key_ = dict(node_to_use.properties)[key]
                    if isinstance(properties_key_, str) and key != 'cop':
                        properties_list.append(f'({key}:{properties_key_})')
                    elif key == 'cop':
                        copula_sing = Singleton(
                            int(properties_key_['id']),
                            properties_key_['named_entity'],
                            frozenset(dict(properties_key_['properties']).items()),
                            int(properties_key_['min']),
                            int(properties_key_['max']),
                            properties_key_['type'],
                            float(properties_key_['confidence'])
                        )
                        properties_list.append(f'({key}:{get_node_string(copula_sing)})')
                    else:
                        for node in properties_key_:
                            if key == 'None':  # It is a node with kernel (most likely)
                                properties_list.append(self.to_string(node))
                            else:
                                properties_list.append(f'({key}:{get_node_string(node)})')

            return f'[{", ".join(properties_list)}]' if len(properties_list) > 0 else ''

        def get_node_string(node):
            node_string = node.named_entity if node is not None and isinstance(node, Singleton) else 'None'

            # Join SetOfSingletons
            node_string = f"{node.type.name}({', '.join(get_node_string(entity) for entity in node.entities)})" if isinstance(
                node, SetOfSingletons) and node_string == 'None' else node_string

            # Add properties
            node_string = f"{node_string}{get_node_properties_string(node)}"

            return node_string

        if node is None:
            node = self
        edge_label = node.kernel.edgeLabel.named_entity
        source = get_node_string(node.kernel.source)
        target = get_node_string(node.kernel.target)

        properties = get_node_properties_string(node)

        if node.kernel:
            return f"{edge_label}({source}, {target}){properties}"
        else:
            return node.named_entity

