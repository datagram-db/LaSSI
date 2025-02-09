__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox, Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

import re
from collections import defaultdict
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

    def extract_properties(self, p):
        if (self.entities is None) or len(self.entities) == 0:
            yield from []
        else:
            for x in self.entities:
                yield from x.extract_properties(p)

    @staticmethod
    def update_entities(node, new_entities):
        return SetOfSingletons(
            id=node.id,
            type=node.type,
            entities=tuple(new_entities),
            min=node.min,
            max=node.max,
            confidence=node.confidence,
            root=node.root
        )


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

    def extract_properties(self, p):
        if (self.source is None) and (self.target is None):
            yield from []
        else:
            if self.source is not None:
                yield from self.source.extract_properties(p)
            if self.target is not None:
                yield from self.target.extract_properties(p)

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

    def extract_properties(self, p):
        if ((self.properties is None) or len(self.properties) == 0) and self.kernel is None:
            yield from []
        else:
            for k, v in self.properties:
                if isinstance(v, list) or isinstance(v, tuple):
                   for x in v:
                       if p(k, x):
                           yield k, x
                       if type(x).__name__ == 'Relationship' or type(x).__name__ == 'Singleton' or type(x).__name__ == 'SetOfSingletons':
                            yield from x.extract_properties(p)
                elif p(k, v):
                    yield k, v
                if type(v).__name__ == 'Relationship' or type(v).__name__ == 'Singleton' or type(
                            v).__name__ == 'SetOfSingletons':
                    yield from v.extract_properties(p)
            if self.kernel is not None:
                yield from self.kernel.extract_properties(p)

    @staticmethod
    def get_props(node, properties=None):
        if properties is None:
            properties = dict()
        if node is None:
            return None

        if isinstance(node, Singleton):
            return dict(node.properties)
        elif isinstance(node, SetOfSingletons):
            for entity in node.entities:
                properties |= Singleton.get_props(entity, properties)
            return properties

    @staticmethod
    def update_node_props(node, node_props):
        from LaSSI.ner.node_functions import create_props_for_singleton
        return Singleton(
            id=node.id,
            named_entity=node.named_entity,
            min=node.min,
            max=node.max,
            type=node.type,
            confidence=node.confidence,
            kernel=node.kernel,
            properties=create_props_for_singleton(node_props),
        )

    @staticmethod
    def update_name(node, new_name):
        return Singleton(
            id=node.id,
            named_entity=new_name,
            min=node.min,
            max=node.max,
            type=node.type,
            confidence=node.confidence,
            kernel=node.kernel,
            properties=node.properties,
        )

    @staticmethod
    def update_kernel(node, new_node, kernel_part):
        return Singleton(
            id=node.id,
            named_entity='',
            type='SENTENCE',
            min=node.min,
            max=node.max,
            confidence=1,
            kernel=Relationship(
                source=new_node if kernel_part == "source" else node.kernel.source,
                target=new_node if kernel_part == "target" else node.kernel.target,
                edgeLabel=new_node if kernel_part == "edgeLabel" else node.kernel.edgeLabel,
                isNegated=node.kernel.isNegated,
            ),
            properties=node.properties,
        )

    @staticmethod
    def remove_prop(node, prop_name):
        node_props = dict(node.properties)
        node_props.pop(prop_name)
        return Singleton.update_node_props(node, node_props)

    @staticmethod
    def add_root_property(node):
        if isinstance(node, SetOfSingletons):
            return SetOfSingletons(
                id=node.id,
                type=node.type,
                entities=node.entities,
                min=node.min,
                max=node.max,
                confidence=node.confidence,
                root=True
            )

        node_props = dict(node.properties)
        node_props['kernel'] = 'root'
        return Singleton.update_node_props(node, node_props)


    @staticmethod
    def strip_root_properties(node):
        if isinstance(node, SetOfSingletons):
            return SetOfSingletons(
                id=node.id,
                type=node.type,
                entities=tuple(Singleton.strip_root_properties(entity) for entity in node.entities),
                min=node.min,
                max=node.max,
                confidence=node.confidence,
                root=False
            )

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

    # Rewrite Singleton(kernel) in form edgeLabel[props](source[props], target[props])[props]
    def to_string(self, node=None):
        def get_node_properties_string(node_to_use):
            if node_to_use is None or not isinstance(node_to_use, Singleton):
                return ''

            props_to_ignore = ['begin', 'pos', 'end', 'kernel', 'lemma', 'specification', 'number', 'root', 'expl', 'cc', 'conj', 'neg']
            properties_list = defaultdict(list)
            for key in dict(node_to_use.properties):
                if key not in props_to_ignore:
                    properties_key_ = dict(node_to_use.properties)[key]
                    if isinstance(properties_key_, str) and properties_key_ != '':
                        try:
                            key = str(int(float(key)))
                        except ValueError:
                            key = key

                        properties_list[key].append(properties_key_)
                    else:
                        if isinstance(properties_key_, Singleton):
                            properties_list[key].append(get_node_string(properties_key_))
                        else:
                            for node in properties_key_:
                                if key == 'SENTENCE':  # It is a node with kernel (most likely)
                                    properties_list[key].append(self.to_string(node))
                                else:
                                    if key in {'nmod', 'nmod_poss', 'acl_relcl'}:
                                        properties_list[key].append(get_node_string(node))
                                    else:
                                        properties_list[key].append(get_node_string(node))

            return re.sub(r"(nmod|nmod_poss|acl_relcl):\1", r"\1", f"""[{", ".join(f'({k}:{v[0] if len(v) == 1 else "[" + ", ".join(v) + "]"})' for k, v in properties_list.items())}]""" if len(properties_list) > 0 else '')

        def get_node_string(node):
            node_string = node.named_entity if node is not None and isinstance(node, Singleton) else 'None'

            # If node_string is empty, it is a kernel so return that
            if node_string == '':
                node_string = node.to_string(node) if node_string == "" else node_string
            else:
                # Join SetOfSingletons
                node_string = f"{node.type.name}({', '.join(get_node_string(entity) for entity in node.entities)})" if isinstance(
                    node, SetOfSingletons) and node_string == 'None' else node_string

                # Add properties
                node_string = f"{node_string}{get_node_properties_string(node)}"
            return node_string

        if node is None:
            node = self

        if node.kernel:
            edge_label = get_node_string(node.kernel.edgeLabel) if node.kernel.edgeLabel is not None else 'None'
            source = get_node_string(node.kernel.source) if node.kernel.source is not None else 'None'
            target = get_node_string(node.kernel.target) if node.kernel.target is not None else 'None'

            properties = get_node_properties_string(node)

            return f"{edge_label}({source}, {target}){properties}" if not node.kernel.isNegated else f"NOT({edge_label}({source}, {target}){properties})"
        else:
            return node.named_entity

