__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2024, Oliver R. Fox"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"
__email__ = "ollie.fox5@gmail.com"
__status__ = "Production"

import json
import re
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from typing import List

from LaSSI.external_services.Services import Services
from LaSSI.files.JSONDump import json_dumps
from LaSSI.structures.internal_graph.EntityRelationship import Relationship, NodeEntryPoint, Singleton, SetOfSingletons, \
    Grouping, deserialize_NodeEntryPoint


@dataclass(order=True, frozen=True, eq=True)
class Sentence:
    kernel: Relationship
    properties: dict = field(default_factory=lambda: {
        'time': List[NodeEntryPoint],
        'loc': List[NodeEntryPoint]
    })

    @classmethod
    def from_dict(cls, c):
        return cls(kernel=Relationship.from_dict(c.get('kernel')),
                   properties={k: [deserialize_NodeEntryPoint(x) for x in v] for k, v in c.get('properties').items()}
                   )


def replaceNamed(entity: Singleton, s: str) -> Singleton:
    return Singleton(id=entity.id,
                     named_entity=s,
                     properties=entity.properties,
                     min=entity.min,
                     max=entity.max,
                     type=entity.type,
                     confidence=entity.confidence)


def create_existential(edges, nodes):
    for key in nodes:
        node = nodes[key]
        if isinstance(node, SetOfSingletons):
            node = node.entities[0]
        for prop in dict(node.properties):
            if 'kernel' in prop or len(nodes) == 1:
                edges.append(Relationship(
                    source=node,
                    target=Singleton(
                        id=-1,
                        named_entity="there",
                        properties=frozenset(dict().items()),
                        min=-1,
                        max=-1,
                        type="non_verb",
                        confidence=-1
                    ),
                    edgeLabel=Singleton(
                        id=-1,
                        named_entity="is",
                        properties=frozenset(dict().items()),
                        min=-1,
                        max=-1,
                        type="verb",
                        confidence=-1
                    ),
                    isNegated=False
                ))

                return


def create_cop(node, kernel, target_or_source):
    if target_or_source == 'target':
        # TODO: Ollie: Is it correct to say if target is None add to source otherwise add to target?
        if kernel.target is None:
            temp_prop = dict(copy(kernel.source.properties))
            target_json = json_dumps(node)
            target_dict = dict(json.loads(target_json))
            temp_prop['cop'] = target_dict
            new_source = Singleton(
                id=kernel.source.id,
                named_entity=kernel.source.named_entity,
                properties=temp_prop,
                min=kernel.source.min,
                max=kernel.source.max,
                type=kernel.source.type,
                confidence=kernel.source.confidence
            )
            kernel = Relationship(
                source=new_source,
                target=node,
                edgeLabel=kernel.edgeLabel,
                isNegated=kernel.isNegated
            )
        else:
            # if isinstance(node, SetOfSingletons):
            #     temp_prop = dict(copy(node.entities[0].properties))
            # else:
            temp_prop = dict(copy(node.properties))
            target_json = json_dumps(node)
            target_dict = dict(json.loads(target_json))
            temp_prop['cop'] = target_dict
            new_target = Singleton(
                id=kernel.target.id,
                named_entity=kernel.target.named_entity,
                properties=temp_prop,
                min=kernel.target.min,
                max=kernel.target.max,
                type=kernel.target.type,
                confidence=kernel.target.confidence
            )
            kernel = Relationship(
                source=kernel.source,
                target=new_target,
                edgeLabel=kernel.edgeLabel,
                isNegated=kernel.isNegated
            )
    else:
        temp_prop = dict(copy(kernel.source.properties))
        target_json = json_dumps(node)
        target_dict = dict(json.loads(target_json))
        temp_prop['cop'] = target_dict
        new_source = Singleton(
            id=kernel.source.id,
            named_entity=kernel.source.named_entity,
            properties=temp_prop,
            min=kernel.source.min,
            max=kernel.source.max,
            type=kernel.source.type,
            confidence=kernel.source.confidence
        )
        kernel = Relationship(
            source=new_source,
            target=kernel.target,
            edgeLabel=kernel.edgeLabel,
            isNegated=kernel.isNegated
        )
    return kernel


def create_sentence_obj(edges, nodes, transitive_verbs, negations) -> List[Sentence]:
    if len(edges) <= 0:
        create_existential(edges, nodes)
    # With graph created, make the 'Sentence' object
    kernel = None
    kernel, properties = assign_kernel(edges, kernel, negations, nodes, transitive_verbs)
    kernel_nodes = set()
    if kernel is not None:
        if kernel.source is not None:
            kernel_nodes.add(kernel.source)
        if kernel.target is not None:
            kernel_nodes.add(kernel.target)
    for edge in edges:
        # Source
        kernel, properties, kernel_nodes = add_to_properties(kernel, edge.source, 'source', kernel_nodes, properties)

        # Target
        kernel, properties, kernel_nodes = add_to_properties(kernel, edge.target, 'target', kernel_nodes, properties)

    edge_label_name = kernel.edgeLabel.named_entity
    el = lemmatize_verb(edge_label_name)

    # if kernel.edgeLabel.type != "verb":
    #     raise Exception("Relationships in the kernel should be verbs")

    ## TODO: this is done for future work, where a phrase might contain more than one sentence
    return [Sentence(
        kernel=Relationship(
            source=kernel.source,
            target=kernel.target,
            edgeLabel=replaceNamed(kernel.edgeLabel, el),
            isNegated=kernel.isNegated
        ),
        properties=dict(properties)
    )]


def lemmatize_verb(edge_label_name):
    stNLP = Services.getInstance().getStanzaSTNLP()
    lemmatizer = Services.getInstance().getWTLemmatizer()
    return " ".join(map(lambda y: y["lemma"], filter(lambda x: x["upos"] != "AUX", stNLP(
        lemmatizer.lemmatize(edge_label_name, 'v')).to_dict()[0])))


def add_to_properties(kernel, node, source_or_target, kernel_nodes, properties, type_key = None):
    lemma_node_edge_label_name = lemmatize_verb(node.named_entity)
    lemma_kernel_edge_label_name = lemmatize_verb(kernel.edgeLabel.named_entity)

    if node is not None and lemma_node_edge_label_name != lemma_kernel_edge_label_name:
        if type_key is None:
            type_key = get_node_type(node)
        if type_key == 'JJ' or type_key == 'JJS':
            kernel_nodes.add(node)
            kernel = create_cop(node, kernel, source_or_target)
        if type_key not in 'NEG' and type_key not in 'existential':
            if node.type == Grouping.MULTIINDIRECT:
                return kernel, properties, kernel_nodes
            elif node not in kernel_nodes and node not in properties[type_key]:
                properties[type_key].append(node)
    return kernel, properties, kernel_nodes


def get_node_type(node):
    return node.type if isinstance(node.type, str) else node.type.name


def assign_kernel(edges, kernel, negations, nodes, transitive_verbs):
    properties = defaultdict(list)
    for edge in edges:
        if edge.edgeLabel.type == "verb":
            edge_label = edge.edgeLabel
            if edge.isNegated:
                for name in negations:
                    if bool(re.search(rf"\b{re.escape(name)}\b", edge_label.named_entity)):
                        edge_label_name = edge_label.named_entity.replace(name, "")
                        edge_label_name = edge_label_name.strip()
                        edge_label = Singleton(edge_label.id, edge_label_name, edge_label.properties, edge_label.min,
                                               edge_label.max,
                                               edge_label.type, edge_label.confidence)
                        break

            # If not a transitive verb, remove target as target reflects direct object
            if len(Services.getInstance().lemmatize_sentence(edge.edgeLabel.named_entity).intersection(
                    transitive_verbs)) == 0:
                kernel = Relationship(
                    source=edge.source,
                    target=None,
                    edgeLabel=edge_label,
                    isNegated=edge.isNegated
                )
                break
            else:
                kernel = edge
                break
    # If kernel is none, look for existential
    if kernel is None:
        for edge in edges:
            if isinstance(edge.source, SetOfSingletons):
                for entity in edge.source.entities:
                    for prop in entity.properties:
                        if prop[1] == '∃':
                            kernel = edge
                            break
            else:
                for prop in edge.target.properties:
                    if prop[1] == '∃':
                        kernel = edge
                        break
    if kernel is None:
        n = len(edges)
        create_existential(edges, nodes)

        if n <= len(edges):
            kernel = edges[-1]
    return kernel, properties
