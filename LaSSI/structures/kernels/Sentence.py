__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2024, Oliver R. Fox"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"
__email__ = "ollie.fox5@gmail.com"
__status__ = "Production"

import json
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from typing import List

from LaSSI.external_services.Services import Services
from LaSSI.files.JSONDump import EnhancedJSONEncoder
from LaSSI.structures.internal_graph.EntityRelationship import Relationship, NodeEntryPoint, Singleton, SetOfSingletons, \
    Grouping


@dataclass(order=True, frozen=True, eq=True)
class Sentence:
    kernel: Relationship
    properties: dict = field(default_factory=lambda: {
        'time': List[NodeEntryPoint],
        'loc': List[NodeEntryPoint]
    })


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

def create_cop(edge, kernel, targetOrSource):
    if targetOrSource == 'target':
        temp_prop = dict(copy(kernel.target.properties))
        target_json = json.dumps(edge.target, cls=EnhancedJSONEncoder)
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
        target_json = json.dumps(edge.source, cls=EnhancedJSONEncoder)
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

def create_sentence_obj(edges, nodes, transitive_verbs, negations):
    if len(edges) <= 0:
        create_existential(edges, nodes)
    # With graph created, make the 'Sentence' object
    kernel = None
    properties = defaultdict(list)
    for edge in edges:
        if edge.edgeLabel.type == "verb":
            edge_label = edge.edgeLabel
            if edge.isNegated:
                for name in negations:
                    if name in edge_label.named_entity:
                        edge_label_name = edge_label.named_entity.replace(name, "")
                        edge_label_name = edge_label_name.strip()
                        edge_label = Singleton(edge_label.id, edge_label_name, edge_label.properties, edge_label.min, edge_label.max,
                                               edge_label.type, edge_label.confidence)
                        break

            # If not a transitive verb, remove target as target reflects direct object
            if len(Services.getInstance().lemmatize_sentence(edge.edgeLabel.named_entity).intersection(transitive_verbs)) == 0:
                kernel = Relationship(
                    source=edge.source,
                    target=None,
                    edgeLabel=edge_label,
                    isNegated=edge.isNegated
                )
            else:
                kernel = edge
    # If kernel is none, look for existential
    if kernel is None:
        for edge in edges:
            if isinstance(edge.source, SetOfSingletons):
                for entity in edge.target.entities:
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
    kernel_nodes = set()
    if kernel is not None:
        if kernel.source is not None:
            kernel_nodes.add(kernel.source)
        if kernel.target is not None:
            kernel_nodes.add(kernel.target)
    for edge in edges:
        type_key = edge.source.type if isinstance(edge.source.type, str) else edge.source.type.name

        # Source
        if type_key == 'JJ' or type_key == 'JJS':
            kernel = create_cop(edge, kernel, 'source')
            continue
        if type_key not in 'NEG':
            if edge.source.type == Grouping.MULTIINDIRECT:
                continue
            elif edge.source not in kernel_nodes and edge.source not in properties[type_key]:
                properties[type_key].append(edge.source)
        type_key = edge.target.type if isinstance(edge.target.type, str) else edge.target.type.name

        # Target
        if type_key == 'JJ' or type_key == 'JJS':
            kernel = create_cop(edge, kernel, 'target')
            continue
        if type_key not in 'NEG':
            if edge.target.type == Grouping.MULTIINDIRECT:
                continue
                # for target_edge in edge.target.entities:
                #     type_key2 = target_edge.type if isinstance(target_edge.type, str) else target_edge.type.name
                #     if type_key2 not in 'NEG':
                #         if target_edge not in kernel_nodes and target_edge not in properties[type_key2]:
                #             properties[type_key2].append(target_edge)
            elif edge.target not in kernel_nodes and edge.target not in properties[type_key]:
                properties[type_key].append(edge.target)

    stNLP = Services.getInstance().getStanzaSTNLP()
    lemmatizer = Services.getInstance().getWTLemmatizer()
    # from gsmtosimilarity.stanza_pipeline import StanzaService
    el = " ".join(map(lambda y: y["lemma"], filter(lambda x: x["upos"] != "AUX", stNLP(
        lemmatizer.lemmatize(kernel.edgeLabel.named_entity, 'v')).to_dict()[0])))
    return Sentence(
        kernel=Relationship(
            source=kernel.source,
            target=kernel.target,
            edgeLabel=replaceNamed(kernel.edgeLabel, el),
            isNegated=kernel.isNegated
        ),
        properties=dict(properties)
    )