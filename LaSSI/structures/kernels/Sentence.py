__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2024, Oliver R. Fox"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"
__email__ = "ollie.fox5@gmail.com"
__status__ = "Production"

import re
from collections import defaultdict
from copy import copy
from typing import Any, Tuple

from LaSSI.external_services.Services import Services
from LaSSI.ner.node_functions import create_existential_node, create_props_for_singleton
from LaSSI.ner.string_functions import lemmatize_verb, check_semi_modal
from LaSSI.structures.internal_graph.EntityRelationship import Relationship,  Singleton, SetOfSingletons, Grouping

# @dataclass(order=True, frozen=True, eq=True)
# class Sentence(Singleton):
#     kernel: Relationship
#     properties: dict = field(default_factory=lambda: {
#         'time': List[NodeEntryPoint],
#         'loc': List[NodeEntryPoint]
#     })
#
#     # @classmethod
#     # def from_dict(cls, c):
#     #     return cls(kernel=Relationship.from_dict(c.get('kernel')),
#     #                properties={k: [deserialize_NodeEntryPoint(x) for x in v] for k, v in c.get('properties').items()}
#     #                )
copula_types = {'JJ', 'JJS', 'RB'}


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
        if isinstance(node, SetOfSingletons) and len(node.entities) == 1:
            node = node.entities[0]

        # node_props = dict(node.properties)
        if is_kernel_in_props(node) or len(nodes) == 1:
            if node.type == 'verb':
                node_props = dict(node.properties)
                new_target = None
                if 'extra' in node_props:
                    # TODO: If node has an extra, make as target
                    new_target = Singleton(
                        id=-1,
                        named_entity=node_props['extra'],
                        properties=frozenset(dict().items()),
                        min=-1,
                        max=-1,
                        type="None",
                        confidence=-1
                    )
                    node_props.pop('extra')
                    node = Singleton.update_node_props(node, node_props)

                edges.append(Relationship(
                    source=create_existential_node(),
                    target=new_target,
                    edgeLabel=node,
                    isNegated=False
                ))
            else:
                edges.append(Relationship(
                    source=node,
                    target=create_existential_node(),
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
            temp_prop['cop'] = node

            new_source = Singleton(
                id=kernel.source.id,
                named_entity=kernel.source.named_entity,
                properties=create_props_for_singleton(temp_prop),
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
        else:
            # if isinstance(node, SetOfSingletons):
            #     temp_prop = dict(copy(node.entities[0].properties))
            # else:
            temp_prop = dict(copy(kernel.target.properties))
            temp_prop['cop'] = node

            if isinstance(kernel.target, Singleton):
                new_target = Singleton(
                    id=kernel.target.id,
                    named_entity=kernel.target.named_entity,
                    properties=create_props_for_singleton(temp_prop),
                    min=kernel.target.min,
                    max=kernel.target.max,
                    type=kernel.target.type,
                    confidence=kernel.target.confidence
                )
            else:
                new_target = kernel.target

            # Only add the copula if it differs from the target name
            kernel_target = new_target
            if isinstance(kernel.target, Singleton) and kernel.target is not None and kernel.target.named_entity == node.named_entity:
                kernel_target = kernel.target

            kernel = Relationship(
                source=kernel.source,
                target=kernel_target,
                edgeLabel=kernel.edgeLabel,
                isNegated=kernel.isNegated
            )
    else:
        temp_prop = dict(copy(kernel.source.properties))
        temp_prop['cop'] = node

        # Only add the copula if it differs from the target name
        kernel_target = kernel.target
        if isinstance(kernel.target, Singleton) and kernel.target is not None and kernel.target.named_entity == node.named_entity:
            kernel_target = None

        new_source = Singleton(
            id=kernel.source.id,
            named_entity=kernel.source.named_entity,
            properties=create_props_for_singleton(temp_prop),
            min=kernel.source.min,
            max=kernel.source.max,
            type=kernel.source.type,
            confidence=kernel.source.confidence
        )
        kernel = Relationship(
            source=new_source,
            target=kernel_target,
            edgeLabel=kernel.edgeLabel,
            isNegated=kernel.isNegated
        )
    return kernel

def create_sentence_obj(edges, nodes, negations, root_sentence_id, found_proposition_labels, node_functions, given_edge_to_loop):
    edge_to_loop = [False, None]

    root_node = nodes[root_sentence_id]
    if root_node.type == 'verb' and len(edges) <= 0:
        return create_edge_kernel(root_node), edge_to_loop
    elif len(edges) <= 0:
        return root_node, edge_to_loop

    # With graph created, make the 'Sentence' object
    kernel = None
    kernel = assign_kernel(edges, kernel, negations, nodes, root_sentence_id, found_proposition_labels)
    kernel_nodes = set()
    properties = defaultdict(list)

    if kernel is not None:
        # Add relevant nodes to kernel_nodes, and check if source or target should be a pronoun
        if kernel.source is not None:
            kernel, kernel_nodes = analyse_kernel_node(kernel, kernel_nodes, "source")
        if kernel.target is not None:
            kernel, kernel_nodes = analyse_kernel_node(kernel, kernel_nodes, "target")

    for edge in edges:
        # Source
        kernel, properties, kernel_nodes = add_to_properties(kernel, edge.source, 'source', kernel_nodes, properties, negations, node_functions)

        # Target
        kernel, properties, kernel_nodes = add_to_properties(kernel, edge.target, 'target', kernel_nodes, properties, negations, node_functions)

        # Add edge
        if edge.edgeLabel.named_entity in {'acl_relcl', 'nmod', 'nmod_poss'}:
            valid_nodes = [kernel.source, kernel.target]
            edge_kernel = Singleton(
                id=edge.edgeLabel.id,
                named_entity="",
                type="SENTENCE",
                min=node_functions.get_min_from_nodes(valid_nodes),
                max=node_functions.get_max_from_nodes(valid_nodes),
                confidence=1,
                kernel=Relationship(
                    source=edge.source,
                    target=edge.target,
                    edgeLabel=edge.edgeLabel,
                    isNegated=edge.isNegated
                ),
                properties=frozenset(dict()),
            )
            kernel, properties, kernel_nodes = add_to_properties(kernel, edge_kernel, 'edgeLabel', kernel_nodes, properties, negations, node_functions)

        # If we have an edge that is a verb and not already in the kernel nodes, use this as the "edge to loop", out the next iteration on the same root node
        if edge.edgeLabel.type == 'verb' and kernel.edgeLabel is not None and edge.edgeLabel is not None and kernel.edgeLabel.named_entity != edge.edgeLabel.named_entity and not is_node_in_kernel_nodes(edge.edgeLabel, kernel_nodes):
            edge_to_loop = [True, edge]

    # If we have a kernel returned from the previous loop, add this to the properties
    if len(given_edge_to_loop) > 2:
        # Check which order the kernel should be, should we add the NEW kernel as a property, or remain as the root kernel, based on the topological position of the nodes
        topological_node_id_positions = {x.id: idx for idx, x in enumerate(nodes.values())}
        kernel_id_to_check = get_kernel_top_id(kernel, topological_node_id_positions)
        returned_kernel_id_to_check = get_kernel_top_id(given_edge_to_loop[2].kernel, topological_node_id_positions)

        if topological_node_id_positions[kernel_id_to_check] < topological_node_id_positions[returned_kernel_id_to_check]:
            kernel, properties, kernel_nodes = add_to_properties(
                given_edge_to_loop[2].kernel,
                node_functions.convert_relationship_to_sentence(root_sentence_id, kernel),
                'target', kernel_nodes,  properties, negations, node_functions
            )
        else:
            kernel, properties, kernel_nodes = add_to_properties(
                kernel, given_edge_to_loop[2] , 'target', kernel_nodes, properties, negations, node_functions
            )

    edge_label = replaceNamed(kernel.edgeLabel, lemmatize_verb(kernel.edgeLabel.named_entity)) if kernel.edgeLabel is not None else None

    properties_to_keep = defaultdict()
    new_kernel = None
    for key in properties:
        if key == 'verb': # TODO: Check for "NOT" prop / SetOfSingletons
            verbs_to_keep = []
            for node in properties['verb']:
                if not 'mark' in node.properties and root_sentence_id == node.id:
                    new_kernel = node
                else:
                    verbs_to_keep.append(node)
            properties_to_keep[key] = verbs_to_keep
        else:
            properties_to_keep[key] = properties[key]

    final_kernel = node_functions.convert_relationship_to_sentence(root_sentence_id, kernel, edge_label, properties_to_keep)

    if new_kernel is not None:
        valid_nodes = node_functions.get_valid_nodes([kernel.source, kernel.target])
        final_kernel = Singleton(
            id=root_sentence_id,
            named_entity="",
            type="SENTENCE",
            min=node_functions.get_min_from_nodes(valid_nodes),
            max=node_functions.get_max_from_nodes(valid_nodes),
            confidence=1,
            kernel=Relationship(
                source=create_existential_node(),
                target=final_kernel,
                edgeLabel=new_kernel,
                isNegated=new_kernel.isNegated, # TODO: Check this is correctly negated
            ),
            properties=create_props_for_singleton(properties_to_keep),
        )
        if edge_to_loop[0]:
            edge_to_loop.append(final_kernel)
        return final_kernel, edge_to_loop
    else:
        if edge_to_loop[0]:
            edge_to_loop.append(final_kernel)
        return final_kernel, edge_to_loop


def get_kernel_top_id(kernel, topological_node_id_positions):
    return kernel.source.id if kernel.source.id in topological_node_id_positions else kernel.target.id if kernel.target is not None and kernel.target.id in topological_node_id_positions else None


def analyse_kernel_node(kernel, kernel_nodes, kernel_node_type):
    if kernel_node_type == 'source':
        kernel_node = kernel.source
    else:
        kernel_node = kernel.target

    kernel_nodes = add_to_kernel_nodes(kernel_node, kernel_nodes)

    if isinstance(kernel_node, Singleton):
        # If is 'det' or adjective AND a pronoun of any type
        if ('det' in dict(kernel_node.properties) and dict(kernel_node.properties)['det'] == 'det' or kernel_node.type == 'JJ' or kernel_node.type == 'JJS') and (len({kernel_node.named_entity.lower()}.intersection(Services.getInstance().getParmenides().getPronouns())) != 0):
            kernel_node = Singleton(
                id=kernel_node.id,
                named_entity=kernel_node.named_entity,
                properties=kernel_node.properties,
                min=kernel_node.min,
                max=kernel_node.max,
                type="PRON",
                confidence=kernel_node.confidence
            )
            if kernel_node_type == 'source':
                kernel = Relationship(
                    source=kernel_node,
                    target=kernel.target,
                    edgeLabel=kernel.edgeLabel,
                    isNegated=kernel.isNegated
                )
            else:
                kernel = Relationship(
                    source=kernel.source,
                    target=kernel_node,
                    edgeLabel=kernel.edgeLabel,
                    isNegated=kernel.isNegated
                )

    return kernel, kernel_nodes


def add_to_kernel_nodes(node, kernel_nodes):
    # if is_node_in_kernel_nodes(node, kernel_nodes):
    #     return kernel_nodes

    if isinstance(node, SetOfSingletons):
        kernel_nodes.add(node)
        for entity in node.entities:
            add_to_kernel_nodes(entity, kernel_nodes)
    else:
        # Check if properties has any Singleton's and add to kernel nodes also
        for key in dict(node.properties):
            properties_key_ = dict(node.properties)[key]
            if not isinstance(properties_key_, str):
                if isinstance(properties_key_, Singleton):
                    add_to_kernel_nodes(properties_key_, kernel_nodes)
                else:
                    for prop_node in properties_key_:
                        if isinstance(prop_node, Singleton):
                            add_to_kernel_nodes(prop_node, kernel_nodes)
                        elif isinstance(prop_node, SetOfSingletons):
                            kernel_nodes.add(node)
                            for prop_entity in prop_node.entities:
                                add_to_kernel_nodes(prop_entity, kernel_nodes)

        if node.kernel is not None:
            if node.kernel.edgeLabel is not None:
                add_to_kernel_nodes(node.kernel.edgeLabel, kernel_nodes)
            if node.kernel.source is not None:
                add_to_kernel_nodes(node.kernel.source, kernel_nodes)
            if node.kernel.target is not None:
                add_to_kernel_nodes(node.kernel.target, kernel_nodes)
        else:
            kernel_nodes.add(node)

    return kernel_nodes


def add_to_properties(kernel, node, source_or_target, kernel_nodes, properties, negations, node_functions, type_key = None):
    lemma_kernel_edge_label_name = lemmatize_verb(kernel.edgeLabel.named_entity) if kernel.edgeLabel is not None else ""

    # Check if action matched edge label, and if it contains a negation, negate the kernel
    if isinstance(node, SetOfSingletons) and node.type == Grouping.NOT:
        is_negated = True
        node_props = dict(node.entities[0].properties)
        if 'action' in node_props and len(node_props['action']) > 0:
            # Get label from action prop
            lemma_node_edge_label_name = lemmatize_verb(node_props['action'])

            # Remove negation from label
            query_words = lemma_node_edge_label_name.split()
            result_words = [word for word in query_words if word.lower() not in negations]
            lemma_node_edge_label_name = ' '.join(result_words)

            # If lemmatized action == kernel edge, then negate
            if lemma_node_edge_label_name == lemma_kernel_edge_label_name:
                kernel = Relationship(
                    source=kernel.source,
                    target=kernel.target,
                    edgeLabel=kernel.edgeLabel,
                    isNegated=is_negated
                )
                return kernel, properties, kernel_nodes
        else:
            lemma_node_edge_label_name = node.entities[0].named_entity
    elif isinstance(node, Singleton): # Get label name from action if present, otherwise just the node name
        node_props = dict(node.properties)
        if 'action' in node_props and len(node_props['action']) > 0:
            # Get label from action prop
            lemma_node_edge_label_name = lemmatize_verb(node_props['action'])
        else:
            lemma_node_edge_label_name = lemmatize_verb(node.named_entity)
    else:
        lemma_node_edge_label_name = None

    # Check for a SetOfSingletons, or if the node name or action label is not equal to the kernel edge label
    if (isinstance(node, SetOfSingletons)) or (node is not None and lemma_node_edge_label_name != lemma_kernel_edge_label_name):
        if type_key is None:
            if source_or_target == 'edgeLabel':
                type_key = node.kernel.edgeLabel.named_entity
            else:
                type_key = node_functions.get_node_type(node)
        if (type_key in copula_types) and not is_node_in_kernel_nodes(node, kernel_nodes):
            kernel_nodes = add_to_kernel_nodes(node, kernel_nodes)
            kernel = create_cop(node, kernel, source_or_target)
        if 'NEG' not in type_key and 'NOT' not in type_key and 'existential' not in type_key:
            if (
                    (node.type == Grouping.MULTIINDIRECT) or
                    (isinstance(node, Singleton) and (node.named_entity == "but" or node.named_entity == "and")) or
                    (isinstance(node, SetOfSingletons) and node.type == Grouping.AND and 'NEG' in node_functions.get_node_type(node.entities[0]) and len(node.entities) == 1)
            ):
                return kernel, properties, kernel_nodes
            elif not is_node_in_kernel_nodes(node, kernel_nodes) and node not in properties[type_key]:
                if isinstance(node, SetOfSingletons):
                    for entity in node.entities:
                        if not is_node_in_kernel_nodes(entity, kernel_nodes):
                            kernel_nodes = add_to_kernel_nodes(entity, kernel_nodes)
                            properties[type_key].append(entity)
                else:
                    kernel_nodes = add_to_kernel_nodes(node, kernel_nodes)
                    properties[type_key].append(node)
    return kernel, properties, kernel_nodes

# TODO: Is this enough information to determine? As ID might differ but have all same properties etc.
#  min and max might differ despite being the same node (i.e. a MULTIINDIRECT might have larger min max)?
def is_node_in_kernel_nodes(check_node, kernel_nodes):
    for kernel_node in kernel_nodes:
        if isinstance(check_node, Singleton) and isinstance(kernel_node, Singleton) and check_node.kernel is None:
            if (check_node.named_entity == kernel_node.named_entity and check_node.id == kernel_node.id
                    #and check_node.type == kernel_node.type and check_node.min == kernel_node.min and check_node.max == kernel_node.max
                #check_node.named_entity == kernel_node.named_entity and
            ) or (check_node.type == 'verb' and lemmatize_verb(check_node.named_entity) == lemmatize_verb(kernel_node.named_entity) and check_node.id == kernel_node.id):
                return True
        elif isinstance(check_node, SetOfSingletons) and isinstance(kernel_node, SetOfSingletons):
            if check_node.entities == kernel_node.entities and check_node.type == kernel_node.type and check_node.min == kernel_node.min and check_node.max == kernel_node.max:
                return True
        elif isinstance(check_node, Singleton) and isinstance(kernel_node, Singleton) and check_node.kernel is not None and kernel_node.kernel is not None:
            if (
                    ((check_node.kernel.source is not None and kernel_node.kernel.source is not None and check_node.kernel.source.id == kernel_node.kernel.source.id) or (check_node.kernel.source is None and kernel_node.kernel.source is None))
                    and
                    ((check_node.kernel.target is not None and kernel_node.kernel.target is not None and check_node.kernel.target.id == kernel_node.kernel.target.id) or (check_node.kernel.target is None and kernel_node.kernel.target is None))
                    and
                    ((check_node.kernel.edgeLabel is not None and kernel_node.kernel.edgeLabel is not None and check_node.kernel.edgeLabel.named_entity == kernel_node.kernel.edgeLabel.named_entity) or (check_node.kernel.edgeLabel is None and kernel_node.kernel.edgeLabel is None))
            ):
                return True

    return False


def assign_kernel(edges, kernel, negations, nodes, root_sentence_id, found_proposition_labels):
    chosen_edge = None
    for edge in edges:
        if edge.edgeLabel.type == "verb" and (root_sentence_id in found_proposition_labels or edge.edgeLabel.named_entity not in found_proposition_labels.values()) and edge.source.id == root_sentence_id:
            chosen_edge = edge
            break

    for edge in edges:
        if ((((edge.edgeLabel.type == "verb" or edge.source.type == "verb") and chosen_edge is None) or (chosen_edge is not None and chosen_edge == edge))
                and (root_sentence_id in found_proposition_labels or edge.edgeLabel.named_entity not in found_proposition_labels.values())):
            # edge_label = edge.edgeLabel if edge.edgeLabel.type == "verb" and (root_sentence_id in found_proposition_labels or edge.edgeLabel.named_entity not in found_proposition_labels.values()) else edge.source  # If the source is a verb, assign it to the edge label
            edge_label = edge.edgeLabel if edge.edgeLabel.type == "verb" else edge.source  # If the source is a verb, assign it to the edge label

            # If not semi-modal AND not in nodes then remove source
            edge_source = create_existential_node() if (isinstance(edge.source, Singleton) and not check_semi_modal(edge.source.named_entity) and len([x for x in nodes.keys() if edge.source.id == x]) == 0 and edge.source.type != 'existential') or edge.source == edge_label else edge.source
            edge_target = edge.target

            if edge.isNegated:
                for name in negations:
                    # If edge label contains negations (no, not), remove them
                    if bool(re.search(rf"\b{re.escape(name)}\b", edge_label.named_entity)):
                        edge_label_name = edge_label.named_entity.replace(name, "")
                        edge_label_name = edge_label_name.strip()
                        edge_label = Singleton(
                            edge_label.id,
                            edge_label_name,
                            edge_label.properties,
                            edge_label.min,
                            edge_label.max,
                            edge_label.type,
                            edge_label.confidence
                        )
                        break

            # If not a transitive verb, remove target as target reflects direct object
            if len({lemmatize_verb(x) for x in Services.getInstance().lemmatize_sentence(edge_label.named_entity)}.intersection(Services.getInstance().getParmenides().getTransitiveVerbs())) == 0:
                kernel = Relationship(
                    source=edge_source,
                    target=None,
                    edgeLabel=edge_label,
                    isNegated=edge.isNegated
                )
                break
            else:
                kernel = Relationship(
                    source=edge_source,
                    target=edge_target,
                    edgeLabel=edge_label,
                    isNegated=edge.isNegated
                )
                break

    # If kernel is none, look for existential
    if kernel is None:
        for edge in edges:
            # Look in source
            kernel = find_existential_in_properties(edge.source)
            if kernel is not None:
                break

            # Look in target
            kernel = find_existential_in_properties(edge.target)
            if kernel is not None:
                break

    # If we cannot find existential, create it instead
    if kernel is None:
        n = len(edges)
        create_existential(edges, nodes)

        # If we cannot create it, just use the edge (unchanged)
        if n <= len(edges):
            kernel = edges[-1]

    # Check if source or target have "case" property, if so remove it (to be added to properties of the kernel later)
    # TODO: Check within SetOfSingletons, if any elements have a case, then add to properties...
    if case_in_props(Singleton.get_props(kernel.source)):
        kernel = Relationship(
            source=create_existential_node(),
            target=kernel.target,
            edgeLabel=kernel.edgeLabel,
            isNegated=kernel.isNegated
        )
    if case_in_props(Singleton.get_props(kernel.target)):
        kernel = Relationship(
            source=kernel.source,
            target=None,
            edgeLabel=kernel.edgeLabel,
            isNegated=kernel.isNegated
        )
    if kernel.edgeLabel.type != "verb":
        kernel = Relationship(
            source=kernel.source,
            target=kernel.target,
            edgeLabel=None,
            isNegated=kernel.isNegated
        )

    return kernel


def find_existential_in_properties(node):
    new_kernel = None
    if isinstance(node, SetOfSingletons):
        for entity in node.entities:
            find_existential_in_properties(entity)
    else:
        for prop in node.properties:
            if prop[1] == '∃':
                new_kernel = node
                break
    return new_kernel


def case_in_props(node_props, return_props=False):
    if node_props is None:
        return False

    # Ignore "by" as passive sentence: https://www.uc.utoronto.ca/passive-voice
    # Ignore "of" and "'s" as "possessive": https://en.m.wikipedia.org/wiki/English_possessive
    ignore_cases = ['by', "'s", 'of']
    found_cases = []

    for key in node_props:
        if key == 'case':
            return True
        try:
            case_position = float(key)
            if node_props[key] not in ignore_cases:
                if not return_props:
                    return True
                else:
                    found_cases.append(node_props[key])
        except ValueError:
            continue

    if not return_props:
        return False
    else:
        return found_cases

def create_edge_kernel(node):
    return Singleton(
        id=node.id,
        named_entity="",
        type="SENTENCE",
        min=node.min,
        max=node.max,
        confidence=1,
        kernel=Relationship(
            source=create_existential_node(),
            target=None,
            edgeLabel=node,
            isNegated=False
        ),
        properties=frozenset(), # TODO: Should this be empty?
    )

def is_kernel_in_props(node, check_jj=True):
    if isinstance(node, SetOfSingletons):
        # If the Set has a true root, return, otherwise check children for root
        if node.root:
            return node.root
        else:
            for entity in node.entities:
                return is_kernel_in_props(entity)

    if isinstance(node, Singleton):
        node_props = dict(node.properties)
        return (('kernel' in node_props or 'root' in node_props) and (
                    'JJ' not in node.type and check_jj or not check_jj)) or 'verb' in node.type
    elif isinstance(node, dict):
        node_props = node['properties']
        return 'kernel' in node_props or 'root' in node_props

    # TODO: Do we need to check if the JJ is/not a verb?
    # ('JJ' not in x.type or ('JJ' in x.type and self.is_label_verb(x.named_entity))))