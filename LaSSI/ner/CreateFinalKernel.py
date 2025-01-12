import itertools
import re
import string

import numpy
from collections import defaultdict
from LaSSI.external_services.Services import Services
from LaSSI.ner.MergeSetOfSingletons import merge_properties
from LaSSI.ner.node_functions import create_props_for_singleton, create_existential_node
from LaSSI.ner.string_functions import is_label_verb, lemmatize_verb, check_semi_modal, has_auxiliary
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, SetOfSingletons, Relationship
from LaSSI.structures.kernels.Sentence import is_kernel_in_props, create_edge_kernel, create_existential, \
    is_node_in_kernel_nodes, create_sentence_obj, case_in_props


class CreateFinalKernel:
    def __init__(self, nodes, gsm_json, edges, negations, node_functions):
        self.services = Services.getInstance()
        self.existentials = self.services.getExistentials()
        self.negations = negations
        self.edges = edges
        self.nodes = nodes
        self.gsm_json = gsm_json
        self.node_functions = node_functions

    def constructSentence(self) -> Singleton:
        from LaSSI.structures.kernels.Sentence import create_sentence_obj

        found_proposition_labels = {}
        true_targets = set()
        new_edges = []
        position_pairs = self.get_position_pairs()

        if len(self.edges) > 0:
            prototypical_prepositions = Services.getInstance().getParmenides().getPrototypicalPrepositions()

            edge_labels = defaultdict(set)
            for edge in self.edges:
                edge_labels[(edge.source.id, edge.target.id)].add(edge.edgeLabel.named_entity)
            edge_labels = {key: "dep" in value and len(value) > 1 for key, value in edge_labels.items()}

            for edge in self.edges:
                if edge.edgeLabel.named_entity == 'dep' and edge_labels[(edge.source.id, edge.target.id)]:
                    continue
                new_edge = None
                found_prototypical_prepositions = False
                first_word = edge.edgeLabel.named_entity.split()[0]
                for p in prototypical_prepositions:
                    # Check if prototypical preposition is in edge label but NOT just the edge label (i.e. "to" in "to steal" = TRUE, "like" in "like" = FALSE) AND whether the first word is NOT a verb
                    if (
                            (not is_label_verb(first_word) and re.search(r"\b" + p + r"\b",
                                                                         edge.edgeLabel.named_entity) and p != edge.edgeLabel.named_entity and not case_in_props(
                                dict(edge.target.properties)))
                            or
                            (edge.edgeLabel.named_entity.endswith(
                                'ing') and not self.node_functions.check_node_coordinations_for_auxiliary(edge,
                                                                                                          self.edges) and edge.source.type != "existential")
                    ):
                        found_prototypical_prepositions = True

                        # Add root property to target
                        self.nodes[edge.target.id] = Singleton.add_root_property(edge.target)
                        new_edge = Relationship(
                            source=edge.source,
                            target=self.nodes[edge.target.id],
                            edgeLabel=edge.edgeLabel,
                            isNegated=edge.isNegated
                        )

                        # re.sub(r"\b"+p+r"\b", "", edge.edgeLabel.named_entity).strip()
                        found_proposition_labels[edge.target.id] = edge.edgeLabel.named_entity
                        break

                # If the edge is a verb and source is a 'root', remove 'root' from the target node of the edge
                if not found_prototypical_prepositions and is_label_verb(
                        edge.edgeLabel.named_entity) and is_kernel_in_props(
                        edge.source) and edge.target.type != 'existential':
                    self.nodes[edge.target.id] = Singleton.strip_root_properties(edge.target)
                    new_edge = Relationship(
                        source=edge.source,
                        target=self.nodes[edge.target.id],
                        edgeLabel=edge.edgeLabel,
                        isNegated=edge.isNegated
                    )

                if new_edge is None:
                    new_edge = edge
                    if not is_kernel_in_props(edge.target):
                        true_targets.add(edge.target.id)
                new_edges.append(new_edge)
                new_edges = [Relationship.from_nodes(r, self.nodes) for r in new_edges]

        filtered_nodes = set()
        filtered_top_node_ids = set()
        remove_set = set()
        # new_edges = list(set(new_edges)) # Remove duplicate edges

        # Loop over every source and target for every edge
        for edge_node in itertools.chain.from_iterable(map(lambda x: [x.source, x.target], new_edges)):
            # Check if edge node is not None, in true targets, is a root or existential
            if ((edge_node is not None and edge_node.id not in true_targets) and
                    (is_kernel_in_props(edge_node))
                    # or
                    # (isinstance(edge_node, Singleton) and edge_node.type == 'existential')
            ):
                filtered_nodes.add(edge_node.id)
                if isinstance(edge_node, SetOfSingletons):
                    for entity in edge_node.entities:
                        remove_set.add(entity.id)

            if edge_node is None or edge_node.id in filtered_top_node_ids:
                continue
            filtered_top_node_ids.add(edge_node.id)

        # IDs of root nodes in topological order
        filtered_nodes = filtered_nodes - remove_set
        filtered_top_node_ids = [x.id for x in self.nodes.values() if
                                 x.id in filtered_nodes]  # Filter node IDs in topological order based on self.nodes order

        if len(filtered_top_node_ids) == 0:
            filtered_top_node_ids = [list(self.nodes)[-1]] if len(self.nodes) == 1 else [
                self.node_functions.get_node_id(self.nodes[x].id) for x in self.nodes if
                is_kernel_in_props(self.nodes[x], False)]  # Get last node
            filtered_top_node_ids = list(map(int, numpy.unique(filtered_top_node_ids)))

        used_edges = set()
        for node_id in filtered_top_node_ids:
            edge_to_loop = [True, None]  # Represents: [should_loop?, edge_to_use_for_kernel, returned_kernel_from_last_loop]
            while edge_to_loop[0]:
                descendant_node_ids = self.node_functions.node_bfs(new_edges, node_id)

                # Ensure the d_nodes are in the edge, and not in used_edges, unless we have proposition labels
                filtered_edges = [
                    x for x in new_edges if
                    ((x.source.id in descendant_node_ids and x.target.id in descendant_node_ids) or (
                                x.target.id in found_proposition_labels))
                    and
                    ((x.source.id, x.target.id) not in used_edges or (
                                (x.source.id, x.target.id) in used_edges and len(found_proposition_labels) > 0))
                    and
                    (not (x.target.id == node_id and x.target.type == 'verb'))
                ]

                # If we have an edge from the previous iteration use this as our edges
                if edge_to_loop[1] is not None:
                    filtered_edges = [edge_to_loop[1]]

                used_edges = set(map(lambda y: (y.source.id, y.target.id), filtered_edges))
                descendant_nodes = {key: x for key, x in self.nodes.items() if x.id in descendant_node_ids}
                kernel, edge_to_loop = create_sentence_obj(filtered_edges, descendant_nodes, self.negations, node_id,
                                                           found_proposition_labels, self.node_functions, edge_to_loop)
                kernel = self.kernel_post_processing(kernel, position_pairs)

                # Only check for empty kernel if more than one root node, as if there is only 1 root node we need something, even if it is empty...
                if len(filtered_top_node_ids) > 1:
                    kernel = self.check_if_empty_kernel(kernel)  # Check we do not have be(?, ?) as a kernel
                    if kernel is not None:
                        self.nodes[node_id] = kernel
                else:
                    self.nodes[node_id] = kernel

                # Remove 'root' property from nodes
                self.nodes = {k: Singleton.strip_root_properties(v) if v.id in descendant_node_ids else v for k, v in
                              self.nodes.items()}

                # Re-instantiate new node properties across all relationships and new 'kernel' node
                new_edges = [Relationship.from_nodes(r, self.nodes) for r in new_edges]

                # If this current "kernel" we return is none, then pop so we use the previous kernel, if we have more than one root node
                if ((isinstance(kernel, Singleton) and kernel.kernel is None) or (kernel is None)) and len(
                        filtered_top_node_ids) > 1 and node_id == filtered_top_node_ids[-1]:
                    filtered_top_node_ids.pop()

        # Return the last node ('highest' kernel)
        final_kernel = self.nodes[filtered_top_node_ids[-1]]

        # If "final kernel" does not have a kernel
        actioned_node = self.find_actioned_node_in_kernel(final_kernel)
        if not hasattr(final_kernel, 'kernel') or final_kernel.kernel is None or (
                actioned_node and final_kernel.kernel.edgeLabel.named_entity == 'be'):
            if final_kernel.type == 'verb':
                # If a Singleton of type verb, make this an edge with no source or target
                final_kernel = create_edge_kernel(final_kernel)
                final_kernel = self.kernel_post_processing(final_kernel, position_pairs)
            else:
                edges = []
                nodes = {final_kernel.id: final_kernel}
                if not hasattr(final_kernel, 'properties') or not actioned_node:
                    create_existential(edges, nodes)
                    final_kernel, edge_to_loop = create_sentence_obj(edges, nodes, self.negations, final_kernel.id, {}, self.node_functions, edge_to_loop)
                    final_kernel = self.kernel_post_processing(final_kernel, position_pairs)
                else:
                    # Get label from action prop
                    final_kernel_props = dict(actioned_node.properties)
                    actioned = 'actioned' in final_kernel_props
                    lemma_node_edge_label_name = lemmatize_verb(
                        final_kernel_props['actioned']) if actioned else lemmatize_verb(final_kernel_props['action'])

                    # Remove negation from label
                    query_words = lemma_node_edge_label_name.split()
                    result_words = [word for word in query_words if word.lower() not in self.negations]
                    refactored_lemma_node_edge_label_name = ' '.join(result_words)

                    final_kernel = Singleton(
                        id=final_kernel.id,
                        named_entity='',
                        type='SENTENCE',
                        min=final_kernel.min,
                        max=final_kernel.max,
                        confidence=1,
                        kernel=Relationship(
                            source=actioned_node if not actioned else create_existential_node(),
                            target=actioned_node if actioned else None,
                            edgeLabel=Singleton(
                                id=-1,
                                named_entity=refactored_lemma_node_edge_label_name,
                                type='verb',
                                min=final_kernel.min,
                                max=final_kernel.max,
                                confidence=1,
                                kernel=None,
                                properties=frozenset(),  # TODO: Keep properties
                            ),
                            isNegated=lemma_node_edge_label_name != refactored_lemma_node_edge_label_name,
                        ),
                        properties=final_kernel.properties,
                    )
                    final_kernel = self.kernel_post_processing(final_kernel, position_pairs)

        final_kernel = self.remove_duplicate_properties(final_kernel)
        # final_kernel = self.check_if_empty_kernel(final_kernel)

        print(f"{final_kernel.to_string()}\n")
        return final_kernel

    def find_actioned_node_in_kernel(self, kernel):
        if isinstance(kernel, Singleton):
            if kernel.kernel is not None:
                props = Singleton.get_props(kernel.kernel.source)
                if kernel.kernel.source is not None and 'actioned' in props:
                    return kernel.kernel.source

                props = Singleton.get_props(kernel.kernel.target)
                if kernel.kernel.target is not None and 'actioned' in props:
                    return kernel.kernel.target
            elif kernel is not None:
                if 'actioned' in dict(kernel.properties):
                    return kernel
        return False

    def get_position_pairs(self):
        position_pairs = {}
        for edge in self.edges:
            source_pos = self.node_functions.get_min_position(edge.source)
            target_pos = self.node_functions.get_min_position(edge.target)
            if target_pos > source_pos:
                if edge.source.id in position_pairs:
                    if target_pos < position_pairs[edge.source.id]:
                        position_pairs[edge.source.id] = target_pos
                else:
                    position_pairs[edge.source.id] = target_pos
        return position_pairs

    # Check if final kernel is "empty" be(?, ?) and use the properties of 'SENTENCE'
    def check_if_empty_kernel(self, kernel):
        properties_to_keep = dict()
        new_kernel = None
        if (
                isinstance(kernel,
                           Singleton) and kernel.kernel is not None and kernel.kernel.edgeLabel is not None and kernel.kernel.edgeLabel.named_entity == "be"
                and
                (
                        ((kernel.kernel.source is not None and kernel.kernel.source.type == 'existential') and (
                                kernel.kernel.target is not None and kernel.kernel.target.type == 'existential'))
                        or
                        ((kernel.kernel.source is None) and (kernel.kernel.target is None))
                )
        ):
            node_props = dict(kernel.properties)
            if node_props is not None and 'SENTENCE' in node_props:
                for key in node_props:
                    if key == 'SENTENCE':
                        new_kernel = node_props['SENTENCE'][0] # TODO: Safe to use 0th element?
                        new_kernel = self.check_if_empty_kernel(new_kernel)
                    else:
                        properties_to_keep[key] = node_props[key]

        if new_kernel is not None:
            if len(properties_to_keep) > 0:
                return Singleton.update_node_props(new_kernel, properties_to_keep)
            else:
                return new_kernel
        else:
            return kernel

    # Rewrite kernel if positions are not correct
    def kernel_post_processing(self, kernel, position_pairs):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return kernel

        # Check kernel positions
        properties_to_keep = dict(kernel.properties)
        new_target = kernel.kernel.target
        if kernel.id in position_pairs and kernel.kernel.edgeLabel is not None and check_semi_modal(
                kernel.kernel.edgeLabel.named_entity):
            position_value = position_pairs[kernel.id]
            if 'SENTENCE' in dict(kernel.properties):
                properties_to_keep['SENTENCE'] = []
                for sentence_elm in list(dict(kernel.properties)['SENTENCE']):
                    if int(float(dict(sentence_elm.kernel.edgeLabel.properties)['pos'])) == position_value:
                        new_target = sentence_elm
                    else:
                        properties_to_keep['SENTENCE'].append(sentence_elm)

            return Singleton(
                id=kernel.id,
                named_entity="",
                type="SENTENCE",
                min=kernel.min,
                max=kernel.max,
                confidence=kernel.confidence,
                kernel=Relationship(
                    source=kernel.kernel.source,
                    target=new_target,
                    edgeLabel=kernel.kernel.edgeLabel,
                    isNegated=kernel.kernel.isNegated
                ),
                properties=create_props_for_singleton(properties_to_keep),
            )
        elif kernel.kernel.edgeLabel is None and kernel.kernel.source is not None and kernel.kernel.target is not None and (
                (kernel.kernel.source.type.startswith('JJ') and kernel.kernel.target.type == 'ENTITY') or (
                kernel.kernel.source.type.startswith('JJ') and kernel.kernel.target.type == 'ENTITY')):
            new_edges = []
            root_id = None
            if kernel.kernel.source.type.startswith('JJ') and kernel.kernel.target.type == 'ENTITY':
                node_props = merge_properties(dict(kernel.kernel.target.properties),
                                              {kernel.kernel.source.type: kernel.kernel.source})
                root_id = kernel.kernel.target.id
                self.nodes[root_id] = Singleton.update_node_props(kernel.kernel.target, node_props)
                create_existential(new_edges, {0: self.nodes[kernel.kernel.target.id]})
            elif kernel.kernel.target.type.startswith('JJ') and kernel.kernel.source.type == 'ENTITY':
                node_props = merge_properties(dict(kernel.kernel.source.properties),
                                              {kernel.kernel.target.type: kernel.kernel.target})
                root_id = kernel.kernel.source.id
                self.nodes[root_id] = Singleton.update_node_props(kernel.kernel.source, node_props)
                create_existential(new_edges, {0: self.nodes[kernel.kernel.target.id]})

            kernel, edge_to_loop = create_sentence_obj(new_edges, self.nodes, self.negations, root_id, {},
                                                       self.node_functions, [False, None])
            kernel = self.kernel_post_processing(kernel, position_pairs)
        return kernel

    def remove_duplicate_properties(self, kernel, kernel_nodes=None):
        if kernel.kernel is None:
            return kernel

        properties_to_keep = defaultdict(list)

        if kernel_nodes is None:
            kernel_nodes = set()
        kernel_nodes = self.add_to_kernel_nodes(kernel, kernel_nodes)

        # Add 'nmod' source and target to kernel nodes, so duplicate nodes are not added to properties
        for key in dict(kernel.properties):
            properties_key_ = dict(kernel.properties)[key]
            if isinstance(properties_key_, str):
                continue
            else:
                for node in properties_key_:
                    if key == 'nmod':
                        properties_to_keep[key].append(self.remove_duplicate_properties(node, kernel_nodes))
                        kernel_nodes = self.add_to_kernel_nodes(node, kernel_nodes)

        # Check if empty kernel is in properties and remove, recursively iterate through kernels to remove duplicate properties
        for key in dict(kernel.properties):
            properties_key_ = dict(kernel.properties)[key]
            if isinstance(properties_key_, str):
                continue
            else:
                for node in properties_key_:
                    if key == 'SENTENCE':
                        if self.check_if_empty_kernel(node) is not None:
                            # If given property is `be(? OR in kernel_nodes, ? OR in kernel_nodes)`, then do not add as property as it is redundant
                            if (hasattr(node, 'kernel') and not (node.kernel.edgeLabel.named_entity == "be" and (
                                    (
                                            (kernel_nodes is not None and node.kernel.source.type != 'existential' and node.kernel.source in kernel_nodes)
                                            and (node.kernel.target is not None and node.kernel.target.type == 'existential')
                                    )
                                    or
                                    (
                                            (kernel_nodes is not None and node.kernel.target.type != 'existential' and node.kernel.target in kernel_nodes)
                                            and (node.kernel.source is not None and node.kernel.source.type == 'existential')
                                    )
                            ))) or kernel_nodes is None or not hasattr(node, 'kernel'):
                                if not is_node_in_kernel_nodes(node, kernel_nodes):
                                    properties_to_keep[key].append(self.remove_duplicate_properties(node, kernel_nodes))
                                else:
                                    self.remove_duplicate_properties(node, kernel_nodes)
                    elif not is_node_in_kernel_nodes(node, kernel_nodes):
                        if hasattr(node, 'properties'):
                            inner_properties_to_keep = dict()
                            for inner_key in dict(node.properties):
                                value = dict(node.properties)[inner_key]
                                if (value in string.punctuation) or (kernel.kernel.edgeLabel is not None and isinstance(value, str) and not re.search(
                                        r"\b" + value + r"\b", kernel.kernel.edgeLabel.named_entity)) or not isinstance(
                                        value, str) or kernel.kernel.edgeLabel is None:
                                    inner_properties_to_keep[inner_key] = value
                            properties_to_keep[key].append(Singleton.update_node_props(node, inner_properties_to_keep))
                        else:
                            properties_to_keep[key].append(node)

        # Check if we have duplicate kernels now they are all added, and keep most relevant one (i.e. two equal kernels but only one has properties)
        if 'SENTENCE' in properties_to_keep:
            sentence_properties = properties_to_keep['SENTENCE']
            if len(sentence_properties) > 1:  # Check for more than one SENTENCE in props
                equal_kernels = [sentence for sentence in sentence_properties if
                                 sentence.kernel == sentence_properties[0].kernel]
                if len(equal_kernels) > 1:  # Check we have at leasts two "equal" sentences
                    sentences_with_props = [sentence for sentence in equal_kernels if len(sentence.properties) > 0]

                    found_properties = defaultdict(list)
                    for given_sentence in sentences_with_props:
                        found_properties = merge_properties(found_properties, dict(given_sentence.properties))
                    properties_to_keep["SENTENCE"] = [Singleton(
                        id=equal_kernels[0].id,
                        named_entity=equal_kernels[0].named_entity,
                        type=equal_kernels[0].type,
                        min=equal_kernels[0].min,
                        max=equal_kernels[0].max,
                        confidence=equal_kernels[0].confidence,
                        kernel=equal_kernels[0].kernel,
                        properties=create_props_for_singleton(found_properties),
                    )]

        return Singleton(
            id=kernel.id,
            named_entity=kernel.named_entity,
            type=kernel.type,
            min=kernel.min,
            max=kernel.max,
            confidence=kernel.confidence,
            kernel=kernel.kernel,
            properties=create_props_for_singleton(properties_to_keep),
        )

    def add_to_kernel_nodes(self, node, kernel_nodes):
        if isinstance(node, SetOfSingletons):
            kernel_nodes.add(node)
            for entity in node.entities:
                self.add_to_kernel_nodes(entity, kernel_nodes)
        else:
            if node.kernel is not None:
                kernel_nodes.add(node)
                # Check if properties has any Singleton's and add to kernel nodes also
                if node.kernel.edgeLabel is not None:
                    self.add_singletons_from_node_properties(node.kernel.edgeLabel, kernel_nodes)
                    self.add_to_kernel_nodes(node.kernel.edgeLabel, kernel_nodes)
                if node.kernel.source is not None:
                    self.add_singletons_from_node_properties(node.kernel.source, kernel_nodes)
                    self.add_to_kernel_nodes(node.kernel.source, kernel_nodes)
                if node.kernel.target is not None:
                    self.add_singletons_from_node_properties(node.kernel.target, kernel_nodes)
                    self.add_to_kernel_nodes(node.kernel.target, kernel_nodes)
            else:
                kernel_nodes.add(node)

        return kernel_nodes

    def add_singletons_from_node_properties(self, node, kernel_nodes):
        if isinstance(node, Singleton):
            for key in dict(node.properties):
                properties_key_ = dict(node.properties)[key]
                if not isinstance(properties_key_, str):
                    if isinstance(properties_key_, Singleton):
                        self.add_to_kernel_nodes(properties_key_, kernel_nodes)
                    else:
                        for prop_node in properties_key_:
                            if isinstance(prop_node, Singleton):
                                self.add_to_kernel_nodes(prop_node, kernel_nodes)
                            elif isinstance(prop_node, SetOfSingletons):
                                kernel_nodes.add(node)
                                for prop_entity in prop_node.entities:
                                    self.add_to_kernel_nodes(prop_entity, kernel_nodes)
