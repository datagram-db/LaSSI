import itertools
import re
import numpy
from collections import defaultdict
from LaSSI.external_services.Services import Services
from LaSSI.ner.string_functions import is_label_verb, lemmatize_verb
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, SetOfSingletons, Relationship
from LaSSI.structures.kernels.Sentence import is_kernel_in_props, create_edge_kernel, create_existential, \
    is_node_in_kernel_nodes


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
                for p in prototypical_prepositions:
                    # Check if prototypical preposition is in edge label but NOT just the edge label (i.e. "to" in "to steal" = TRUE, "like" in "like" = FALSE)
                    if re.search(r"\b"+p+r"\b", edge.edgeLabel.named_entity) and p != edge.edgeLabel.named_entity:
                        found_prototypical_prepositions = True
                        # Add root property to target
                        if isinstance(edge.target, Singleton):
                            target_props = dict(edge.target.properties)
                            target_props['kernel'] = 'root'
                            new_edge = Relationship(
                                source=edge.source,
                                target=Singleton.update_node_props(edge.target, target_props),
                                edgeLabel=edge.edgeLabel,
                                isNegated=edge.isNegated
                            )
                        elif isinstance(edge.target, SetOfSingletons):
                            new_edge = Relationship(
                                source=edge.source,
                                target=SetOfSingletons(
                                    id=edge.target.id,
                                    type=edge.target.type,
                                    entities=edge.target.entities,
                                    min=edge.target.min,
                                    max=edge.target.max,
                                    confidence=edge.target.confidence,
                                    root=True
                                ),
                                edgeLabel=edge.edgeLabel,
                                isNegated=edge.isNegated
                            )

                        # re.sub(r"\b"+p+r"\b", "", edge.edgeLabel.named_entity).strip()
                        found_proposition_labels[edge.target.id] = edge.edgeLabel.named_entity
                        break

                # If the edge is a verb and source is a 'root', remove 'root' from the target node of the edge
                if not found_prototypical_prepositions and is_label_verb(edge.edgeLabel.named_entity) and is_kernel_in_props(edge.source):
                    if isinstance(edge.target, Singleton):
                        new_edge = Relationship(
                            source=edge.source,
                            target=Singleton.strip_root_properties(edge.target),
                            edgeLabel=edge.edgeLabel,
                            isNegated=edge.isNegated
                        )
                    else:
                        new_target = SetOfSingletons(
                            id=edge.target.id,
                            type=edge.target.type,
                            entities=tuple(Singleton.strip_root_properties(entity) for entity in edge.target.entities),
                            # entities=edge.target.entities,
                            min=edge.target.min,
                            max=edge.target.max,
                            confidence=edge.target.confidence,
                            root=False
                        )
                        new_edge = Relationship(
                            source=edge.source,
                            target=new_target,
                            edgeLabel=edge.edgeLabel,
                            isNegated=edge.isNegated
                        )

                if new_edge is None:
                    new_edge = edge
                    if not is_kernel_in_props(edge.target):
                        true_targets.add(edge.target.id)
                new_edges.append(new_edge)

        filtered_nodes = set()
        filtered_top_node_ids = set()
        remove_set = set()
        # new_edges = list(set(new_edges)) # Remove duplicate edges

        # Loop over every source and target for every edge
        for edge_node in itertools.chain.from_iterable(map(lambda x: [x.source, x.target], new_edges)):
            if edge_node is None or edge_node.id in filtered_top_node_ids:
                continue
            filtered_top_node_ids.add(edge_node.id)

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

        # IDs of root nodes in topological order
        filtered_nodes = filtered_nodes - remove_set
        filtered_top_node_ids = [x.id for x in self.nodes.values() if x.id in filtered_nodes] # Filter node IDs in topological order based on self.nodes order

        if len(filtered_top_node_ids) == 0:
            filtered_top_node_ids = [list(self.nodes)[-1]] if len(self.nodes) == 1 else [self.node_functions.get_node_id(self.nodes[x].id) for x in self.nodes if is_kernel_in_props(self.nodes[x], False)]  # Get last node
            filtered_top_node_ids = list(map(int, numpy.unique(filtered_top_node_ids)))

        for node_id in filtered_top_node_ids:
            descendant_nodes = self.node_functions.node_bfs(new_edges, node_id)
            filtered_edges = [x for x in new_edges if (x.source.id in descendant_nodes and x.target.id in descendant_nodes) or (x.target.id in found_proposition_labels)]
            kernel = create_sentence_obj(filtered_edges, {key: x for key, x in self.nodes.items() if x.id in descendant_nodes}, self.negations, node_id, found_proposition_labels, self.node_functions)
            kernel = self.kernel_post_processing(kernel, position_pairs)
            self.nodes[node_id] = kernel

            # Remove 'root' property from nodes
            self.nodes = {k: Singleton.strip_root_properties(v) if isinstance(v, Singleton) and v.id in descendant_nodes else v for k, v in self.nodes.items()}

            # Re-instantiate new node properties across all relationships
            new_edges = [Relationship.from_nodes(r, self.nodes) for r in new_edges]

        # Return the last node ('highest' kernel)
        final_kernel = self.nodes[filtered_top_node_ids[-1]]

        # If "final kernel" does not have a kernel
        if not hasattr(final_kernel, 'kernel') or final_kernel.kernel is None:
            if final_kernel.type == 'verb':
                # If a Singleton of type verb, make this an edge with no source or target
                final_kernel = create_edge_kernel(final_kernel)
                final_kernel = self.kernel_post_processing(final_kernel, position_pairs)
            else:
                edges = []
                nodes = {final_kernel.id: final_kernel}
                if not hasattr(final_kernel, 'properties') or (not 'action' in dict(final_kernel.properties) and not 'actioned' in dict(final_kernel.properties)):
                    create_existential(edges, nodes)
                    final_kernel = create_sentence_obj(edges, nodes, self.negations, final_kernel.id, {}, self.node_functions)
                    final_kernel = self.kernel_post_processing(final_kernel, position_pairs)
                else:
                    # Get label from action prop
                    final_kernel_props = dict(final_kernel.properties)
                    actioned = 'actioned' in final_kernel_props
                    lemma_node_edge_label_name = lemmatize_verb(final_kernel_props['actioned']) if actioned else lemmatize_verb(final_kernel_props['action'])

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
                            source=final_kernel if not actioned else None,
                            target=final_kernel if actioned else None,
                            edgeLabel=Singleton(
                                id=-1,
                                named_entity=refactored_lemma_node_edge_label_name,
                                type='verb',
                                min=final_kernel.min,
                                max=final_kernel.max,
                                confidence=1,
                                kernel=None,
                                properties={}, # TODO: Keep properties
                            ),
                            isNegated=lemma_node_edge_label_name != refactored_lemma_node_edge_label_name,
                        ),
                        properties={},
                    )
                    final_kernel = self.kernel_post_processing(final_kernel, position_pairs)
        else:
            final_kernel = self.remove_duplicate_properties(final_kernel)

        final_kernel = self.check_if_empty_kernel(final_kernel)

        print(f"{final_kernel.to_string()}\n")
        return final_kernel

    # Check if final kernel is "empty" and use the properties of 'SENTENCE'
    def check_if_empty_kernel(self, kernel):
        if kernel.kernel is not None and kernel.kernel.edgeLabel is not None and kernel.kernel.edgeLabel.named_entity == "be" and kernel.kernel.source is not None and kernel.kernel.source.type == 'existential' and kernel.kernel.target is not None and kernel.kernel.target.type == 'existential':
            node_props = dict(kernel.properties)
            if node_props is not None and 'SENTENCE' in node_props:
                new_kernel = node_props['SENTENCE'][0] # TODO: Safe to use 0th element?
                return self.check_if_empty_kernel(new_kernel)
        else:
            return kernel

    # Rewrite kernel if positions are not correct
    def kernel_post_processing(self, kernel, position_pairs):
        if not isinstance(kernel, Singleton) or kernel.kernel is None:
            return kernel

        # Check kernel positions
        properties_to_keep = dict(kernel.properties)
        new_target = kernel.kernel.target
        if kernel.id in position_pairs and kernel.kernel.edgeLabel is not None and self.check_semi_modal(kernel.kernel.edgeLabel.named_entity):
            position_value = position_pairs[kernel.id]
            if 'SENTENCE' in dict(kernel.properties):
                properties_to_keep['SENTENCE'] = []
                for sentence_elm in list(dict(kernel.properties)['SENTENCE']):
                    if int(float(dict(sentence_elm.properties)['pos'])) == position_value:
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
                properties=frozenset({k: tuple(v) if isinstance(v, list) else v for k, v in properties_to_keep.items()}.items()),
            )
        return kernel


    def check_semi_modal(self, label):
        return len({label}.intersection(Services.getInstance().getParmenides().getSemiModalVerbs())) != 0


    def remove_duplicate_properties(self, kernel, kernel_nodes=None):
        if kernel.kernel is None:
            return kernel

        properties_to_keep = defaultdict(list)

        if kernel_nodes is None:
            kernel_nodes = []
        kernel_nodes.append(kernel.kernel.source) if kernel.kernel.source is not None else kernel_nodes
        kernel_nodes.append(kernel.kernel.target) if kernel.kernel.target is not None else kernel_nodes
        kernel_nodes.append(kernel.kernel.edgeLabel) if kernel.kernel.edgeLabel is not None else kernel_nodes

        for key in dict(kernel.properties):
            properties_key_ = dict(kernel.properties)[key]
            if isinstance(properties_key_, str):
                continue
            else:
                for node in properties_key_:
                    if key == 'SENTENCE':
                        if not is_node_in_kernel_nodes(node, kernel_nodes):
                            properties_to_keep[key].append(self.remove_duplicate_properties(node, kernel_nodes))
                        else:
                            self.remove_duplicate_properties(node, kernel_nodes)
                    elif not is_node_in_kernel_nodes(node, kernel_nodes):
                        properties_to_keep[key].append(node)

        return Singleton(
            id=kernel.id,
            named_entity=kernel.named_entity,
            type=kernel.type,
            min=kernel.min,
            max=kernel.max,
            confidence=kernel.confidence,
            kernel=kernel.kernel,
            properties=frozenset({k: tuple(v) if isinstance(v, list) else v for k, v in properties_to_keep.items()}.items()),
        )