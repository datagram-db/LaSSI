__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

import itertools
import json
import re
from collections import defaultdict, deque
from copy import copy
from itertools import repeat
from typing import List

from LaSSI.Parmenides.paremenides import Parmenides
from LaSSI.external_services.Services import Services
from LaSSI.files.JSONDump import EnhancedJSONEncoder
from LaSSI.ner.MergeSetOfSingletons import merge_properties
from LaSSI.structures.internal_graph.EntityRelationship import Singleton, Grouping, SetOfSingletons, Relationship
from LaSSI.structures.internal_graph.Graph import Graph
from LaSSI.structures.kernels.Sentence import get_node_type, lemmatize_verb


# TODO: This is doing a bit more than "AssignTypeToSingleton", should move some functions into different classes?
class AssignTypeToSingleton:
    def __init__(self, is_simplistic_rewriting, meu_db_row, negations=None):
        if negations is None:
            negations = {'not', 'no'}
        self.negations = negations
        self.associations = set()
        self.meu_entities = defaultdict(set)
        self.node_id_map = dict()
        self.nodes = dict()
        self.edges = None
        self.services = Services.getInstance()
        self.existentials = self.services.getExistentials()
        self.is_simplistic_rewriting = is_simplistic_rewriting
        self.meu_db_row = meu_db_row

    def get_gsm_item_from_id(self, gsm_json, gsm_id):
        gsm_id_json = {row['id']: row for row in gsm_json}  # Associate ID to key value
        if gsm_id in gsm_id_json:
            return gsm_id_json[gsm_id]
        else:
            return None

    def get_node_id(self, node_id):
        value = self.node_id_map.get(node_id, node_id)
        if value is not None and value in self.node_id_map.keys():  # Check if value exists and is a key
            return self.node_id_map.get(value)
        else:
            return value

    def get_original_node_id(self, node_id):
        for old_id, new_id in self.node_id_map.items():
            # If we have MULTIINDIRECT, we just want the original ID as the passed on
            if old_id in self.nodes and self.nodes[old_id].type == Grouping.MULTIINDIRECT and len(
                    self.nodes[old_id].entities) == 1:
                return node_id

            if new_id == node_id:
                return old_id
        return node_id

    def resolve_node_id(self, node_id):
        if node_id is None:
            return Singleton(
                id=-1,
                named_entity="?" + str(self.existentials.increaseAndGetExistential()),
                properties=frozenset(),
                min=-1,
                max=-1,
                type='existential',
                confidence=1
            )
        else:
            return self.nodes[self.get_node_id(node_id)]

    def clear(self):
        self.associations.clear()
        self.meu_entities.clear()
        self.nodes.clear()

    def get_current_state_nodes(self):
        return self.nodes

    def does_string_have_negations(self, edge_label_name):
        return bool(re.search(r"\b(" + "|".join(re.escape(neg) for neg in self.negations) + r")\b", edge_label_name))

    def freshId(self):
        v = self.max_id
        self.max_id += 1
        return v

    def freshIdAndAddToNodeMap(self, id_to_map):
        fresh_id = self.freshId()
        self.node_id_map[id_to_map] = fresh_id
        return fresh_id

    def groupGraphNodes(self, gsm_json):
        # Used for when we want to generate a "fresh ID"
        self.max_id = max(map(lambda x: int(x["id"]), gsm_json)) + 1

        # Phase -1
        gsm_json = self.topologicalSort(gsm_json)

        # Phase 0
        self.preProcessing(gsm_json)

        # Phase 1
        self.extractLogicalConnectorsAsGroups(gsm_json)

        # Phase 1.5
        self.check_for_but(gsm_json)  # TODO: Move to kernel creation

        # TODO: These phases may be removed, as it already happens in Phase 1.1, as it might only need to happen for SetOfSingletons
        # Phase 2
        for key in self.nodes:
            self.associateNodeToMeuMatches(self.nodes[key])
        # Phase 3
        for item in self.associations:
            self.singletonTypeResolution(item)

        # Phase 4
        self.resolveGraphNERs()

    # Phase -1
    def getJsonAsAdjacencyList(self, gsm_json):
        gsm_id_json = {row['id']: row for row in gsm_json}

        num_of_nodes = max(item['id'] for item in gsm_json) + 1
        adj = [[] for _ in range(num_of_nodes)]

        for i in gsm_id_json:
            node = gsm_id_json[i]
            if len(node['phi']) > 0:
                for edge in node['phi']:
                    adj[edge['score']['parent']].append(edge['score']['child'])

        return adj, num_of_nodes

    def topologicalSortUtil(self, v, adj, visited, stack):
        # Mark the current node as visited
        visited[v] = True

        # Recur for all adjacent vertices
        for i in adj[v]:
            if not visited[i]:
                self.topologicalSortUtil(i, adj, visited, stack)

        # Push current vertex to stack which stores the result
        stack.append(v)

    def topologicalSort(self, gsm_json):
        adj, num_of_nodes = self.getJsonAsAdjacencyList(gsm_json)

        stack = []
        visited = [False] * num_of_nodes

        for i in range(num_of_nodes):
            if not visited[i]:
                self.topologicalSortUtil(i, adj, visited, stack)

        json_ids = {item['id'] for item in gsm_json}

        # Create a dictionary mapping IDs to their positions in the stack,
        # but only for IDs that exist in the JSON data
        id_order = {id: index for index, id in enumerate(stack) if id in json_ids}
        return sorted(gsm_json, key=lambda item: id_order.get(item['id']))

    # Phase 0
    def preProcessing(self, gsm_json):
        # Pre-processing not semantically driven
        # Scan for 'inherit' edges and contain them in the node that has that edge
        number_of_nodes = range(len(gsm_json))
        for row in number_of_nodes:
            gsm_item = gsm_json[row]
            edges_to_keep = []
            for edge in gsm_item['phi']:
                if 'inherit_' in edge['containment']:
                    node_to_inherit = self.get_gsm_item_from_id(gsm_json, edge['score']['child'])
                    if edge['containment'].endswith('_edge'):
                        gsm_item['xi'] = node_to_inherit['xi']
                        gsm_item['ell'] = node_to_inherit['ell']
                        new_properties = merge_properties(dict(gsm_item['properties']), dict(node_to_inherit['properties']))
                        gsm_item['properties'] = new_properties

                    for edge_to_inherit in node_to_inherit['phi']:
                        edge_to_inherit['score']['parent'] = gsm_item['id']
                        gsm_item['phi'].append(dict(edge_to_inherit))

                    # Remove edges from node that has been inherited
                    node_to_inherit['phi'] = []
                elif 'mark' in edge['containment']:
                    mark_target_node = self.get_gsm_item_from_id(gsm_json, edge['score']['child'])
                    if 'IN' in mark_target_node['ell'] or 'TO' in mark_target_node['ell']:
                        gsm_item['properties']['mark'] = mark_target_node['ell'][0]
                else:
                    edges_to_keep.append(edge)

            # Ignore 'inherit_edge' as they are accounted for, keep all other edges
            gsm_item['phi'] = edges_to_keep

    # Phase 1
    def extractLogicalConnectorsAsGroups(self, gsm_json):
        # Get all nodes from resulting graph and create list of Singletons
        number_of_nodes = range(len(gsm_json))

        # Phase 1.1
        self.create_all_singleton_nodes(gsm_json, number_of_nodes)

        # Check if conjugation ('conj') or ('multipleindobj') exists and if true exists, merge into SetOfSingletons
        # Also if 'compound' relationship is present, merge parent and child nodes
        for row in number_of_nodes:
            gsm_item = gsm_json[row]
            grouped_nodes = []
            has_conj = 'conj' in gsm_item['properties']
            has_multipleindobj = 'multipleindobj' in gsm_item['ell']
            is_compound = False
            has_compound_prt = False
            group_type = None
            norm_confidence = 1.0

            # TODO: Change this property to broader name?
            if len(gsm_item['xi']) > 1 and 'subjpass' in gsm_item['xi']:
                gsm_item['properties']['subjpass'] = gsm_item['xi'][1]

            # Get all nodes from edges of root node
            if has_conj or has_multipleindobj:
                for edge in gsm_item['phi']:
                    if 'orig' in edge['containment'] and (
                            (not has_multipleindobj) or self.is_label_verb(edge['containment'])):
                        node_id = self.get_node_id(edge['content'])
                        node = self.nodes[node_id]

                        # Merge properties from parent into children
                        # TODO: Should we really only be merging desired properties like 'kernel' (e.g. we now get conj property in children)?
                        if isinstance(node, Singleton):
                            node_props = merge_properties(gsm_item['properties'], dict(node.properties))
                            self.nodes[node_id] = Singleton.update_node_props(node, node_props)

                        grouped_nodes.append(self.nodes[node_id])
                        norm_confidence *= node.confidence
            else:
                # If not 'conjugation' or 'multipleindobj', then check for compound edges
                is_compound, norm_confidence = self.get_relationship_entities(grouped_nodes, gsm_json, gsm_item,
                                                                              is_compound,
                                                                              norm_confidence, 'compound', False)
                if not is_compound:
                    has_conj, norm_confidence = self.get_relationship_entities(grouped_nodes, gsm_json, gsm_item,
                                                                               has_conj,
                                                                               norm_confidence, 'conj', False)
                    if has_conj:
                        grouped_nodes.insert(0, self.nodes[self.get_node_id(gsm_item['id'])])
                    else:
                        has_compound_prt, norm_confidence = self.get_relationship_entities(grouped_nodes, gsm_json,
                                                                                           gsm_item,
                                                                                           has_compound_prt,
                                                                                           norm_confidence,
                                                                                           'compound_prt')

            # Determine conjugation type
            if has_conj:
                conj = gsm_item['properties']['conj'].strip() if 'conj' in gsm_item['properties'] else ""
                if len(conj) == 0:
                    from LaSSI.structures.internal_graph.from_raw_json_graph import bfs
                    def f(nodes, id):
                        cc_names = list(map(lambda y: nodes[y['score']['child']]['xi'][0],
                                            filter(lambda x: x['containment'] == 'cc', nodes[id]['phi'])))
                        if len(cc_names) > 0:
                            return ' '.join(cc_names)
                        else:
                            return None

                    conj = bfs(gsm_json, gsm_item['id'], f)

                if 'and' in conj or 'but' in conj:
                    group_type = Grouping.AND
                elif ('nor' in conj) or ('neither' in conj):
                    group_type = Grouping.NEITHER
                elif 'or' in conj:
                    group_type = Grouping.OR
                else:
                    group_type = Grouping.NONE

            if self.is_simplistic_rewriting and len(grouped_nodes) > 0:
                sorted_entities = sorted(grouped_nodes, key=lambda x: float(dict(x.properties)['pos']))
                sorted_entity_names = list(map(getattr, sorted_entities, repeat('named_entity')))

                all_types = list(map(getattr, sorted_entities, repeat('type')))
                specific_type = self.services.getParmenides().most_specific_type(all_types)

                if group_type == Grouping.OR:
                    name = " or ".join(sorted_entity_names)
                elif group_type == Grouping.AND:
                    name = " and ".join(sorted_entity_names)
                elif group_type == Grouping.NEITHER:
                    name = " nor ".join(sorted_entity_names)
                    name = f"neither {name}"
                else:
                    name = " ".join(sorted_entity_names)

                self.nodes[gsm_item['id']] = Singleton(
                    id=gsm_item['id'],
                    named_entity=name,
                    properties=frozenset(gsm_item['properties'].items()),
                    min=min(grouped_nodes, key=lambda x: x.min).min,
                    max=max(grouped_nodes, key=lambda x: x.max).max,
                    type=specific_type,
                    confidence=norm_confidence
                )
            elif not self.is_simplistic_rewriting:
                self.create_set_of_singletons(group_type, grouped_nodes, gsm_item, has_conj, has_multipleindobj, is_compound, has_compound_prt, norm_confidence, gsm_json)

        self.remove_duplicate_nodes()

    # Phase 1.1
    def create_all_singleton_nodes(self, gsm_json, number_of_nodes):
        for row in number_of_nodes:
            gsm_item = gsm_json[row]
            if 'conj' in gsm_item['properties'] or 'multipleindobj' in gsm_item['ell']:
                continue  # Add SetOfSingletons later as we need ALL Singletons first
            else:
                min_value = -1
                max_value = -1
                if len(gsm_item['xi']) > 0 and gsm_item['xi'][0] != '' and gsm_item['ell'][0] != '∃': # TODO: Checking for ∃ might not be valid...
                    name = gsm_item['xi'][0]
                    min_value = int(gsm_item['properties']['begin'])
                    max_value = int(gsm_item['properties']['end'])
                    node_type = gsm_item['ell'][0] if len(gsm_item['ell']) > 0 else "None"
                else:
                    # xi might be empty if the node is invented, therefore existential
                    name = "?" + str(self.existentials.increaseAndGetExistential())
                    node_type = 'existential'

                # If we have "root" in "ell", add it to properties
                if len(gsm_item['ell']) > 1:
                    gsm_item['properties']['kernel'] = gsm_item['ell'][1]

                if len(gsm_item['xi']) > 1 and 'subjpass' in gsm_item['xi']:
                    gsm_item['properties']['subjpass'] = gsm_item['xi'][1]

                # Add 'det' to properties
                if len(gsm_item['ell']) > 0 and 'det' in gsm_item['ell']:
                    gsm_item['properties']['det'] = gsm_item['ell'][0]

                self.nodes[gsm_item['id']] = Singleton(
                    id=gsm_item['id'],
                    named_entity=name,
                    properties=frozenset(gsm_item['properties'].items()),
                    min=min_value,
                    max=max_value,
                    type=node_type,
                    confidence=1.0
                )

                # TODO: Need to resolve grouped entities before merging, is there a more elegant way to do this? Could this thus be removed from the phases?
                self.associteNodeToBestMeuMatches(self.nodes[gsm_item['id']])
                self.singletonTypeResolution(self.nodes[gsm_item['id']])

    # Phase 1.2
    def get_relationship_entities(self, grouped_nodes, gsm_json, gsm_item, has_relationship, norm_confidence,
                                  edge_relationship_label='compound', deplete_phi=True):
        gsm_id_json = {row['id']: row for row in gsm_json}  # Associate ID to key value
        edges_to_remove = []
        for idx, edge in enumerate(gsm_item['phi']):
            is_current_edge_relationship = edge['containment'] in edge_relationship_label
            if is_current_edge_relationship:
                has_relationship = True
                child_node = self.nodes[edge['score']['child']]
                grouped_nodes.append(child_node)
                norm_confidence *= child_node.confidence

                child_gsm_item = gsm_id_json[edge['score']['child']]
                if len(child_gsm_item['phi']) > 0:
                    # Remove current compound edge
                    edges_to_remove.append(idx)
                    self.get_relationship_entities(grouped_nodes, gsm_json, child_gsm_item, has_relationship,
                                                   norm_confidence, edge_relationship_label)
                else:
                    edges_to_remove.append(idx)

        # Remove compound edges as no longer needed
        if deplete_phi:
            gsm_item['phi'] = [i for j, i in enumerate(gsm_item['phi']) if j not in edges_to_remove]
        return has_relationship, norm_confidence

    # Phase 1.3
    def create_set_of_singletons(self, group_type, grouped_nodes, gsm_item, has_conj, has_multipleindobj, is_compound,
                                 has_compound_prt, norm_confidence, gsm_json):
        if has_conj:
            new_id = self.freshIdAndAddToNodeMap(gsm_item['id'])
            self.nodes[new_id] = SetOfSingletons(
                id=new_id,
                type=group_type,
                entities=tuple(grouped_nodes),
                min=min(grouped_nodes, key=lambda x: x.min).min,
                max=max(grouped_nodes, key=lambda x: x.max).max,
                confidence=norm_confidence,
                root=any(map(self.is_kernel_in_props, grouped_nodes))
            )
        elif is_compound:
            grouped_nodes.insert(0, self.nodes[self.get_node_id(gsm_item['id'])])
            new_id = self.freshIdAndAddToNodeMap(gsm_item['id'])
            self.nodes[new_id] = SetOfSingletons(
                id=new_id,
                type=Grouping.GROUPING,
                entities=tuple(grouped_nodes),
                min=min(grouped_nodes, key=lambda x: x.min).min,
                max=max(grouped_nodes, key=lambda x: x.max).max,
                confidence=norm_confidence,
                root=any(map(self.is_kernel_in_props, grouped_nodes))
            )
        elif has_multipleindobj:
            self.nodes[gsm_item['id']] = SetOfSingletons(
                id=gsm_item['id'],
                type=Grouping.MULTIINDIRECT,
                entities=tuple(grouped_nodes),
                min=min(grouped_nodes, key=lambda x: x.min).min,
                max=max(grouped_nodes, key=lambda x: x.max).max,
                confidence=norm_confidence,
                root=any(map(self.is_kernel_in_props, grouped_nodes))
            )
        elif has_compound_prt:
            grouped_nodes.insert(0, self.nodes[self.get_node_id(gsm_item['id'])])
            new_id = self.freshIdAndAddToNodeMap(gsm_item['id'])

            sorted_entities = sorted(grouped_nodes, key=lambda x: float(dict(x.properties)['pos']))
            sorted_entity_names = list(map(getattr, sorted_entities, repeat('named_entity')))

            all_types = list(map(getattr, sorted_entities, repeat('type')))
            specific_type = self.services.getParmenides().most_specific_type(all_types)
            name = " ".join(sorted_entity_names)

            self.nodes[new_id] = Singleton(
                id=new_id,
                named_entity=name,
                properties=frozenset(gsm_item['properties'].items()),
                min=min(grouped_nodes, key=lambda x: x.min).min,
                max=max(grouped_nodes, key=lambda x: x.max).max,
                type=specific_type,
                confidence=1
            )

        node_to_resolve = self.nodes[self.get_node_id(gsm_item['id'])]
        if isinstance(node_to_resolve, SetOfSingletons):
            self.resolveSpecificSetOfSingletonsNERs(node_to_resolve)
            self.checkForSpecificNegation(gsm_json, self.nodes[self.get_node_id(gsm_item['id'])])

    # Phase 1.4
    def remove_duplicate_nodes(self):
        ids_to_remove = set()
        for idx, node in self.nodes.items():
            for jdx, another_node in self.nodes.items():
                if (idx != jdx and isinstance(node, SetOfSingletons) and
                        isinstance(another_node, SetOfSingletons) and
                        node.type == another_node.type):
                    if set(node.entities).issubset(set(another_node.entities)):
                        ids_to_remove.add(idx)
        for x in ids_to_remove:
            self.nodes.pop(x)

    # Phase 1.5
    def check_for_but(self, gsm_json):
        number_of_nodes = range(len(gsm_json))
        for row in number_of_nodes:
            gsm_item = gsm_json[row]
            but_gsm_item = None
            is_but = self.is_node_but(gsm_item)  # Is the current node BUT

            # Check for if the current node has an EDGE that contains BUT
            if len(gsm_item['phi']) > 0:
                for edge in gsm_item['phi']:
                    but_gsm_item = self.get_gsm_item_from_id(gsm_json, edge['score']['child'])
                    has_but = self.is_node_but(but_gsm_item)

                    # Check if BUT has edges, if it does, then we will use those later
                    if len(but_gsm_item['phi']) > 0:
                        has_but = False

                    if has_but:
                        break
                    else:
                        but_gsm_item = None

            if is_but:  # If current node "BUT", then check if it has children
                norm_confidence = 1.0
                but_node = self.nodes[self.get_node_id(gsm_item['id'])]
                grouped_nodes = []

                # Create group of all nodes attached to "but"
                for edge in gsm_item['phi']:
                    if 'neg' not in edge['containment']:
                        node = self.nodes[self.get_node_id(edge['content'])]
                        grouped_nodes.append(node)
                        norm_confidence *= node.confidence

                self.create_but_group(but_node, grouped_nodes, gsm_item, gsm_json, norm_confidence)
            elif but_gsm_item is not None:  # This node has a BUT child, so therefore create group
                norm_confidence = 1.0
                parent_node = self.nodes[self.get_node_id(gsm_item['id'])]
                grouped_nodes = [parent_node]
                self.create_but_group(parent_node, grouped_nodes, gsm_item, gsm_json, norm_confidence)

    def create_but_group(self, node, grouped_nodes, gsm_item, gsm_json, norm_confidence):
        # If node is a NOT and it equals the content of grouped nodes, then it is already a BUT and no need to do this again
        if isinstance(node, SetOfSingletons) and node.type == Grouping.NOT and node == grouped_nodes[0]:
            return
        if len(grouped_nodes) > 0:
            new_id = self.freshIdAndAddToNodeMap(gsm_item['id'])

            # Create an AND grouping from given child nodes
            new_but_node = SetOfSingletons(
                id=new_id,
                type=Grouping.AND,
                entities=tuple(grouped_nodes),
                min=min(grouped_nodes, key=lambda x: x.min).min,
                max=max(grouped_nodes, key=lambda x: x.max).max,
                confidence=norm_confidence
            )
            self.nodes[new_id] = new_but_node

            # If "but" has a negation, group the original grouped nodes in a NOT and then wrap again in an AND
            if self.checkForSpecificNegation(gsm_json, node, create_node=False):
                # Create new NOT node within nodes
                if grouped_nodes[0].id != gsm_item['id']:
                    new_new_id = self.freshIdAndAddToNodeMap(grouped_nodes[0].id)
                else:
                    new_new_id = self.freshId()

                self.nodes[new_new_id] = SetOfSingletons(
                    id=new_new_id,
                    type=Grouping.NOT,
                    entities=tuple(grouped_nodes),
                    min=min(grouped_nodes, key=lambda x: x.min).min,
                    max=max(grouped_nodes, key=lambda x: x.max).max,
                    confidence=norm_confidence
                )

                self.nodes[new_id] = SetOfSingletons(
                    id=new_id,
                    type=Grouping.AND,
                    entities=[self.nodes[new_new_id]],
                    min=min(grouped_nodes, key=lambda x: x.min).min,
                    max=max(grouped_nodes, key=lambda x: x.max).max,
                    confidence=norm_confidence
                )
            # else:
            #     # Otherwise return the new node
            #     self.nodes[new_id] = new_but_node

    def is_node_but(self, gsm_item):
        return len(gsm_item['xi']) > 0 and 'but' in gsm_item['xi'][0].lower() and 'cc' in gsm_item['ell'][0].lower()

    # Phase 2
    def associateNodeToMeuMatches(self, node):
        # If key is SetOfSingletons, loop over each Singleton and make association to type
        # Giacomo: FIX, but only if they are not logical predicates
        from LaSSI.structures.internal_graph.EntityRelationship import SetOfSingletons
        if isinstance(node, SetOfSingletons) and ((node.type == Grouping.NONE) or (node.type == Grouping.GROUPING)):
            for entity in node.entities:
                self.associateNodeToMeuMatches(entity)

            # merged_node = self.create_merged_node(node)  # So we can check if the concatenated name is in meuDB
            # self.associateNodeToMeuMatches(node, meu_db_row)
        elif isinstance(node, Singleton):
            # assign_type_to_singleton(item, stanza_row, nodes, key)
            self.associteNodeToBestMeuMatches(node)

    # Phase 2.1
    def associteNodeToBestMeuMatches(self, item):
        # Loop over Stanza MEU and GSM result and evaluate overlapping words from chars
        for meu in self.meu_db_row.multi_entity_unit:
            start_meu = meu.start_char
            end_meu = meu.end_char
            start_graph = item.min
            end_graph = item.max
            # https://scicomp.stackexchange.com/questions/26258/the-easiest-way-to-find-intersection-of-two-intervals/26260#26260
            if start_graph > end_meu or start_meu > end_graph:
                continue
            else:
                if not (start_graph > end_meu or start_meu > end_graph):
                    self.meu_entities[item].add(meu)
                    self.associations.add(item)

    # Phase 3
    def singletonTypeResolution(self, item):
        if len(self.meu_entities[item]) > 0:
            if item.type == '∃' or item.type.startswith("JJ") or item.type.startswith("IN") or item.type.startswith(
                    "NEG"):
                best_score = item.confidence
                best_items = [item]
                best_type = item.type
            else:
                # TODO: Fix typing information (e.g. Golden Gate Bridge has GPE (0.8) and ENTITY (1.0)
                best_score = max(map(lambda y: y.confidence, self.meu_entities[item]))
                # best_items = [
                #     y for y in self.meu_entities[item]
                #     if (len(item.named_entity.split(' ')) == 1 and y.confidence == best_score) or
                #        (y.confidence == best_score and
                #         len(item.named_entity.split(' ')) > 1 and
                #         y.monad is not None and
                #        y.monad.lower() == item.named_entity.lower())
                # ]
                # TODO: min and max might not be correct when coming from an inherit edge
                best_items = [
                    y for y in self.meu_entities[item]
                    if y.confidence == best_score
                ]
                if len(best_items) == 0:
                    return
                if len(best_items) == 1:
                    best_item = best_items[0]
                    best_type = best_item.type
                else:
                    best_types = list(set(map(lambda best_item: best_item.type, best_items)))
                    if len(best_types) == 1:
                        best_type = best_types[0]
                    ## TODO! type disambiguation, in future works, needs to take into account also the verb associated to it!
                    elif "verb" in best_types:
                        best_type = "verb"
                    elif "PERSON" in best_types:
                        best_type = "PERSON"
                    elif "DATE" in best_types or "TIME" in best_types:
                        best_type = "DATE"
                    elif "GPE" in best_types:
                        best_type = "GPE"
                    elif "LOC" in best_types:
                        best_type = "LOC"
                    elif "ENTITY" in best_types:
                        best_type = "ENTITY"
                    else:
                        best_type = "None"
            from LaSSI.structures.internal_graph.EntityRelationship import Singleton
            self.nodes[item.id] = Singleton(
                id=item.id,
                named_entity=item.named_entity,
                properties=item.properties,
                min=item.min,
                max=item.max,
                type=best_type,
                confidence=best_score
            )

    ## Phase 4
    def resolveGraphNERs(self):
        for key in self.nodes:
            node = self.nodes[key]
            # association = nbb[item]
            # best_score = association.confidence
            if not isinstance(node, SetOfSingletons):
                self.ConfidenceAndEntityExpand(key)

        for key in self.nodes:
            node = self.nodes[key]
            # association = nbb[item]
            # best_score = association.confidence
            if isinstance(node, SetOfSingletons):
                self.ConfidenceAndEntityExpand(key)

        # With types assigned, merge SetOfSingletons into Singleton
        for key in self.nodes:
            node = self.nodes[key]
            if isinstance(node, SetOfSingletons):
                from LaSSI.ner.MergeSetOfSingletons import GraphNER_withProperties
                if node.type == Grouping.MULTIINDIRECT:
                    # # If entities is only 1 item, then we can simply replace the item.id
                    # if len(node.entities) == 1:
                    #     entity = node.entities[0]
                    #     if isinstance(entity, SetOfSingletons):
                    #         self.nodes[node.id] = GraphNER_withProperties(entity, self.is_simplistic_rewriting, meu_db_row,
                    #                                                       parmenides, existentials)
                    #     else:
                    #         self.node_id_map[node.id] = entity.id
                    #         self.nodes[node.id] = self.ConfidenceAndEntityExpand(entity.id)
                    # # If more than 1 item, then we replace the entity.id for each orig of the 'multiindirectobj'
                    # else:
                    #     # nodes[item.id] = associate_to_container(item, nbb)
                    for entity in node.entities:
                        if isinstance(entity, SetOfSingletons):
                            self.nodes[entity.id] = GraphNER_withProperties(
                                entity,
                                self.is_simplistic_rewriting,
                                self.meu_db_row,
                                self.services.getParmenides(),
                                self.existentials
                            )
                        else:
                            self.node_id_map[node.id] = entity.id
                            self.ConfidenceAndEntityExpand(entity.id)
                if node.type == Grouping.GROUPING:
                    self.nodes[key] = GraphNER_withProperties(
                        node,
                        self.is_simplistic_rewriting,
                        self.meu_db_row,
                        self.services.getParmenides(),
                        self.existentials
                    )

    def resolveSpecificSetOfSingletonsNERs(self, node_to_resolve):
        for node in node_to_resolve.entities:
            if not isinstance(node, SetOfSingletons):
                self.ConfidenceAndEntityExpand(node.id)

        if isinstance(node_to_resolve, SetOfSingletons):
            self.ConfidenceAndEntityExpand(node_to_resolve.id)

        from LaSSI.ner.MergeSetOfSingletons import GraphNER_withProperties
        if node_to_resolve.type == Grouping.MULTIINDIRECT:
            for entity in node_to_resolve.entities:
                if isinstance(entity, SetOfSingletons):
                    self.nodes[entity.id] = GraphNER_withProperties(
                        entity,
                        self.is_simplistic_rewriting,
                        self.meu_db_row,
                        self.services.getParmenides(),
                        self.existentials
                    )
                else:
                    self.node_id_map[node_to_resolve.id] = entity.id
                    self.ConfidenceAndEntityExpand(entity.id)
        if node_to_resolve.type == Grouping.GROUPING:
            self.nodes[node_to_resolve.id] = GraphNER_withProperties(
                node_to_resolve,
                self.is_simplistic_rewriting,
                self.meu_db_row,
                self.services.getParmenides(),
                self.existentials
            )

    # Phase 4.1
    def ConfidenceAndEntityExpand(self, key):
        from LaSSI.structures.internal_graph.EntityRelationship import SetOfSingletons
        item = self.nodes[key]
        if isinstance(item, SetOfSingletons):
            confidence = 1.0
            new_set = list(map(lambda x: self.ConfidenceAndEntityExpand(x.id), item.entities))
            for entity in new_set:
                confidence *= entity.confidence
            set_item = SetOfSingletons(
                id=item.id,
                type=item.type,
                entities=tuple(new_set),
                min=item.min,
                max=item.max,
                confidence=confidence
            )
            self.nodes[key] = set_item
            return set_item
        else:
            if item.id not in self.nodes:
                self.nodes[item.id] = item
                return item
            else:
                return self.nodes[item.id]

    # Phase 5
    def checkForNegation(self, gsm_json):
        # Check for negation in node 'xi' and 'properties'
        grouped_nodes = []
        for row in range(len(gsm_json)):
            gsm_item = gsm_json[row]  # Find node in JSON so we can get child nodes
            key = gsm_item['id']
            try:
                node = self.nodes[self.get_node_id(key)]  # Singleton node

                if node.type == Grouping.NOT or (node.type == Grouping.AND and node.entities[0].type == Grouping.NOT):
                    return

                # # TODO: Check if xi is in negations or NEG is in ell
                # if len(gsm_item['xi']) > 0 and gsm_item['xi'][0] in self.negations or 'NEG' in gsm_item['ell']:
                #     is_prop_negated = True

                # Check if name is not/no or if negation found in properties
                ## TODO: there was a case where node didn't have the field named_entity
                ##       Please double check if this will fix it...
                # if is_prop_negated:
                #     self.nodes[key] = SetOfSingletons(
                #         id=key,
                #         type=Grouping.NOT,
                #         entities=tuple([node]),
                #         min=node.min,
                #         max=node.max,
                #         confidence=node.confidence
                #     )
                # else:
                # if hasattr(node, "named_entity") and node.named_entity in self.negations:
                grouped_nodes.append(node)
                for edge in gsm_item['phi']:
                    if edge['score']['child'] in self.nodes:  # Node might have been removed so check key exists
                        child = self.nodes[self.get_node_id(edge['score']['child'])]
                        if (hasattr(child, "named_entity") and child.named_entity in self.negations) or child.type == 'NEG':
                            # grouped_nodes.append(child)
                            fresh_id = self.freshIdAndAddToNodeMap(key)
                            self.nodes[fresh_id] = SetOfSingletons(
                                id=fresh_id,
                                type=Grouping.NOT,
                                entities=tuple(grouped_nodes),
                                min=min(grouped_nodes, key=lambda x: x.min).min,
                                max=max(grouped_nodes, key=lambda x: x.max).max,
                                confidence=child.confidence
                            )
                grouped_nodes = []
            except KeyError:
                continue

    def checkForSpecificNegation(self, gsm_json, node, grouped_nodes=None, create_node=True):
        if node.type == Grouping.NOT and len(node.entities) == 1:
            node = node.entities[0]

        # if node.type == Grouping.NOT:
        #     return

        if isinstance(node, SetOfSingletons):
            # grouped_nodes = [node]
            new_entities = []
            for entity in node.entities:
                # self.checkForSpecificNegation(gsm_json, entity, grouped_nodes, create_node)
                new_entity = self.checkForSpecificNegation(gsm_json, entity, grouped_nodes, create_node)
                if isinstance(new_entity, SetOfSingletons):
                    new_entities.append(new_entity)
                else:
                    new_entities.append(entity)
            if len(new_entities) > 0:
                self.nodes[node.id] = SetOfSingletons(
                    id=node.id,
                    type=node.type,
                    entities=tuple(new_entities),
                    min=min(new_entities, key=lambda x: x.min).min,
                    max=max(new_entities, key=lambda x: x.max).max,
                    confidence=node.confidence
                )
                node = self.nodes[node.id]
                return node

        if grouped_nodes is None:
            grouped_nodes = [node]

        # Check for negation in node 'xi' and 'properties'
        node_id = self.get_original_node_id(node.id)
        gsm_item = self.get_gsm_item_from_id(gsm_json, node_id)
        if gsm_item is not None:
            for edge in gsm_item['phi']:
                if edge['score']['child'] in self.nodes:  # Node might have been removed so check key exists
                    child = self.nodes[self.get_node_id(edge['score']['child'])]
                    if (hasattr(child, "named_entity") and child.named_entity in self.negations) or child.type == 'NEG':
                        # grouped_nodes.append(child)
                        if create_node:
                            fresh_id = self.freshIdAndAddToNodeMap(node_id)
                            self.nodes[fresh_id] = SetOfSingletons(
                                id=fresh_id,
                                type=Grouping.NOT,
                                entities=tuple(grouped_nodes),
                                min=min(grouped_nodes, key=lambda x: x.min).min,
                                max=max(grouped_nodes, key=lambda x: x.max).max,
                                confidence=child.confidence
                            )
                            return self.nodes[fresh_id]
                        else:
                            return True

    def constructIntermediateGraph(self, gsm_json, rejected_edges, non_verbs) -> Graph:
        self.edges = []

        # Add child node if 'amod' as properties of source node
        for row, gsm_item, edge in self.iterateOverEdges(gsm_json, rejected_edges):
            edge_label_name = edge['containment']  # Name of edge label
            if 'amod' in edge_label_name or 'advmod' in edge_label_name:  # TODO: Is this 'amod' or just 'mod' as could have 'nmod' (e.g. (traffic)-[nmod]->(Newcastle)
                source_node = self.nodes[self.get_node_id(edge['score']['parent'])]
                target_node = self.nodes[self.get_node_id(edge['score']['child'])]
                temp_prop = dict(copy(source_node.properties))
                if isinstance(target_node, Singleton):
                    type_key = get_node_type(target_node)
                else:
                    from LaSSI.external_services.Services import Services
                    type_key = self.services.getParmenides().most_general_type(
                        map(lambda x: x.type, target_node.entities))

                if type_key not in temp_prop:
                    temp_prop[type_key] = (target_node,)
                elif not isinstance(temp_prop[type_key], list):
                    temp_prop[type_key] = (temp_prop[type_key], target_node)
                else:
                    temp_prop[type_key] += (target_node,)

                self.nodes[self.get_node_id(edge['score']['parent'])] = Singleton(
                    id=source_node.id,
                    named_entity=source_node.named_entity,
                    properties=frozenset(temp_prop.items()),
                    min=source_node.min,
                    max=source_node.max,
                    type=source_node.type,  # TODO: This type is the ENUM case (i.e. 3 instead of NOT) so use dacite
                    confidence=source_node.confidence
                )

        for row in range(len(gsm_json)):
            gsm_item = gsm_json[row]

            if len(gsm_item['phi']) == 0:
                source_node_id = gsm_item['id']
                target_node_id = None
                source_node = self.resolve_node_id(source_node_id)
                if isinstance(source_node, Singleton):
                    source_properties = dict(source_node.properties)
                    if 'action' in source_properties:
                        edge_label_name = source_properties['action']
                        is_verb = self.is_label_verb(edge_label_name)
                        if is_verb:
                            self.create_edges(edge_label_name, gsm_item, non_verbs, source_node_id, target_node_id)

            for edge in gsm_item['phi']:
                # Check if child node has 'multipleindobj' type, to reassign edges from each 'orig'
                child_gsm_item = self.get_gsm_item_from_id(gsm_json, edge['score']['child'])
                if 'multipleindobj' in child_gsm_item['ell']:
                    edge_label_name = edge['containment']  # Name of edge label

                    source_node_id = edge['score']['parent']

                    for child_edge in child_gsm_item['phi']:
                        target_node_id = child_edge['score']['child']
                        self.create_edges(edge_label_name, gsm_item, non_verbs, source_node_id, target_node_id)

                # Make sure current edge is not in list of rejected edges, e.g. 'compound'
                if edge['containment'] not in rejected_edges:
                    if 'orig' not in edge['containment']:
                        edge_label_name = edge['containment']  # Name of edge label

                        source_node_id = edge['score']['parent']
                        target_node_id = edge['score']['child']
                        self.create_edges(edge_label_name, gsm_item, non_verbs, source_node_id, target_node_id)

        # print(json.dumps(Graph(self.edges), cls=EnhancedJSONEncoder))
        return Graph(self.edges)

    def is_label_verb(self, edge_label_name):
        edge_label_name = lemmatize_verb(edge_label_name)
        parmenides_types = {str(x)[len(Parmenides.parmenides_ns):] for x in
                            self.services.getParmenides().typeOf(edge_label_name)}
        is_verb = any(map(lambda x: 'Verb' in x, parmenides_types)) or len(parmenides_types) == 0
        return is_verb

    def iterateOverEdges(self, gsm_json, rejected_edges):
        for row in range(len(gsm_json)):
            gsm_item = gsm_json[row]
            for edge in gsm_item['phi']:
                # Make sure current edge is not in list of rejected edges, e.g. 'compound'
                if edge['containment'] not in rejected_edges:
                    if 'orig' not in edge['containment']:
                        yield row, gsm_item, edge

    def create_edges(self, edge_label_name, gsm_item, non_verbs, source_node_id, target_node_id):
        source_node = self.resolve_node_id(source_node_id)
        target_node = self.resolve_node_id(target_node_id)

        has_negations = self.does_string_have_negations(edge_label_name)

        # Giacomo: the former code was not able to handle: "does not play"
        query_words = edge_label_name.split()
        result_words = [word for word in query_words if word.lower() not in self.negations]
        edge_label_name = ' '.join(result_words)

        # Check if name of edge is in "non verbs"
        edge_type = "verb"
        for non_verb in non_verbs:
            if edge_label_name == non_verb.strip():
                edge_type = "non_verb"
                break
        if self.nodes[self.get_node_id(source_node_id)].type == Grouping.MULTIINDIRECT:
            # TODO: I don't think this logic is correct (e.g. The mouse is eaten by the cat)
            for node in self.nodes[self.get_node_id(source_node_id)].entities:
                self.edges.append(Relationship(
                    source=source_node,
                    target=self.resolve_node_id(node.id),
                    edgeLabel=Singleton(id=gsm_item['id'], named_entity=edge_label_name,
                                        properties=frozenset(dict().items()),
                                        min=-1,
                                        max=-1, type=edge_type,
                                        confidence=self.nodes[gsm_item['id']].confidence),
                    isNegated=has_negations
                ))
        elif 'amod' not in edge_label_name and 'advmod' not in edge_label_name and 'neg' not in edge_label_name:
            self.edges.append(Relationship(
                source=source_node,
                target=target_node,
                edgeLabel=Singleton(
                    id=gsm_item['id'],
                    named_entity=edge_label_name,
                    properties=frozenset(dict().items()),
                    min=-1,
                    max=-1,
                    type=edge_type,
                    confidence=self.nodes[self.get_node_id(gsm_item['id'])].confidence
                ),
                isNegated=has_negations
            ))

    def match_whole_word(self, w):
        return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

    def constructSentence(self) -> List[Singleton]:
        from LaSSI.structures.kernels.Sentence import create_sentence_obj

        found_proposition_labels = {}
        true_targets = set()
        new_edges = []
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

                # If the edge is a verb, remove 'root' from the target node of the edge
                if not found_prototypical_prepositions and self.is_label_verb(edge.edgeLabel.named_entity) and self.is_kernel_in_props(edge.source):
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
                            # entities=tuple(Singleton.strip_root_properties(entity) for entity in edge.target.entities),
                            entities=edge.target.entities,
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
                    if not self.is_kernel_in_props(edge.target):
                        true_targets.add(edge.target.id)
                new_edges.append(new_edge)

        filtered_nodes = set()
        filtered_top_node_ids = set()
        remove_set = set()

        # Loop over every source and target for every edge
        for edge_node in itertools.chain.from_iterable(map(lambda x: [x.source, x.target], new_edges)):
            if edge_node is None or edge_node.id in filtered_top_node_ids:
                continue
            filtered_top_node_ids.add(edge_node.id)

            # Check if edge node is not None, in true targets, is a root or existential
            if ((edge_node is not None and edge_node.id not in true_targets) and
                    ((isinstance(edge_node, Singleton) and self.is_kernel_in_props(edge_node)) or
                     (isinstance(edge_node, SetOfSingletons) and any(map(self.is_kernel_in_props, edge_node.entities))))
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

        for node_id in filtered_top_node_ids:
            descendant_nodes = self.node_bfs(new_edges, node_id)
            filtered_edges = [x for x in new_edges if (x.source.id in descendant_nodes or x.target.id in descendant_nodes) or (x.target.id in found_proposition_labels)]
            kernel = create_sentence_obj(filtered_edges, {key: x for key, x in self.nodes.items() if x.id in descendant_nodes}, self.negations, node_id, found_proposition_labels)
            self.nodes[node_id] = kernel

            # Remove 'root' property from nodes
            self.nodes = {k: Singleton.strip_root_properties(v) if isinstance(v, Singleton) and v.id in descendant_nodes else v for k, v in self.nodes.items()}

            # Re-instantiate new node properties across all relationships
            new_edges = [Relationship.from_nodes(r, self.nodes) for r in new_edges]

        # return create_sentence_obj(self.edges, self.nodes, self.negations)

        # Return the last node, as this will be the 'highest' kernel
        final_kernel = self.nodes[filtered_top_node_ids[-1]]
        print(f"{final_kernel.to_string()}\n")
        return final_kernel

    def is_kernel_in_props(self, x):
        if isinstance(x, SetOfSingletons):
            return x.root
        return (('kernel' in dict(x.properties) or 'root' in dict(x.properties)) and 'JJ' not in x.type)
        # TODO: Do we need to check if the JJ is/not a verb?
        # ('JJ' not in x.type or ('JJ' in x.type and self.is_label_verb(x.named_entity))))

    def node_bfs(self, edges, s):
        nodes = defaultdict(set)

        for x in edges:
            nodes[x.source.id].add(x.target.id)

        visited = set()

        # Create a queue for BFS
        q = deque()

        # Mark the source node as visited and enqueue it
        visited.add(s)
        q.append(s)

        # Iterate over the queue
        while q:
            id = q.popleft()
            for dst in nodes[id]:
                if dst not in visited:
                    visited.add(dst)
                    q.append(dst)

        return visited