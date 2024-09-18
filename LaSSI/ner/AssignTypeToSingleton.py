__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

from collections import defaultdict
from itertools import repeat
from typing import List

from LaSSI.structures.internal_graph.EntityRelationship import Singleton, Grouping, SetOfSingletons, Relationship
from LaSSI.structures.kernels.Sentence import Sentence
from LaSSI.structures.meuDB.meuDB import MeuDB
from LaSSI.structures.internal_graph.Graph import Graph

class AssignTypeToSingleton:
    def __init__(self, negations=None):
        if negations is None:
            negations = {'not', 'no', 'but'}
        self.negations = negations
        self.associations = set()
        self.meu_entities = defaultdict(set)
        self.nodes = dict()
        self.edges = None

    def clear(self):
        self.associations.clear()
        self.meu_entities.clear()
        self.nodes.clear()

    def associteNodeToBestMeuMatches(self, item, stanza_row:MeuDB):
        # Loop over Stanza MEU and GSM result and evaluate overlapping words from chars
        for meu in stanza_row.multi_entity_unit:
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
        else:
            if item.id not in self.nodes:
                self.nodes[item.id] = item
                return item
            else:
                return self.nodes[item.id]

    def singletonTypeResolution(self):
        # if len(self.nodes) > 0:
        #     return self.nodes
        for item in self.associations:
            # for association in associations:
            if len(self.meu_entities[item]) > 0:
                if item.type == '∃' or item.type.startswith("JJ") or item.type.startswith("IN") or item.type.startswith(
                        "NEG"):
                    best_score = item.confidence
                    best_items = [item]
                    best_type = item.type
                else:
                    best_score = max(map(lambda y: y.confidence, self.meu_entities[item]))
                    best_items = [y for y in self.meu_entities[item] if y.confidence == best_score]
                    best_type = None
                    if len(best_items) == 1:
                        best_item = best_items[0]
                        best_type = best_item.type
                    else:
                        best_types = list(set(map(lambda best_item: best_item.type, best_items)))
                        best_type = None
                        if len(best_types) == 1:
                            best_type = best_types[0]
                        ## TODO! type disambiguation, in future works, needs to take into account also the verb associated to it!
                        elif "PERSON" in best_types:
                            best_type = "PERSON"
                        elif "DATE" in best_types or "TIME" in best_types:
                            best_type = "DATE"
                        elif "GPE" in best_types:
                            best_type = "GPE"
                        elif "LOC" in best_types:
                            best_type = "LOC"
                        else:
                            best_type = "None"
                from LaSSI.structures.internal_graph.EntityRelationship import Singleton
                self.nodes[item] = Singleton(
                    id=item.id,
                    named_entity=item.named_entity,
                    properties=item.properties,
                    min=item.min,
                    max=item.max,
                    type=best_type,
                    confidence=best_score
                )

    def associateNodeToMeuMatches(self, item, stanza_row):
        # If key is SetOfSingletons, loop over each Singleton and make association to type
        # Giacomo: FIX, but only if they are not logical predicates
        from LaSSI.structures.internal_graph.EntityRelationship import SetOfSingletons
        if isinstance(item,
                      SetOfSingletons):  # and ((nodes[key].type == Grouping.NONE) or (nodes[key].type == Grouping.GROUPING)):
            for entity in item.entities:
                self.associateNodeToMeuMatches(entity, stanza_row)
        else:
            # assign_type_to_singleton(item, stanza_row, nodes, key)
            self.associteNodeToBestMeuMatches(item, stanza_row)

    def freshId(self):
        v = self.maxId
        self.maxId += 1
        return v

    def groupGraphNodes(self, parsed_json, simplsitic, parmenides, stanza_row):
        self.maxId = max(map(lambda x: int(x["id"]), parsed_json))+1

        # Phase 1
        self.extractLogicalConnectorsAsGroups(parmenides, parsed_json, simplsitic)

        #Phase 2
        for key in self.nodes:
            self.associateNodeToMeuMatches(self.nodes[key], stanza_row)

        #Phase 3
        self.singletonTypeResolution()

        #Phase 4
        self.resolveGraphNERs(simplsitic, parmenides, stanza_row)

    def extractLogicalConnectorsAsGroups(self, parmenides, parsed_json, simplsitic):
        # Get all nodes from resulting graph and create list of Singletons
        number_of_nodes = range(len(parsed_json))
        for row in number_of_nodes:
            item = parsed_json[row]
            if 'conj' in item['properties'] or 'multipleindobj' in item['ell']:
                continue  # add set of singletons later as we might not have all nodes yet
            else:
                minV = -1
                maxV = -1
                typeV = "None"
                if len(item['xi']) > 0:
                    name = item['xi'][0]
                    minV = int(item['properties']['begin'])
                    maxV = int(item['properties']['end'])
                    typeV = item['ell'][0] if len(item['ell']) > 0 else "None"
                else:
                    name = '?'  # xi might be empty if the node is invented

                self.nodes[item['id']] = Singleton(
                    id=item['id'],
                    named_entity=name,
                    properties=frozenset(item['properties'].items()),
                    min=minV,
                    max=maxV,
                    type=typeV,
                    confidence=1.0
                )
        # Check if conjugation ('conj') exists and if true exists, merge into SetOfSingletons
        # Also if 'compound' relationship is present, merge parent and child nodes
        for row in number_of_nodes:
            item = parsed_json[row]
            grouped_nodes = []
            has_conj = 'conj' in item['properties']
            has_multipleindobj = 'multipleindobj' in item['ell']
            is_compound = False
            group_type = None
            norm_confidence = 1.0
            if has_conj:
                conj = item['properties']['conj'].strip()
                if len(conj) == 0:
                    from LaSSI.structures.internal_graph.from_raw_json_graph import bfs
                    conj = bfs(parsed_json, item['id'])

                if 'and' in conj or 'but' in conj:
                    group_type = Grouping.AND
                elif ('nor' in conj) or ('neither' in conj):
                    group_type = Grouping.NEITHER
                elif 'or' in conj:
                    group_type = Grouping.OR
                else:
                    group_type = Grouping.NONE
                for edge in item['phi']:
                    if 'orig' in edge['containment']:
                        node = self.nodes[edge['content']]
                        grouped_nodes.append(node)
                        norm_confidence *= node.confidence
            elif has_multipleindobj:
                for edge in item['phi']:
                    if 'orig' in edge['containment']:
                        node = self.nodes[edge['content']]
                        grouped_nodes.append(node)
                        norm_confidence *= node.confidence
            else:
                for edge in item['phi']:
                    is_current_edge_compound = 'compound' in edge['containment']
                    if is_current_edge_compound:
                        is_compound = True
                        node = self.nodes[edge['score']['child']]
                        grouped_nodes.append(node)
                        norm_confidence *= node.confidence

            if simplsitic and len(grouped_nodes) > 0:
                sorted_entities = sorted(grouped_nodes, key=lambda x: float(dict(x.properties)['pos']))
                sorted_entity_names = list(map(getattr, sorted_entities, repeat('named_entity')))

                all_types = list(map(getattr, sorted_entities, repeat('type')))
                specific_type = parmenides.most_specific_type(all_types)

                if group_type == Grouping.OR:
                    name = " or ".join(sorted_entity_names)
                elif group_type == Grouping.AND:
                    name = " and ".join(sorted_entity_names)
                elif group_type == Grouping.NEITHER:
                    name = " nor ".join(sorted_entity_names)
                    name = f"neither {name}"
                else:
                    name = " ".join(sorted_entity_names)

                self.nodes[item['id']] = Singleton(
                    id=item['id'],
                    named_entity=name,
                    properties=frozenset(item['properties'].items()),
                    min=min(grouped_nodes, key=lambda x: x.min).min,
                    max=max(grouped_nodes, key=lambda x: x.max).max,
                    type=specific_type,
                    confidence=norm_confidence
                )
            elif not simplsitic:
                if has_conj:
                    if group_type == Grouping.NEITHER:
                        grouped_nodes = [
                            SetOfSingletons(id=x.id, type=Grouping.NOT, entities=tuple([x]), min=x.min, max=x.max,
                                            confidence=x.confidence) for x in grouped_nodes]
                        grouped_nodes = tuple(grouped_nodes)
                        group_type = Grouping.AND
                    self.nodes[item['id']] = SetOfSingletons(
                        id=item['id'],
                        type=group_type,
                        entities=tuple(grouped_nodes),
                        min=min(grouped_nodes, key=lambda x: x.min).min,
                        max=max(grouped_nodes, key=lambda x: x.max).max,
                        confidence=norm_confidence
                    )
                elif is_compound:
                    grouped_nodes.insert(0, self.nodes[item['id']])
                    neu_id = self.freshId()
                    self.nodes[neu_id] = SetOfSingletons(
                        id=neu_id,
                        type=Grouping.GROUPING,
                        entities=tuple(grouped_nodes),
                        min=min(grouped_nodes, key=lambda x: x.min).min,
                        max=max(grouped_nodes, key=lambda x: x.max).max,
                        confidence=norm_confidence
                    )
                elif has_multipleindobj:
                    self.nodes[item['id']] = SetOfSingletons(
                        id=item['id'],
                        type=Grouping.MULTIINDIRECT,
                        entities=tuple(grouped_nodes),
                        min=min(grouped_nodes, key=lambda x: x.min).min,
                        max=max(grouped_nodes, key=lambda x: x.max).max,
                        confidence=norm_confidence
                    )

    def resolveGraphNERs(self, simplistic, parmenides, stanza_row):
        ## Phase 4
        for key in self.nodes:
            item = self.nodes[key]
            # association = nbb[item]
            # best_score = association.confidence
            if not isinstance(item, SetOfSingletons):
                self.ConfidenceAndEntityExpand(key)

        for key in self.nodes:
            item = self.nodes[key]
            # association = nbb[item]
            # best_score = association.confidence
            if isinstance(item, SetOfSingletons):
                self.ConfidenceAndEntityExpand(key)

        # With types assigned, merge SetOfSingletons into Singleton
        for key in self.nodes:
            item = self.nodes[key]
            if isinstance(item, SetOfSingletons):
                from LaSSI.ner.MergeSetOfSingletons import GraphNER_withProperties
                if item.type == Grouping.MULTIINDIRECT:
                    # If entities is only 1 item, then we can simply replace the item.id
                    if len(item.entities) == 1:
                        entity = item.entities[0]
                        if isinstance(entity, SetOfSingletons):
                            self.nodes[item.id] = GraphNER_withProperties(entity, simplistic, stanza_row, parmenides)
                        else:
                            self.nodes[item.id] = self.ConfidenceAndEntityExpand(entity.id)
                    # If more than 1 item, then we replace the entity.id for each orig of the 'multiindirectobj'
                    else:
                        # nodes[item.id] = associate_to_container(item, nbb)
                        for entity in item.entities:
                            if isinstance(entity, SetOfSingletons):
                                self.nodes[entity.id] = GraphNER_withProperties(entity, simplistic, stanza_row, parmenides)
                            else:
                                self.ConfidenceAndEntityExpand(entity.id)
                if item.type == Grouping.GROUPING:
                        self.nodes[key] = GraphNER_withProperties(item, simplistic, stanza_row, parmenides)

    def checkForNegation(self, parsed_json):
        # Check for negation in node 'xi' and 'properties
        for key in self.nodes:
            grouped_nodes = []
            sing_item = self.nodes[key]  # Singleton node

            for row in range(len(parsed_json)):  # Find node in JSON so we can get child nodes
                json_item = parsed_json[row]
                is_prop_negated = False

                for prop_key in json_item['properties']:  # Check properties for negation
                    prop = json_item['properties'][prop_key]
                    if prop in self.negations:
                        is_prop_negated = True

                # Check if name is not/no or if negation found in properties
                if key == json_item['id']:
                    ## TODO: there was a case where sing_item didn't have the field named_entity
                    ##       Please double check if this will fix it...
                    if hasattr(sing_item, "named_entity") and sing_item.named_entity in self.negations:
                        for edge in json_item['phi']:
                            child = self.nodes[edge['score']['child']]
                            if child.named_entity not in self.negations:
                                grouped_nodes.append(child)
                                self.nodes[key] = SetOfSingletons(
                                    type=Grouping.NOT,
                                    entities=tuple(grouped_nodes),
                                    min=min(grouped_nodes, key=lambda x: x.min).min,
                                    max=max(grouped_nodes, key=lambda x: x.max).max,
                                    confidence=child.confidence
                                )
                    elif is_prop_negated:
                        self.nodes[key] = SetOfSingletons(
                            type=Grouping.NOT,
                            entities=tuple([sing_item]),
                            min=sing_item.min,
                            max=sing_item.max,
                            confidence=sing_item.confidence
                        )

    def getCurrentStateNodes(self):
        return self.nodes

    def constructIntermediateGraph(self, parsed_json, rejected_edges, non_verbs) -> Graph:
        self.edges = []
        for row in range(len(parsed_json)):
            item = parsed_json[row]
            for edge in item['phi']:
                # Make sure current edge is not in list of rejected edges, e.g. 'compound'
                if edge['containment'] not in rejected_edges:
                    if 'orig' not in edge['containment']:
                        x = edge['containment']  # Name of edge label

                        # Giacomo: the former code was not able to handle: "does not play"
                        querywords = x.split()
                        resultwords = [word for word in querywords if word.lower() not in self.negations]
                        x = ' '.join(resultwords)

                        has_negations = any(map(lambda x: x in edge['containment'], self.negations))

                        # Check if name of edge is in "non verbs"
                        edge_type = "verb"
                        for non_verb in non_verbs:
                            if x == non_verb.strip():
                                edge_type = "non_verb"
                                break

                        if self.nodes[edge['score']['child']].type == Grouping.MULTIINDIRECT:
                            for node in self.nodes[edge['score']['child']].entities:
                                self.edges.append(Relationship(
                                    source=self.nodes[edge['score']['parent']],
                                    target=self.nodes[node.id],
                                    edgeLabel=Singleton(id=item['id'], named_entity=x,
                                                        properties=frozenset(dict().items()),
                                                        min=-1,
                                                        max=-1, type=edge_type,
                                                        confidence=self.nodes[item['id']].confidence),
                                    isNegated=has_negations
                                ))
                        else:
                            self.edges.append(Relationship(
                                source=self.nodes[edge['score']['parent']],
                                target=self.nodes[edge['score']['child']],
                                edgeLabel=Singleton(id=item['id'], named_entity=x,
                                                    properties=frozenset(dict().items()),
                                                    min=-1,
                                                    max=-1, type=edge_type,
                                                    confidence=self.nodes[item['id']].confidence),
                                isNegated=has_negations
                            ))

        return Graph(self.edges)
        # return edges

    def constructSentence(self, transitive_verbs) -> List[Sentence]:
        from LaSSI.structures.kernels.Sentence import create_sentence_obj
        return create_sentence_obj(self.edges, self.nodes, transitive_verbs, self.negations)