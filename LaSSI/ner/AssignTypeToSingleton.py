__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox, Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

from collections import defaultdict

from LaSSI.structures.meuDB.meuDB import MeuDB


class AssignTypeToSingleton:
    def __init__(self):
        self.associations = set()
        self.meu_entities = defaultdict(set)
        self.final_assigment = dict()

    def clear(self):
        self.associations.clear()
        self.meu_entities.clear()
        self.final_assigment.clear()

    def assign_type_to_singleton_1(self, item, stanza_row:MeuDB):
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
                # start_intersection = max(start_meu, start_graph)
                # end_intersection = min(end_meu, end_graph)
                if not (start_graph > end_meu or start_meu > end_graph):
                    self.meu_entities[item].add(meu)
                    self.associations.add(item)
                # if start_graph >= start_intersection and start_meu >= start_intersection and \
                #         end_graph <= end_intersection and end_meu <= end_intersection:
                #     if meu not in meu_entities:
                #         meu_entities.append(meu)
                #     if item not in entities:
                #         entities.append(item)
        # self.associations[item] = entities
        # print(associations[item])
        # print(meu_entities)

    def associate_to_container(self, nodes_key, item):
        from LaSSI.structures.internal_graph.EntityRelationship import SetOfSingletons
        if isinstance(nodes_key, SetOfSingletons):
            # if isinstance(item, SetOfSingletons):
            #     set_item = item
            # else:
            #     new_set = []
            confidence = 1.0
            new_set = list(map(lambda x: self.associate_to_container(x, x), nodes_key.entities))
            for entity in new_set:  # nodes_key.entities:
                # if (entity.confidence < best_score or math.isnan(best_score)) and \
                #         item.named_entity == entity.named_entity:  # Should this be == or in?
                #     new_set.append(item)
                #     confidence *= item.confidence
                # else:
                #     new_set.append(entity)
                confidence *= entity.confidence

            set_item = SetOfSingletons(
                id=item.id,
                type=nodes_key.type,
                entities=tuple(new_set),
                min=nodes_key.min,
                max=nodes_key.max,
                confidence=confidence
            )
            return set_item
        # nodes[key] = set_item
        # return set_item.confidence
        else:
            if item not in self.final_assigment:
                self.final_assigment[item] = item
                return item
            else:
                return self.final_assigment[item]
                # best_score = nbb[item].confidence
                # if nodes_key.confidence > best_score or math.isnan(best_score):
                #     # nodes[key] = item
                #     return item
                # else:
                #     # nodes[key] = association
                #     return nbb[item]

    def assign_type_to_all_singletons(self):
        if len(self.final_assigment) > 0:
            return self.final_assigment
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
                self.final_assigment[item] = Singleton(
                    id=item.id,
                    named_entity=item.named_entity,
                    properties=item.properties,
                    min=item.min,
                    max=item.max,
                    type=best_type,
                    confidence=best_score
                )
                # if isinstance(association, Singleton):
                #
                # else:
                #     self.final_assigment[item] = association
        return self.final_assigment