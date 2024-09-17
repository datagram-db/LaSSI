from LaSSI.ner.AssignTypeToSingleton import AssignTypeToSingleton


class GraphProvenance:
    def __init__(self, raw_json_graph, meuDB, simplistic):
        self.nodes = dict()
        self.raw_json_graph = raw_json_graph
        self.simplistic = simplistic
        self.stanza_row = meuDB
        self.atts_global = AssignTypeToSingleton()
        self._internal_graph = None

    def _associate_type_to_item(self, item, key, nodes, stanza_row):
        # If key is SetOfSingletons, loop over each Singleton and make association to type
        # Giacomo: FIX, but only if they are not logical predicates
        from LaSSI.structures.internal_graph.EntityRelationship import SetOfSingletons
        if isinstance(item,
                      SetOfSingletons):  # and ((nodes[key].type == Grouping.NONE) or (nodes[key].type == Grouping.GROUPING)):
            for entity in item.entities:
                self._associate_type_to_item(entity, key, nodes, stanza_row)
        else:
            # assign_type_to_singleton(item, stanza_row, nodes, key)
            self.atts_global.assign_type_to_singleton_1(item, stanza_row)

    def internal_graph(self):
        from LaSSI.structures.internal_graph.from_raw_json_graph import group_nodes
        from LaSSI.external_services.Services import Services
        group_nodes(self.nodes, self.raw_json_graph, self.simplistic, Services.getInstance().getParmenides())
        for key in self.nodes:
            self._associate_type_to_item(self.nodes[key], key, self.nodes, self.stanza_row)
        self.atts_global.assign_type_to_all_singletons()
        for key in self.nodes:
            item = self.nodes[key]
            # association = nbb[item]
            # best_score = association.confidence
            from LaSSI.structures.internal_graph.EntityRelationship import SetOfSingletons
            if not isinstance(item, SetOfSingletons):
                self.nodes[key] = self.atts_global.associate_to_container(self.nodes[key], item)

        for key in self.nodes:
            item = self.nodes[key]
            # association = nbb[item]
            # best_score = association.confidence
            if isinstance(item, SetOfSingletons):
                self.nodes[key] = self.atts_global.associate_to_container(self.nodes[key], item)

        print("OK")