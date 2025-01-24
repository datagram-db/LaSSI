from LaSSI.structures.kernels.Sentence import get_prepositions

from LaSSI.external_services.Services import Services
from LaSSI.ner.string_functions import lemmatize_verb

parmenides = Services.getInstance().getParmenides()
logical_rules = parmenides.getLogicalRewritingRules()

def is_name_in_parmenides(name, parmenides_list, should_lemmatize=False):
    return len({lemmatize_verb(name) if should_lemmatize else name}.intersection(parmenides_list)) > 0

def is_nmod(kernel, node, initial_node, has_nmod, value):
    return has_nmod == value

def match_prepositions(kernel, node, initial_node, has_nmod, value):
    prepositions = get_prepositions(node)
    return value in prepositions

def is_materialised(kernel, node, initial_node, has_nmod, value):
    edge_label = kernel.kernel.edgeLabel.named_entity if kernel.kernel.edgeLabel is not None else "None"
    return is_name_in_parmenides(edge_label, parmenides.getMaterialisationVerbs(), True) == value

def is_causative(kernel, node, initial_node, has_nmod, value):
    edge_label = kernel.kernel.edgeLabel.named_entity if kernel.kernel.edgeLabel is not None else "None"
    return is_name_in_parmenides(edge_label, parmenides.getCausativeVerbs(), True) == value

def has_movement(kernel, node, initial_node, has_nmod, value):
    edge_label = kernel.kernel.edgeLabel.named_entity if kernel.kernel.edgeLabel is not None else "None"
    return is_name_in_parmenides(edge_label, parmenides.getMovementVerbs(), True) == value

def is_in_state(kernel, node, initial_node, has_nmod, value):
    return is_name_in_parmenides(node.named_entity, parmenides.getStateVerbs(), True) == value

def has_means(kernel, node, initial_node, has_nmod, value):
    return is_name_in_parmenides(node.named_entity, parmenides.getMeansVerbs(), True) == value

def is_abstract_entity(kernel, node, initial_node, has_nmod, value):
    return is_name_in_parmenides(node.named_entity, parmenides.getAbstractEntities(), True) == value

def has_measurement(kernel, node, initial_node, has_nmod, value):
    # TODO: Aware this is definitely wrong, what is the better way of resolving the measurement?
    if has_nmod:
        possible_measurement = f"{initial_node.kernel.source.named_entity} per {initial_node.kernel.target.named_entity}"
    else:
        possible_measurement = ""
    return is_name_in_parmenides(possible_measurement, parmenides.getUnitsOfMeasure(), False) == value

def has_nmod_part_of(kernel, node, initial_node, has_nmod, value):
    return (kernel.kernel.source.id == (initial_node.kernel.source.id if initial_node.kernel is not None else node.id)) == value

def is_symmetrical(kernel, node, initial_node, has_nmod, value):
    return (len(
        {x for x in parmenides.getNounsWithProperties() if lemmatize_verb(node.named_entity) == str(x.label) and (initial_node.kernel is not None and initial_node.kernel.source.named_entity in str(x.hasProperty))}
    ) > 0) == value

def is_a(kernel, node, initial_node, has_nmod, value):
    return (len({x for x in parmenides.getNounsWithA() if
         node.named_entity == str(x.label) and initial_node.kernel.source.named_entity in str(x.isA)}) > 0) == value

def has_number(kernel, node, initial_node, has_nmod, value):
    # TODO: In our examples, this is the case, will it always be?
    if has_nmod:
        return ("nummod" in dict(initial_node.kernel.source.properties)) == True
    else:
        return ("nummod" in dict(node.properties)) == value

def type_of_node(kernel, node, initial_node, has_nmod, value):
    # TODO: Is "DATE" always "SUTime", could we change the ontology to be matched "DATE"??
    return ("SUTime" if node.type == "DATE" else str(node.type) if node.type in {"GPE", "LOC"} else "None") == value

predicate_interpretation = {
    "isMaterializationVerb": is_materialised,
    "causative_verb": is_causative,
    "SingletonHasBeenMatchedBy": type_of_node,
    "abstract_entity": is_abstract_entity,
    "hasNumber": has_number,
    "hasUnitOfMeasure": has_measurement,
    "verb_of_state": is_in_state,
    "verb_of_motion": has_movement,
    "hasNMod": is_nmod,
    "hasNModPartOf": has_nmod_part_of,
    "hasNModIsA": is_a,
    "isSymmetricalIfComparedToNMod": is_symmetrical,
    "preposition": match_prepositions,
}

def get_matching_logical_rules(kernel, initial_node, has_nmod):
    node = initial_node
    if has_nmod:
        node = node.kernel.target

    for rule in logical_rules:
        rule = logical_rules[rule]
        if all(map(lambda premise: any(map(lambda value: predicate_interpretation[premise.name](kernel, node, initial_node, has_nmod, value), premise.values)), rule.premises)) and all(map(lambda m: any(map(lambda x: not predicate_interpretation[m.name](kernel, node, initial_node, has_nmod, x), m.values)), rule.not_premises)): return node, rule
    return node, None
