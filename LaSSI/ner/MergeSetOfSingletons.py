from collections import defaultdict
from itertools import repeat

from LaSSI.ner.node_functions import create_props_for_singleton
from LaSSI.structures.internal_graph.EntityRelationship import Singleton


def score_from_meu(min_value, max_value, node_type, meu_db_row, parmenides):
    # max_value = max_value
    matched_meus = []

    for meu in meu_db_row.multi_entity_unit:
        start_meu = meu.start_char
        end_meu = meu.end_char
        if min_value == start_meu and end_meu == max_value:
            # TODO: mgu or its opposite...
            if (parmenides.most_specific_type([node_type, meu.type]) == meu.type or
                    node_type == meu.type or
                    node_type == "None"):
                matched_meus.append(meu)

    if len(matched_meus) == 0:
        return 0, "None"
    else:
        max_score = max(map(lambda x: x.confidence, matched_meus))
        return max_score, parmenides.most_specific_type(
            list(map(lambda x: x.type, filter(lambda x: x.confidence == max_score, matched_meus))))


def GraphNER_withProperties(node, is_simplistic_rewriting, meu_db_row, parmenides, existentials):
    from LaSSI.structures.internal_graph.EntityRelationship import Singleton
    from LaSSI.utils.allChunks import allChunks
    import numpy

    chosen_entity = None
    norm_confidence = 1
    fusion_properties = dict()

    # Sort entities based on word position to keep correct order
    sorted_entities = sorted(node.entities, key=lambda x: float(dict(x.properties)['pos']))

    sorted_entity_names = list(map(getattr, sorted_entities, repeat('named_entity')))
    d = dict(zip(range(len(sorted_entity_names)), sorted_entity_names))  # dictionary for storing the replacing elements
    resolved_d = []

    layered_alternatives = defaultdict(list)
    for x in allChunks(list(d.keys())):
        layered_alternatives[len(x)].append(x)

    for layer in layered_alternatives.values():
        max_score = -1
        alternatives = []
        for x in layer:
            # if all(y in set(map(lambda z: z[0], itertools.chain(map(lambda t: (t, ) if isinstance(t, int) else t, d.keys())))) for y in x):
            exp = " ".join(map(lambda z: sorted_entity_names[z], x))
            min_value = min(map(lambda z: sorted_entities[z].min, x))
            max_value = max(map(lambda z: sorted_entities[z].max, x))

            all_types = [sorted_entities[z].type for z in x]
            specific_type = parmenides.most_specific_type(all_types)

            # TODO: Is this okay to do? This is done because min/max no match in MEU, but VERB is important to keep...
            if specific_type == "VERB":
                candidate_meu_score, candidate_meu_type = 1.0, "VERB"
            else:
                candidate_meu_score, candidate_meu_type = score_from_meu(min_value, max_value, specific_type,
                                                                         meu_db_row, parmenides)
            all_meu_score_prod = numpy.prod(list(map(lambda z: sorted_entities[z].confidence, x)))

            # if (score_from_meu(exp, min_value, max_value, specific_type, stanza_row) >=
            #         numpy.prod(list(map(lambda z: score_from_meu(sorted_entities[z].named_entity, sorted_entities[z].min, max_value, sorted_entities[z].type, stanza_row), x)))):
            if (
                    ((candidate_meu_score >= all_meu_score_prod)
                     or
                     ((specific_type != candidate_meu_type) and (
                             parmenides.most_specific_type([specific_type, candidate_meu_type]) == candidate_meu_type)))
                    or
                    (len(resolved_d) > 0 and all(candidate_meu_score >= subarray[1] for subarray in resolved_d) and (
                            specific_type != candidate_meu_type) and (all(
                        parmenides.most_specific_type([subarray[2], candidate_meu_type]) == candidate_meu_type for
                        subarray in resolved_d)))  # Check if current score is greater than previous resolutions
            ):
                if candidate_meu_score > max_score:
                    alternatives = [(x, all_meu_score_prod, exp, candidate_meu_type)]
                    max_score = candidate_meu_score

        if len(alternatives) > 0:
            alternatives.sort(key=lambda x: x[1])
            d = dict(zip(range(len(sorted_entity_names)), sorted_entity_names))
            candidate_delete = set()
            x = alternatives[-1]
            for k, v in d.items():
                if isinstance(k, int):
                    if k in x[0]:
                        candidate_delete.add(k)
                    elif isinstance(k, tuple):
                        if len(set(x[0]).intersection(set(k))) > 0:
                            candidate_delete.add(k)
            for z in candidate_delete:
                d.pop(z)
            d[x[0]] = x[2]
            resolved_d.append([d, x[1], x[3]])  # [d, confidence_score, type]

    print(resolved_d)
    # If resolved_d has > 1 elements, there are multiple resolutions with equal confidence score
    if len(resolved_d) > 1:
        # Therefore find the resolution with the most entities
        highest_num_of_entities = 0
        for current_d in resolved_d:
            # TODO: Check length of key instead of entity_name
            for entity_name in list(current_d[0].values()):
                current_num_of_entities = len(entity_name.split())
                if current_num_of_entities > highest_num_of_entities:
                    highest_num_of_entities = current_num_of_entities
                    d = current_d[0]
    elif len(resolved_d) == 1:
        d = resolved_d[0][0]

    print(d)
    print("OK")

    extra_name = ""
    extra_min = None
    extra_max = None
    extra_props = None

    # TODO: Remove time-space information and add as properties
    for entity in sorted_entities:
        norm_confidence *= entity.confidence

        fusion_properties = merge_properties(dict(entity.properties), fusion_properties)
        if (entity.named_entity == list(d.values())[0] and len(resolved_d) > 0) or entity.type.lower() == "verb":
            chosen_entity = entity
        else:
            extra_name = " ".join((extra_name, entity.named_entity))  # Ensure there is no leading space
            extra_min = entity.min if extra_min is None else extra_min if extra_min < entity.min else entity.min
            extra_max =  entity.max if extra_max is None else extra_max if extra_max > entity.max else entity.max

            # Only keep "core" properties, as other properties will be added to "chosen entity" instead, to make rewriting "easier" later on
            entity_props = {k: v for k, v in dict(entity.properties).items() if k in {'begin', 'end', 'number', 'pos', 'specification'}}
            extra_props = entity_props if extra_props is None else merge_properties(entity_props, extra_props)
    extra_name = extra_name.strip()  # Remove whitespace

    if norm_confidence > candidate_meu_score:
        candidate_meu_score = norm_confidence

    if is_simplistic_rewriting:
        new_properties = {
            "specification": "none",
            "begin": str(sorted_entities[0].min),
            "end": str(sorted_entities[len(sorted_entities) - 1].max),
            "pos": str(dict(sorted_entities[0].properties)['pos']),
            "number": "none"
        }

        new_properties = new_properties | fusion_properties

        if chosen_entity is not None:
            node_type = chosen_entity.type
        else:
            node_type = "ENTITY"

        merged_node = Singleton(
            id=node.id,
            named_entity=extra_name,
            properties=frozenset(new_properties.items()),
            min=sorted_entities[0].min,
            max=sorted_entities[len(sorted_entities) - 1].max,
            type=node_type,  # TODO: What should this type be?
            confidence=norm_confidence
        )
    elif chosen_entity is None:  # Not simplistic
        sing_type = 'None'
        min_value = sorted_entities[0].min
        max_value = sorted_entities[len(sorted_entities) - 1].max

        if extra_name != '':
            name = extra_name
            extra_name = ''
        else:
            sing_type = 'existential'
            name = "?" + str(existentials.increaseAndGetExistential())

        # New properties for ? object
        new_properties = {
            "specification": "none",
            "begin": str(min_value),
            "end": str(max_value),
            "pos": str(dict(sorted_entities[0].properties)['pos']),
            "number": "none"
        }
        if extra_name != '':
            new_properties['extra'] = [generate_extra_singleton(extra_name, extra_min, extra_max, extra_props)]
        new_properties = merge_properties(fusion_properties, new_properties)

        # Get score and type for newly created Singleton
        concat_candidate_meu_score, concat_candidate_meu_type = (
            score_from_meu(min_value, max_value, sing_type, meu_db_row, parmenides))

        candidate_meu_type = parmenides.most_specific_type([concat_candidate_meu_type, candidate_meu_type])

        merged_node = Singleton(
            id=node.id,
            named_entity=name,
            properties=create_props_for_singleton(new_properties),
            min=min_value,
            max=max_value,
            type=candidate_meu_type,
            confidence=candidate_meu_score
        )
    elif chosen_entity is not None:  # Not simplistic and found chosen entity
        # Convert back from frozenset to append new "extra" attribute
        new_properties = merge_properties(fusion_properties, dict(chosen_entity.properties))
        if extra_name != '':
            new_properties['extra'] = [generate_extra_singleton(extra_name, extra_min, extra_max, extra_props)]

        merged_node = Singleton(
            id=node.id,
            named_entity=chosen_entity.named_entity,
            properties=create_props_for_singleton(new_properties),
            min=sorted_entities[0].min,
            max=sorted_entities[len(sorted_entities) - 1].max,
            type=chosen_entity.type,
            confidence=norm_confidence
        )
    else:
        print("Error")
        merged_node = None

    return merged_node


def generate_extra_singleton(extra_name, extra_min, extra_max, extra_props):
    return Singleton(
        id=-1,
        named_entity=extra_name,
        properties=create_props_for_singleton(extra_props),
        min=extra_min,
        max=extra_max,
        type='None',  # TODO: What should this type be?
        confidence=1
    )


def merge_properties(orig_props, new_props, ignore_values=None):
    for key, new_value in new_props.items():
        if ignore_values is None or key not in ignore_values:
            if key in orig_props:
                if key == 'begin' or key == 'pos':
                    orig_props[key] = str(min(float(orig_props[key]), float(new_value)))
                elif key == 'end':
                    orig_props[key] = str(max(float(orig_props[key]), float(new_value)))
            else:
                orig_props[key] = new_value
        elif ignore_values is None:
            orig_props[key] = new_value
        elif key not in orig_props:
            orig_props[key] = new_value
    return orig_props
