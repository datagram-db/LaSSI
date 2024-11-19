import re

from LaSSI.Parmenides.paremenides import Parmenides
from LaSSI.external_services.Services import Services

negations = {'not', 'no'}

def is_label_verb(edge_label_name):
    edge_label_name = lemmatize_verb(edge_label_name)
    parmenides_types = {str(x)[len(Parmenides.parmenides_ns):] for x in
                        Services.getInstance().getParmenides().typeOf(edge_label_name)}
    is_verb = any(map(lambda x: 'Verb' in x, parmenides_types)) or len(parmenides_types) == 0
    return is_verb

def does_string_have_negations(edge_label_name):
    return bool(re.search(r"\b(" + "|".join(re.escape(neg) for neg in negations) + r")\b", edge_label_name))

def match_whole_word(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def lemmatize_verb(edge_label_name):
    if len(edge_label_name) == 0:
        return ""
    stNLP = Services.getInstance().getStanzaSTNLP()
    lemmatizer = Services.getInstance().getWTLemmatizer()
    try:
        return " ".join(map(lambda y: y["lemma"], filter(lambda x: x["upos"] != "AUX", stNLP(lemmatizer.lemmatize(edge_label_name, 'v')).to_dict()[0])))
    except KeyError as e:
        return edge_label_name