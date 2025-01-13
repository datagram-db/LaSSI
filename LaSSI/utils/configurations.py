from dataclasses import dataclass

@dataclass(frozen=True, eq=True, order=True)
class LegacySemanticConfiguration:
    HuggingFace:str = 'sentence-transformers/all-MiniLM-L6-v2'
    string_similarity:str = 'Prevailing'
    string_similarity_prevailing:str = 'HuggingFace'
    verb_similarity:str = 'Prevailing'
    verb_similarity_prevailing:str = 'HuggingFace'
    ConceptNet5Numberbatch_lan:str = 'en'
    ConceptNet5Numberbatch_minTheta:str = 'en'
    prevailing_string_similarity:str = 'string_similarity_prevailing'
    prevailing_verb_similarity:str = 'verb_similarity_prevailing'