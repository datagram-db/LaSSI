import dataclasses
from typing import List


@dataclasses.dataclass(frozen=True)
class MeuDBEntry:
    text: str
    type: str
    start_char: int
    end_char: int
    monad: str
    confidence: float
    id: str = None

    @classmethod
    def from_dict(cls, data):
        return cls(
            text = data.get('text'),
            type = data.get('type'),
            start_char = data.get('start_char'),
            end_char=data.get('end_char'),
            monad=data.get('monad'),
            confidence=data.get('confidence', 1.0),
            id= data.get("id")
        )

@dataclasses.dataclass
class MeuDB:
    first_sentence: str
    multi_entity_unit: List[MeuDBEntry]

    @classmethod
    def from_dict(cls, data):
        return cls(
            first_sentence = data.get('first_sentence'),
            multi_entity_unit = [MeuDBEntry.from_dict(x) for x in data.get('multi_entity_unit')]
        )
