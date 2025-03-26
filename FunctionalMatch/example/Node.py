from dataclasses import dataclass
from typing import Optional

def example_ext(d, **kwargs):
    to_print = kwargs.get("printer", None)
    if to_print is not None:
        print(to_print)
    return d.update("genoveffo", "blue")

def example_call(d, **kwargs):
    return Node(50000, d, d)


@dataclass(frozen=True, order=True, unsafe_hash=True, eq=True)
class Node:
    val: object
    left: Optional['Node']
    right: Optional['Node']

    @staticmethod
    def leaf(val):
        return Node(val, None, None)