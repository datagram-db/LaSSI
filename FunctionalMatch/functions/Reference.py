__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2025, Functional Match"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPLv3"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import ctypes
from typing import Optional

def reference(obj):
    """Returning the pointer associated to the object"""
    return id(obj)

def dereference(ptr):
    """Returning the object the pointer is referring to"""
    return ctypes.cast(ptr, ctypes.py_object).value


from dataclasses import dataclass

@dataclass(frozen=True, eq=True, order=True)
class Reference:
    ref: Optional[int]

    @staticmethod
    def from_object(obj):
        return Reference(ref=reference(obj))

    @staticmethod
    def null_ptr():
        return Reference(None)

    def get(self):
        return dereference(self.ref) if self.ref is not None else None