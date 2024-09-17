__author__ = "Oliver Robert Fox"
__copyright__ = "Copyright 2024, Oliver Robert Fox"
__credits__ = ["Oliver Robert Fox"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver Robert Fox"
__email__ = "ollie.fox5@gmail.com"
__status__ = "Production"

# Function to get JSON key
def item_generator(json_input, lookup_key):
    if isinstance(json_input, dict):
        for k, v in json_input.items():
            if k == lookup_key:
                yield v
            else:
                yield from item_generator(v, lookup_key)
    elif isinstance(json_input, list):
        for item in json_input:
            yield from item_generator(item, lookup_key)