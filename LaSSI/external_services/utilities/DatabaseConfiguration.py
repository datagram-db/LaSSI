import dataclasses
import io

import dacite
import yaml


@dataclasses.dataclass()
class DatabaseConfiguration:
    db: str
    uname: str
    pw: str
    host: str
    port: int
    fuzzy_dbs: dict

def load_db_configuration(file:str|io.IOBase):
    f = file
    arg = None
    if isinstance(file, str):
        f = open(file, "r")
    if isinstance(f, io.IOBase):
        arg = yaml.safe_load(f)
    if isinstance(arg, dict):
        arg = dacite.from_dict(data_class=DatabaseConfiguration, data=arg)
    f.close()
    return arg