import dataclasses

@dataclasses.dataclass()
class DatabaseConfiguration:
    db: str
    uname: str
    pw: str
    host: str
    port: int
    fuzzy_dbs: dict