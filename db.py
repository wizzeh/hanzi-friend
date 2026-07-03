from tinydb import TinyDB, Query
import tinydb.operations as dbops
from tinydb.table import Document

db = TinyDB("db.json")

try:
    db.insert(Document({"type_": "characters_seen", "number_seen": 0}, doc_id=1))
except ValueError:
    pass  # We already inserted it, that's fine.


def characters_seen() -> int:
    return db.search(Query().type_ == "characters_seen")[0]["number_seen"]


def set_characters_seen(number: int):
    db.update(dbops.set("number_seen", number), Query().type_ == "characters_seen")


def increment_characters_seen():
    db.update(dbops.increment("number_seen"), Query().type_ == "characters_seen")


def user_definition(hanzi: str) -> str:
    Definitions = Query()
    found = db.search(
        (Definitions.type_ == "definition") & (Definitions.hanzi == hanzi)
    )
    return found[0]["definition"] if found else ""


def set_user_definition(hanzi: str, definition: str):
    Definitions = Query()
    db.upsert(
        {"type_": "definition", "definition": definition, "hanzi": hanzi},
        (Definitions.type_ == "definition") & (Definitions.hanzi == hanzi),
    )
