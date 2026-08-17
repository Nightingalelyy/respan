from dataclasses import dataclass
from datetime import date

from respan_sdk.utils.serialization import serialize_value


class ArrayLike:
    def tolist(self):
        return [[0.1, 0.2], [0.3, 0.4]]


class ModelLike:
    def model_dump(self):
        return {"name": "record", "vectors": ArrayLike()}


@dataclass
class Container:
    created_on: date
    result: ModelLike


def test_serialize_value_normalizes_sdk_containers_and_array_values():
    assert serialize_value(
        Container(created_on=date(2026, 8, 17), result=ModelLike())
    ) == {
        "created_on": "2026-08-17",
        "result": {
            "name": "record",
            "vectors": [[0.1, 0.2], [0.3, 0.4]],
        },
    }
