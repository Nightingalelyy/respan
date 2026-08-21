from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any, Dict, List


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


def serialize_value(value: Any) -> Any:
    """Convert complex payload values into JSON-serializable structures."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return serialize_value(value=model_dump())

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return serialize_value(value=tolist())

    if is_dataclass(value) and not isinstance(value, type):
        return serialize_value(value=asdict(value))

    if isinstance(value, dict):
        normalized: Dict[str, Any] = {}
        for key, nested_value in value.items():
            normalized[str(key)] = serialize_value(value=nested_value)
        return normalized

    if isinstance(value, (list, tuple, set)):
        result: List[Any] = []
        for nested_value in value:
            result.append(serialize_value(value=nested_value))
        return result

    if hasattr(value, "__dict__"):
        return serialize_value(value=value.__dict__)

    return str(value)
