from datetime import date, datetime
from typing import Any, Dict, List


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))


def serialize_value(value: Any) -> Any:
    """Convert complex payload values into JSON-serializable structures."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

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
