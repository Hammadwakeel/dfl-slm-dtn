"""Build model-callable tools from the shared action contract."""

import json
import math
import re
from pathlib import Path
from typing import Any


_JSON_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


ACTION_SCHEMA_DOCUMENT = json.loads(
    Path(__file__).with_name("slm_action_schema.json").read_text(encoding="utf-8")
)


def _resolve_references(value: Any, root: dict[str, Any]) -> Any:
    if isinstance(value, list):
        return [_resolve_references(item, root) for item in value]
    if not isinstance(value, dict):
        return value

    reference = value.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/"):
        resolved: Any = root
        for part in reference[2:].split("/"):
            resolved = resolved[part.replace("~1", "/").replace("~0", "~")]
        return _resolve_references(resolved, root)
    return {
        key: _resolve_references(item, root)
        for key, item in value.items()
        if key not in {"$defs", "$schema"}
    }


def agent_tool_definitions() -> list[dict[str, Any]]:
    definitions = ACTION_SCHEMA_DOCUMENT["$defs"]["action"]["oneOf"]
    tools = []
    for action_definition in definitions:
        action_property = action_definition["properties"]["action"]
        if not action_property.get("x-agent-tool", False):
            continue
        name = action_property["const"]
        parameter_schema = action_property.get(
            "x-agent-parameters",
            action_definition["properties"]["parameters"],
        )
        parameters = _resolve_references(
            parameter_schema,
            ACTION_SCHEMA_DOCUMENT,
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": action_property["description"],
                    "parameters": parameters,
                },
            }
        )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "finish_agent_workflow",
                "description": (
                    "End the current Agent workflow when the objective is achieved or "
                    "no further experiment is justified. The decision must be based on "
                    "observed tool results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "outcome": {
                            "type": "string",
                            "enum": ["objective_achieved", "no_worthwhile_action", "blocked"],
                        },
                        "summary": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1000,
                        },
                        "evidence": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1000,
                        },
                    },
                    "required": ["outcome", "summary", "evidence"],
                    "additionalProperties": False,
                },
            },
        }
    )
    return tools


def _matches_schema_variant(value: Any, schema: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return False
    if schema.get("additionalProperties") is False and set(value).difference(properties):
        return False
    if not set(schema.get("required", [])).issubset(value):
        return False
    for name, property_schema in properties.items():
        if name not in value:
            continue
        if "const" in property_schema and value[name] != property_schema["const"]:
            return False
        if "enum" in property_schema and value[name] not in property_schema["enum"]:
            return False
    return True


def _normalize_schema_value(value: Any, schema: dict[str, Any]) -> Any:
    schema = _resolve_references(schema, ACTION_SCHEMA_DOCUMENT)
    if isinstance(value, str):
        object_variants = [
            variant
            for keyword in ("oneOf", "anyOf")
            for variant in schema.get(keyword, [])
        ]
        accepts_object = schema.get("type") == "object" or any(
            variant.get("type") == "object" for variant in object_variants
        )
        if accepts_object:
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(decoded, dict):
                    value = decoded

    for branch in schema.get("allOf", []):
        value = _normalize_schema_value(value, branch)

    for keyword in ("oneOf", "anyOf"):
        variants = schema.get(keyword)
        if isinstance(variants, list):
            matching = [
                variant
                for variant in variants
                if _matches_schema_variant(value, variant)
            ]
            if len(matching) == 1:
                return _normalize_schema_value(value, matching[0])

    value_type = schema.get("type")
    if value_type == "integer" and isinstance(value, str):
        if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value):
            return int(value)
    elif value_type == "number" and isinstance(value, str):
        if _JSON_NUMBER.fullmatch(value):
            number = float(value)
            if math.isfinite(number):
                return number
    elif value_type == "boolean" and isinstance(value, str):
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        return {
            name: _normalize_schema_value(item, properties[name])
            if name in properties
            else item
            for name, item in value.items()
        }
    if isinstance(value, list):
        prefix_items = schema.get("prefixItems", [])
        item_schema = schema.get("items")
        return [
            _normalize_schema_value(
                item,
                prefix_items[index]
                if index < len(prefix_items)
                else item_schema
                if isinstance(item_schema, dict)
                else {},
            )
            for index, item in enumerate(value)
        ]
    return value


def agent_tool_action(name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    definitions = ACTION_SCHEMA_DOCUMENT["$defs"]["action"]["oneOf"]
    action_definition = next(
        (
            definition
            for definition in definitions
            if definition["properties"]["action"]["const"] == name
        ),
        None,
    )
    if action_definition is None:
        normalized_parameters = parameters
    else:
        normalized_parameters = _normalize_schema_value(
            parameters,
            action_definition["properties"]["parameters"],
        )
        if name == "start_training" and normalized_parameters.get("architecture") == {}:
            normalized_parameters.pop("architecture")
    return {"action": name, "parameters": normalized_parameters}
