"""Payload benchmark scenario definitions."""

from __future__ import annotations


def payload_scenarios() -> list[dict]:
    return [
        {"name": "flat_0kg", "terrain": "flat", "payload_mass": 0.0, "notes": ""},
        {"name": "rough_0kg", "terrain": "rough", "payload_mass": 0.0, "notes": ""},
        {"name": "rough_1kg", "terrain": "rough", "payload_mass": 1.0, "notes": ""},
        {"name": "rough_2kg", "terrain": "rough", "payload_mass": 2.0, "notes": ""},
    ]
