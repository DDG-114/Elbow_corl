"""Terrain benchmark scenario definitions."""

from __future__ import annotations


def terrain_scenarios() -> list[dict]:
    return [
        {
            "name": "rough_push_1kg",
            "terrain": "rough",
            "payload_mass": 1.0,
            "random_push": True,
            "notes": "random push hook is logged; exact Isaac event config is deferred",
        },
        {
            "name": "stepping_stones_1kg",
            "terrain": "stepping_stones",
            "payload_mass": 1.0,
            "notes": "fallback to roughness level if terrain config is unavailable",
        },
        {
            "name": "stairs_1kg",
            "terrain": "stairs",
            "payload_mass": 1.0,
            "notes": "fallback to roughness level if terrain config is unavailable",
        },
    ]
