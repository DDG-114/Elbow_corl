"""Patch note for integrating TerrainContext into the existing types.py.

Codex should modify the existing `go1_lewm_mpc/common/types.py` rather than
blindly replacing it. The intended change is:

1. Add this import near the top:

    from go1_lewm_mpc.common.terrain_types import TerrainContext

2. Add this optional field to the existing ObsPacket dataclass:

    terrain_context: TerrainContext | None = None

If the project supports Python < 3.10, use Optional[TerrainContext] instead.
"""
