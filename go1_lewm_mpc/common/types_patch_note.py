"""Patch note for TerrainContext integration.

The archived v3.0 plan asked Codex to modify the existing
``go1_lewm_mpc/common/types.py`` instead of replacing it. This repository now
has that integration:

1. ``ObsPacket`` imports ``TerrainContext`` from
   ``go1_lewm_mpc.common.terrain_types``.
2. ``ObsPacket`` has an optional ``terrain_context`` field.
3. ``ObsPacket.__post_init__`` validates that the field is either
   ``TerrainContext`` or ``None``.
"""
