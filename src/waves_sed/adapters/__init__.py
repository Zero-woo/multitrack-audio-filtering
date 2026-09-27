"""Read-only metadata adapters; no inference dependencies are imported."""

from .waves import load_frozen, load_materialized

__all__ = ["load_frozen", "load_materialized"]
