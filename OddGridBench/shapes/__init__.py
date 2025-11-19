# shapes/__init__.py
from .registry import draw_random_shape, draw_shape_by_name, shape_registry
from . import svg_shapes

__all__ = ["draw_random_shape", "draw_shape_by_name", "shape_registry"]
