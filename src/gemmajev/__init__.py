"""A local, fixed-option decision API backed by Gemma 4."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "GemmaJev":
        from .engine import GemmaJev

        return GemmaJev
    raise AttributeError(name)


__all__ = ["GemmaJev"]
