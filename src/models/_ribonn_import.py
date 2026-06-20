"""Compatibility import for the RiboNN submodule's legacy package layout."""

import importlib
import sys


def import_ribonn_model():
    """Return RiboNN without modifying Python's module search path."""
    utils = importlib.import_module("RiboNN.src.utils")
    helpers = importlib.import_module("RiboNN.src.utils.helpers")
    sys.modules.setdefault("src.utils", utils)
    sys.modules.setdefault("src.utils.helpers", helpers)

    module = importlib.import_module("RiboNN.src.model")
    return module.RiboNN
