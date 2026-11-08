"""Every module imports without models or data (runs on GitHub for every pull request)."""
import importlib
import pkgutil

import pytest

import eval as eval_pkg
import src

# Streamlit pages run their UI when imported, so they are not imported here.
STREAMLIT_PAGES = {"src.vision.label_tool"}

MODULES = sorted(
    ({m.name for m in pkgutil.walk_packages(src.__path__, "src.")}
     | {m.name for m in pkgutil.walk_packages(eval_pkg.__path__, "eval.")})
    - STREAMLIT_PAGES
)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
