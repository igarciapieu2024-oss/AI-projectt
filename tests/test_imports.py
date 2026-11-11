"""Every module imports without models or data (runs on GitHub for every pull request)."""
import importlib
import pkgutil

import pytest

import eval as eval_pkg
import src

# streamlit pages run their whole page as soon as they are imported, so this test can't open them;
# they are tested by running them with "streamlit run" instead
STREAMLIT_PAGES = {"src.vision.label_tool"}

MODULES = sorted(
    ({m.name for m in pkgutil.walk_packages(src.__path__, "src.")}  # every file inside src/
     | {m.name for m in pkgutil.walk_packages(eval_pkg.__path__, "eval.")})  # every file inside eval/
    - STREAMLIT_PAGES  # minus the streamlit pages
)

@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
