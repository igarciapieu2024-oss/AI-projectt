"""Every module imports without models or data (runs on GitHub for every pull request)."""
import importlib
import pkgutil

import pytest

import eval as eval_pkg
import src

MODULES = sorted(
    {m.name for m in pkgutil.walk_packages(src.__path__, "src.")}
    | {m.name for m in pkgutil.walk_packages(eval_pkg.__path__, "eval.")}
)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
