"""Both Oxford campaigns use one implementation/provenance/release boundary."""
import importlib.util
import sys
from pathlib import Path


def workflow():
    name = "oxford_workflow"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent.parent / "031_oxford_mergeable_confirmation/oxford_workflow.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def require_confirmation_release():
    return workflow().require_confirmation_release()


def require_development_ready():
    return workflow().require_development_ready()
