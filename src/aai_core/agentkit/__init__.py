"""Agent-evaluation toolkit: comparison-first evaluation with governed evidence.

An experiment is a comparison, not a log. The ``agentkit`` CLI scores every
change against the previous version on the same dataset; the MLflow run,
the lineage tags, and the scorer/prompt versions are byproducts the toolkit
generates. See ``docs/agent-evaluation.md``.

This package imports with base dependencies only; MLflow and Databricks
clients load lazily inside the commands that need them.
"""

from aai_core.agentkit.catalog import CATALOG, ScorerSpec, get_spec
from aai_core.agentkit.config import AgentkitConfig, ProjectContext, load_config

__all__ = [
    "CATALOG",
    "AgentkitConfig",
    "ProjectContext",
    "ScorerSpec",
    "get_spec",
    "load_config",
]
