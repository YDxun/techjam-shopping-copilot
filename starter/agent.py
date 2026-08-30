"""Official entrypoint: directly re-exports the new Agent (so the official evaluator never runs
    the old BM25 baseline).

The official evaluator `evaluator/local_evaluator.py` always imports `Agent` from this module:
    from starter.agent import Agent

this module only re-exports; not a single line of the official evaluator is changed.
The full implementation lives in `agent/main_agent.py`; the old BM25 baseline stays in git history.

Note: the dependency modules (agent/dialogue/, agent/capability_probe.py,
agent/runtime_controller.py etc.)
must ship with the submission, otherwise this entrypoint fails to import in a clean checkout.
"""
from __future__ import annotationsfrom agent.main_agent import Agent__all__ = ["Agent"]
