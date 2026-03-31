#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime LLM routing helpers for math_agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loongflow.framework.pes.context import Context, LLMConfig
from loongflow.framework.pes.context.workspace import Workspace


@dataclass
class LLMSwitchConfig:
    """Configuration for runtime LLM provider switching based on evaluation metrics."""
    enabled: bool = False
    qianfan: Optional[LLMConfig] = None
    chatanywhere: Optional[LLMConfig] = None
    speedup_threshold: float = 1.0
    initial_provider: str = "qianfan"
    faster_provider: str = "qianfan"
    slower_provider: str = "chatanywhere"
    # New parameters for score improvement based switching
    score_improvement_threshold: float = 1.0
    iteration_threshold: int = 2
    speedup_lower_threshold: float = 1.5  # Minimum speedup to use faster_provider

    @classmethod
    def from_any(cls, config: Any) -> "LLMSwitchConfig | None":
        """Create an LLMSwitchConfig from a dict-like object, or return None if empty."""
        if not config:
            return None
        if isinstance(config, cls):
            return config

        data = dict(config)
        qianfan = data.get("qianfan")
        chatanywhere = data.get("chatanywhere")
        return cls(
            enabled=bool(data.get("enabled", False)),
            qianfan=LLMConfig.model_validate(qianfan) if qianfan else None,
            chatanywhere=LLMConfig.model_validate(chatanywhere) if chatanywhere else None,
            speedup_threshold=float(data.get("speedup_threshold", 1.0)),
            initial_provider=str(data.get("initial_provider", "qianfan")),
            faster_provider=str(data.get("faster_provider", "qianfan")),
            slower_provider=str(data.get("slower_provider", "chatanywhere")),
            score_improvement_threshold=float(data.get("score_improvement_threshold", 1.0)),
            iteration_threshold=int(data.get("iteration_threshold", 2)),
            speedup_lower_threshold=float(data.get("speedup_lower_threshold", 1.5)),
        )


def _read_previous_best_evaluation(context: Context) -> Optional[dict[str, Any]]:
    if context.current_iteration <= 1:
        return None

    previous_context = Context(
        task=context.task,
        base_path=context.base_path,
        init_solution=context.init_solution,
        init_evaluation=context.init_evaluation,
        init_score=context.init_score,
        task_id=context.task_id,
        island_id=context.island_id,
        current_iteration=context.current_iteration - 1,
        total_iterations=context.total_iterations,
        trace_id=context.trace_id,
        metadata=context.metadata,
    )
    eval_path = Path(Workspace.get_executor_best_evaluation_path(previous_context))
    if not eval_path.exists():
        return None
    try:
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_speedup(evaluation: Optional[dict[str, Any]]) -> float:
    if not evaluation:
        return 0.0
    metrics = evaluation.get("metrics", {}) or {}
    return float(metrics.get("speedup", 0.0) or 0.0)


def _extract_score(evaluation: Optional[dict[str, Any]]) -> float:
    """Extract the score from evaluation data."""
    if not evaluation:
        return 0.0
    return float(evaluation.get("score", 0.0) or 0.0)


def _extract_parent_score(evaluation: Optional[dict[str, Any]]) -> float:
    """Extract the parent score from evaluation data."""
    if not evaluation:
        return 0.0
    metadata = evaluation.get("metadata", {}) or {}
    # Try to get parent_score from metadata if available
    return float(metadata.get("parent_score", 0.0) or 0.0)


def _read_parent_info(context: Context) -> Optional[dict[str, Any]]:
    """Read the parent info file to get parent score."""
    if context.current_iteration <= 1:
        return None

    previous_context = Context(
        task=context.task,
        base_path=context.base_path,
        init_solution=context.init_solution,
        init_evaluation=context.init_evaluation,
        init_score=context.init_score,
        task_id=context.task_id,
        island_id=context.island_id,
        current_iteration=context.current_iteration - 1,
        total_iterations=context.total_iterations,
        trace_id=context.trace_id,
        metadata=context.metadata,
    )
    parent_path = Path(Workspace.get_planner_parent_info_path(previous_context))
    if not parent_path.exists():
        return None
    try:
        data = json.loads(parent_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _provider_config(switch: LLMSwitchConfig, provider: str) -> LLMConfig:
    if provider == "qianfan" and switch.qianfan is not None:
        return switch.qianfan.model_copy()
    if provider == "chatanywhere" and switch.chatanywhere is not None:
        return switch.chatanywhere.model_copy()
    raise ValueError(f"LLM switch provider '{provider}' is not configured.")


def choose_llm_config(
    context: Context,
    default_config: LLMConfig,
    switch_config: LLMSwitchConfig | None,
) -> tuple[LLMConfig, str, float | None]:
    """Select an LLM config based on previous iteration performance.

    Switching logic:
    - Use faster_provider when ALL three conditions are met:
      1. iteration >= iteration_threshold
      2. (child_score - parent_score) > score_improvement_threshold
      3. speedup >= speedup_lower_threshold
    - Otherwise use slower_provider
    """
    if switch_config is None or not switch_config.enabled:
        return default_config, "default", None

    if context.current_iteration <= 1:
        provider = switch_config.initial_provider
        return _provider_config(switch_config, provider), provider, None

    # New switching logic based on score improvement, iteration count, and speedup
    previous_eval = _read_previous_best_evaluation(context)
    parent_info = _read_parent_info(context)

    child_score = _extract_score(previous_eval)
    parent_score = float(parent_info.get("score", 0.0)) if parent_info else 0.0
    score_improvement = child_score - parent_score
    previous_speedup = _extract_speedup(previous_eval)

    # Check all three conditions
    iteration_threshold_met = context.current_iteration >= switch_config.iteration_threshold
    score_improvement_met = score_improvement > switch_config.score_improvement_threshold
    speedup_threshold_met = previous_speedup >= switch_config.speedup_lower_threshold

    # Use faster_provider only if ALL three conditions are met
    if iteration_threshold_met and score_improvement_met and speedup_threshold_met:
        provider = switch_config.faster_provider
    else:
        provider = switch_config.slower_provider

    return _provider_config(switch_config, provider), provider, previous_speedup
