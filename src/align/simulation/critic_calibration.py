"""Matched-rollout critic learning-rate calibration; requires vendor PyTorch."""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from align.learning.critic_calibration_config import CriticCalibrationConfig
from align.learning.ppo_config import RecurrentPPOConfig
from align.learning.torch_ppo import compute_critic_batch
from align.learning.torch_rollout import TorchRecurrentRollout, TorchSequenceChunks
from align.policies.config import RecurrentPolicyConfig
from align.policies.torch_recurrent import CentralizedRecurrentCritic

SAMPLE_COLUMNS = (
    "critic_learning_rate",
    "environment_id",
    "rollout_step",
    "team_reward",
    "old_value",
    "return",
    "advantage",
    "pre_prediction",
    "post_prediction",
    "value_delta",
    "absolute_value_delta",
    "value_clipped",
)


def _distribution(value: torch.Tensor) -> dict:
    sample = value.detach().float().flatten()
    if sample.numel() == 0 or not bool(torch.isfinite(sample).all()):
        raise ValueError("distribution sample must be nonempty and finite")
    quantiles = torch.quantile(
        sample,
        sample.new_tensor((0.05, 0.25, 0.5, 0.75, 0.95)),
    )
    return {
        "count": int(sample.numel()),
        "minimum": float(sample.min().cpu()),
        "p05": float(quantiles[0].cpu()),
        "p25": float(quantiles[1].cpu()),
        "median": float(quantiles[2].cpu()),
        "p75": float(quantiles[3].cpu()),
        "p95": float(quantiles[4].cpu()),
        "maximum": float(sample.max().cpu()),
        "mean": float(sample.mean().cpu()),
        "standard_deviation": float(sample.std(unbiased=False).cpu()),
    }


def _maximum_parameter_change(before: dict, module: torch.nn.Module) -> float:
    return max(
        float((value.detach() - before[name]).abs().max().cpu())
        for name, value in module.state_dict().items()
        if torch.is_floating_point(value)
    )


def _finite_gradients(module: nn.Module) -> bool:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    return bool(gradients) and all(
        gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients
    )


def calibrate_critic_steps(
    *,
    critic: CentralizedRecurrentCritic,
    chunks: TorchSequenceChunks,
    rollout: TorchRecurrentRollout,
    output: Path,
    policy_config: RecurrentPolicyConfig,
    ppo_config: RecurrentPPOConfig,
    calibration_config: CriticCalibrationConfig,
) -> dict:
    """Apply candidate critic steps to identical weights and rollout samples."""
    if ppo_config.update_epochs != 1:
        raise ValueError("critic step calibration requires one configured PPO epoch")
    candidates = calibration_config.candidate_critic_learning_rates
    if not math.isclose(candidates[0], ppo_config.critic_learning_rate):
        raise ValueError("first candidate must match the configured primary critic rate")

    rows = chunks.critic
    valid = rows.valid_mask.bool()
    environment_ids = rows.environment_ids.unsqueeze(1).expand_as(rows.valid_mask)[valid]
    offsets = torch.arange(rows.valid_mask.shape[1], device=rows.valid_mask.device)
    rollout_steps = (rows.start_steps.unsqueeze(1) + offsets)[valid]
    expected_samples = rollout.config.horizon * rollout.config.num_envs
    if environment_ids.numel() != expected_samples:
        raise ValueError("critic chunks do not contain each environment transition exactly once")

    initial_state = {name: value.detach().clone() for name, value in critic.state_dict().items()}
    with torch.no_grad():
        source = compute_critic_batch(critic, chunks, ppo_config)
    input_distributions = {
        "team_rewards": _distribution(source.team_rewards),
        "old_values": _distribution(source.old_values),
        "bootstrap_values": _distribution(rollout.bootstrap_values),
        "returns": _distribution(source.returns),
        "advantages": _distribution(source.advantages),
        "return_minus_old_value": _distribution(source.returns - source.old_values),
    }

    candidate_results = []
    raw_rows = 0
    with (output / "critic-samples.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        for rate in candidates:
            candidate_config = replace(
                ppo_config,
                critic_learning_rate=float(rate),
                update_epochs=1,
            )
            candidate = CentralizedRecurrentCritic(policy_config).to(
                next(critic.parameters()).device
            )
            candidate.load_state_dict(initial_state, strict=True)
            candidate.train()
            optimizer = torch.optim.Adam(
                candidate.parameters(),
                lr=rate,
                eps=candidate_config.adam_epsilon,
            )
            before = {
                name: value.detach().clone() for name, value in candidate.state_dict().items()
            }
            pre = compute_critic_batch(candidate, chunks, candidate_config)
            pre_mismatch = float((pre.predicted_values - pre.old_values).abs().max().cpu())
            optimizer.zero_grad(set_to_none=True)
            pre.critic.backward()
            if not _finite_gradients(candidate):
                raise ValueError("missing or nonfinite critic calibration gradient")
            gradient_norm = nn.utils.clip_grad_norm_(
                candidate.parameters(), candidate_config.max_gradient_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise ValueError("nonfinite critic calibration gradient norm")
            optimizer.step()
            with torch.no_grad():
                post = compute_critic_batch(candidate, chunks, candidate_config)
            delta = post.predicted_values - post.old_values
            absolute_delta = delta.abs()
            clipped = absolute_delta > candidate_config.value_clip_range
            value_clip_fraction = float(clipped.float().mean().cpu())
            parameter_change = _maximum_parameter_change(before, candidate)
            guidance = {
                "pre_predictions_reproduce_rollout_values": pre_mismatch
                <= calibration_config.pre_prediction_tolerance,
                "post_value_clip_fraction_within_guidance": value_clip_fraction
                <= calibration_config.max_value_clip_fraction,
            }
            result = {
                "critic_learning_rate": float(rate),
                "sample_count": int(delta.numel()),
                "pre_prediction_max_abs_error": pre_mismatch,
                "critic_gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "critic_parameter_max_abs_change": parameter_change,
                "pre_value_loss": float(pre.value.detach().cpu()),
                "post_value_loss": float(post.value.detach().cpu()),
                "pre_explained_variance": float(pre.explained_variance.detach().cpu()),
                "post_explained_variance": float(post.explained_variance.detach().cpu()),
                "post_value_clip_fraction": value_clip_fraction,
                "post_predictions": _distribution(post.predicted_values),
                "value_delta": _distribution(delta),
                "absolute_value_delta": _distribution(absolute_delta),
                "residual": _distribution(post.returns - post.predicted_values),
                "guidance": guidance,
            }
            candidate_results.append(result)

            columns = tuple(
                tensor.detach().cpu()
                for tensor in (
                    environment_ids,
                    rollout_steps,
                    pre.team_rewards,
                    pre.old_values,
                    pre.returns,
                    pre.advantages,
                    pre.predicted_values,
                    post.predicted_values,
                    delta,
                    absolute_delta,
                    clipped,
                )
            )
            for values in zip(*columns, strict=True):
                writer.writerow(
                    {
                        "critic_learning_rate": rate,
                        "environment_id": int(values[0]),
                        "rollout_step": int(values[1]),
                        "team_reward": float(values[2]),
                        "old_value": float(values[3]),
                        "return": float(values[4]),
                        "advantage": float(values[5]),
                        "pre_prediction": float(values[6]),
                        "post_prediction": float(values[7]),
                        "value_delta": float(values[8]),
                        "absolute_value_delta": float(values[9]),
                        "value_clipped": bool(values[10]),
                    }
                )
                raw_rows += 1
            stream.flush()
            del candidate, optimizer

    all_numbers = [
        value
        for result in candidate_results
        for value in (
            result["pre_prediction_max_abs_error"],
            result["critic_gradient_norm_before_clip"],
            result["critic_parameter_max_abs_change"],
            result["pre_value_loss"],
            result["post_value_loss"],
            result["pre_explained_variance"],
            result["post_explained_variance"],
            result["post_value_clip_fraction"],
            *result["value_delta"].values(),
        )
        if isinstance(value, (float, int))
    ]
    passing = [
        result["critic_learning_rate"]
        for result in candidate_results
        if all(result["guidance"].values())
    ]
    checks = {
        "candidate_count_is_exact": len(candidate_results) == len(candidates),
        "each_candidate_used_all_samples": all(
            result["sample_count"] == expected_samples for result in candidate_results
        ),
        "raw_row_count_is_exact": raw_rows == expected_samples * len(candidates),
        "source_and_candidate_metrics_are_finite": all(
            math.isfinite(float(value)) for value in all_numbers
        ),
        "all_candidates_started_from_rollout_values": all(
            result["guidance"]["pre_predictions_reproduce_rollout_values"]
            for result in candidate_results
        ),
        "candidate_parameter_changes_are_positive": all(
            result["critic_parameter_max_abs_change"] > 0 for result in candidate_results
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "matched_rollout": True,
        "candidate_count": len(candidates),
        "sample_count_per_candidate": expected_samples,
        "raw_rows": raw_rows,
        "input_distributions": input_distributions,
        "candidates": candidate_results,
        "rates_within_value_clip_guidance": passing,
        "largest_rate_within_value_clip_guidance": passing[0] if passing else None,
        "guidance_is_acceptance_gate": False,
    }
