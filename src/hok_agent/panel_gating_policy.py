"""R1 panel-gating policy: one small temporal model over derived RGB panel views.

R1 is the owner-authorized post-training milestone. The learnable component is a one-observation
ahead panel-state predictor; the deterministic Router turns its prediction into the applied gate,
so the model never sends anything to the device.

Boundaries this module must keep (see AGENTS.md and BOUNDARIES.md):

- The policy input is only the recent derived RGB equipment views. Device coordinates, execution
  timestamps, action records, reward components, event records and the current measured panel state
  are excluded from the input; they may be used only as targets or audit labels.
- Torch is imported lazily inside the training entry point, so loading this module and validating
  the contract stays dependency-free.
- Every artifact keeps ``semantic_accuracy_verified=false`` and ``promotion_allowed=false``, and the
  frozen myopic rule is retained when the candidate shows no pre-declared gain.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

R1_PANEL_GATING_SCHEMA = "hok-agent-r1-panel-gating-contract-v1"
R1_REQUIRED_EXCLUSIONS = (
    "device coordinates",
    "execution timestamps",
    "action records",
    "reward components",
    "event records",
    "the current measured panel state",
)
R1_REQUIRED_REQUIREMENTS = (
    "minimum_candidate_accuracy_gain",
    "maximum_candidate_wrong_phase_dispatch_count",
    "maximum_ambiguous_step_fraction",
)


class PanelGatingError(ValueError):
    """Raised when the R1 contract or its inputs are invalid."""


def load_r1_panel_gating_contract(path: Path) -> tuple[dict[str, object], str]:
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    policy_input = payload.get("policy_input")
    target = payload.get("target")
    baseline = payload.get("baseline")
    data = payload.get("data")
    requirements = payload.get("pre_declared_requirements")
    claim = payload.get("claim_boundary")
    if (
        payload.get("schema_version") != R1_PANEL_GATING_SCHEMA
        or payload.get("status") != "frozen"
        or not isinstance(policy_input, dict)
        or not isinstance(target, dict)
        or not isinstance(baseline, dict)
        or not isinstance(data, dict)
        or not isinstance(requirements, dict)
        or not isinstance(claim, dict)
    ):
        raise PanelGatingError("r1 panel gating contract differs")
    exclusions = {str(item) for item in cast(list[object], policy_input.get("excluded", []))}
    if exclusions != set(R1_REQUIRED_EXCLUSIONS):
        raise PanelGatingError("r1 policy input exclusions differ")
    if policy_input.get("colour_space") != "rgb" or int(cast(int, policy_input.get("k", 0))) < 2:
        raise PanelGatingError("r1 policy input window differs")
    if baseline.get("frozen") is not True or not baseline.get("rule"):
        raise PanelGatingError("r1 baseline must be a frozen rule")
    if any(key not in requirements for key in R1_REQUIRED_REQUIREMENTS):
        raise PanelGatingError("r1 pre-declared requirements differ")
    if (
        float(cast(float, requirements["minimum_candidate_accuracy_gain"])) <= 0.0
        or int(cast(int, requirements["maximum_candidate_wrong_phase_dispatch_count"])) != 0
        or not 0.0 < float(cast(float, requirements["maximum_ambiguous_step_fraction"])) < 1.0
    ):
        raise PanelGatingError("r1 pre-declared requirements are not a real bar")
    if (
        claim.get("comparison_only") is not True
        or claim.get("independent_feedback_verification_claimed") is not False
        or claim.get("promotion_allowed") is not False
        or claim.get("device_input_added") is not False
        or claim.get("semantic_accuracy_verified") is not False
    ):
        raise PanelGatingError("r1 claim boundary differs")
    if payload.get("feedback_verification_class") != "owner_authorized_bar":
        raise PanelGatingError("r1 must carry the owner-authorized feedback class")
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return payload, digest


def panel_input_window(views: list[object], k: int) -> list[object]:
    """Return the last ``k`` derived views, which is the whole policy input."""
    if k < 2:
        raise PanelGatingError("r1 input window is too short")
    if len(views) < k:
        raise PanelGatingError("r1 input window is not full")
    return list(views[-k:])
