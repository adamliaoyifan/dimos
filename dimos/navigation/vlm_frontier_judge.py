# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VLM-as-judge harness for frontier selection during room exit.

Renders the occupancy grid as a top-down image, annotates candidate frontiers
with numbered circles, and sends the image to a Qwen3-VL-8B server to get
confidence-calibrated exit direction scores.

This is the core "harness engineering" component: it converts geometric
frontier candidates into semantically grounded decisions by having the VLM
reason over the 2D map topology just as a human would when reading a floor
plan.

Architecture (Generator → Evaluator pattern):
    geometric scorer   →  top-K frontier candidates  (Generator)
    VLMFrontierJudge   →  confidence per frontier     (Evaluator)
    score fusion       →  best frontier waypoint      (Decision)

Dependencies:
    - numpy (always available)
    - Pillow (PIL) — available as a transitive dep of OpenCV / torchvision
    - httpx — used by Qwen3LocalVlModel, so always present
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.geometry_msgs.Vector3 import Vector3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour palette for frontier markers (BGR → RGB)
# ---------------------------------------------------------------------------
_FRONTIER_COLORS = [
    (255, 80, 80),    # red-ish
    (80, 200, 80),    # green
    (80, 80, 255),    # blue
    (255, 200, 0),    # amber
    (200, 0, 200),    # magenta
    (0, 200, 200),    # cyan
    (255, 140, 0),    # orange
    (0, 140, 255),    # sky-blue
]


@dataclass
class FrontierJudgement:
    """Per-frontier score returned by the VLM."""

    frontier_index: int
    confidence: float
    """VLM confidence that this frontier leads out of the room [0, 1]."""
    reason: str = ""

    def __repr__(self) -> str:
        return f"FJ(idx={self.frontier_index}, conf={self.confidence:.2f})"


@dataclass
class JudgeResult:
    """Full result from ``VLMFrontierJudge.judge_frontiers``."""

    judgements: list[FrontierJudgement]
    """Per-frontier confidence scores, sorted descending by confidence."""

    vlm_raw_response: str = ""
    """Raw text returned by the VLM (for logging/debugging)."""

    fell_back_to_uniform: bool = False
    """True if the VLM call failed and uniform scores were used instead."""

    def confidence_map(self) -> dict[int, float]:
        """Return {frontier_index: confidence} mapping."""
        return {j.frontier_index: j.confidence for j in self.judgements}


class VLMFrontierJudge:
    """Evaluate frontier candidates using Qwen3-VL-8B visual reasoning.

    The judge renders the occupancy grid as an annotated top-down image and
    asks the VLM to rank candidate frontiers by their likelihood of leading
    OUT of the current room.

    Args:
        vlm_base_url: Base URL of the Qwen3-VL FastAPI server
            (e.g. ``"http://192.168.2.109:8000"``).
        model_name: Model identifier string (informational only for logging).
        timeout_s: Maximum seconds to wait for the VLM response before
            falling back to uniform scores.
        render_scale: Pixels per grid cell in the rendered image.
        obstacle_threshold: Costmap values >= this are treated as lethal
            obstacles (rendered black).
        vlm_judge_top_k: Maximum number of frontier candidates to show the
            VLM.  Excess frontiers are truncated (lowest-scoring first).
    """

    _SYSTEM_PROMPT = (
        "You are a robot navigation advisor analyzing a top-down occupancy grid map. "
        "White pixels represent free space the robot has already traversed. "
        "Black pixels represent obstacles or walls. "
        "Gray pixels represent unexplored unknown territory. "
        "The robot (red filled circle) is currently in the white region and needs "
        "to EXIT this area — it has already explored the white region and found "
        "no target object there. "
        "Numbered colored circles mark candidate frontier waypoints at the boundary "
        "of explored and unexplored space."
    )

    _USER_PROMPT_TEMPLATE = (
        "The robot needs to exit the current room and enter an adjacent corridor "
        "or room to continue searching.\n\n"
        "Candidate frontier waypoints are labeled 1 through {n}.\n\n"
        "Think step-by-step:\n"
        "1. Describe the overall shape of the white (explored) region — "
        "is it a large open room, a long corridor, or a complex shape?\n"
        "2. For each numbered frontier, describe whether it is at a narrow passage "
        "(likely doorway/corridor) or a wide open boundary (likely interior wall).\n"
        "3. Rank the frontiers by how likely each is to lead OUT of this room "
        "into adjacent space.\n\n"
        "Respond with valid JSON only — a list ordered from most to least promising:\n"
        '[{{"id": 1, "confidence": 0.85, "reason": "brief reason"}}, ...]\n\n'
        "Include all {n} frontiers. Confidence values must sum to approximately 1.0."
    )

    def __init__(
        self,
        vlm_base_url: str = "http://192.168.2.109:8000",
        model_name: str = "Qwen3-VL-8B-Instruct",
        timeout_s: float = 8.0,
        render_scale: int = 4,
        obstacle_threshold: int = 99,
        vlm_judge_top_k: int = 5,
    ) -> None:
        self.vlm_base_url = vlm_base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.render_scale = render_scale
        self.obstacle_threshold = obstacle_threshold
        self.vlm_judge_top_k = vlm_judge_top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_costmap_with_frontiers(
        self,
        grid: OccupancyGrid,
        frontiers: list[Vector3],
        robot_pos: Vector3,
        trail: list[tuple[float, float]] | None = None,
    ) -> bytes:
        """Render the occupancy grid as an annotated PNG image.

        Cell encoding:
            - FREE  (grid == 0):   white (255)
            - UNKNOWN (grid == -1): medium gray (128)
            - OBSTACLE (grid >= threshold): dark (30)

        Overlays:
            - Blue polyline: exploration trail (entry path)
            - Red filled circle: robot position
            - Numbered colored circles: frontier candidates

        Args:
            grid: Current occupancy grid.
            frontiers: Candidate frontier waypoints (world frame).
            robot_pos: Current robot position (world frame).
            trail: Optional exploration trail for entry-path visualisation.

        Returns:
            PNG image bytes (suitable for base64 encoding).
        """
        from PIL import Image as PILImage, ImageDraw, ImageFont  # type: ignore[import]

        h, w = grid.grid.shape
        scale = self.render_scale
        img_w, img_h = w * scale, h * scale

        # Build RGBA canvas
        canvas = np.full((img_h, img_w, 3), 128, dtype=np.uint8)  # default gray

        raw = grid.grid.astype(np.int16)  # avoid int8 overflow in comparisons

        # Paint free cells white, obstacle cells dark
        free_mask = raw == 0
        obs_mask = raw >= self.obstacle_threshold

        for gy in range(h):
            for gx in range(w):
                px = gx * scale
                py = gy * scale
                if free_mask[gy, gx]:
                    canvas[py : py + scale, px : px + scale] = 255
                elif obs_mask[gy, gx]:
                    canvas[py : py + scale, px : px + scale] = 30

        # Vectorised version of above (faster for large grids)
        # The loop above is kept for clarity; replace with this for perf:
        # _paint_grid_vectorised(canvas, raw, scale, self.obstacle_threshold)

        pil_img = PILImage.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(pil_img)

        def world_to_px(wx: float, wy: float) -> tuple[int, int]:
            gv = grid.world_to_grid(Vector3(wx, wy, 0.0))
            px = int(gv.x * scale + scale / 2)
            py = int(gv.y * scale + scale / 2)
            return (
                max(0, min(img_w - 1, px)),
                max(0, min(img_h - 1, py)),
            )

        # Draw exploration trail
        if trail and len(trail) >= 2:
            trail_px = [world_to_px(tx, ty) for tx, ty in trail]
            draw.line(trail_px, fill=(100, 100, 255), width=max(1, scale))

        # Draw robot position
        rx, ry = world_to_px(robot_pos.x, robot_pos.y)
        r = max(3, scale * 2)
        draw.ellipse(
            (rx - r, ry - r, rx + r, ry + r),
            fill=(220, 30, 30),
            outline=(255, 255, 255),
        )

        # Draw frontier markers
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=max(10, scale * 3))
        except OSError:
            font = ImageFont.load_default()

        for i, frontier in enumerate(frontiers[: self.vlm_judge_top_k]):
            color = _FRONTIER_COLORS[i % len(_FRONTIER_COLORS)]
            fx, fy = world_to_px(frontier.x, frontier.y)
            fr = max(4, scale * 2)
            draw.ellipse(
                (fx - fr, fy - fr, fx + fr, fy + fr),
                fill=color,
                outline=(255, 255, 255),
            )
            label = str(i + 1)
            draw.text((fx - fr // 2, fy - fr // 2), label, fill=(255, 255, 255), font=font)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    def judge_frontiers(
        self,
        grid: OccupancyGrid | None,
        frontiers: list[Vector3],
        robot_pos: Vector3,
        trail: list[tuple[float, float]] | None = None,
    ) -> JudgeResult:
        """Query Qwen3-VL-8B to rank frontier candidates by exit probability.

        Renders the costmap, constructs a chain-of-thought prompt, sends to
        the VLM server, and parses the JSON response.  Falls back to uniform
        confidence scores on any error so navigation continues gracefully.

        When ``grid`` is None, the costmap rendering step is skipped and the
        VLM receives a text-only prompt (no image), which typically causes a
        fallback to uniform scores.

        Args:
            grid: Current occupancy grid, or None if unavailable.
            frontiers: Candidate frontier waypoints (world frame), already
                filtered/ranked by the geometric scorer.  At most
                ``vlm_judge_top_k`` will be shown to the VLM.
            robot_pos: Current robot position (world frame).
            trail: Optional exploration trail for image overlay.

        Returns:
            JudgeResult with per-frontier confidence scores.
        """
        candidates = frontiers[: self.vlm_judge_top_k]
        n = len(candidates)

        if n == 0:
            return JudgeResult(judgements=[], fell_back_to_uniform=True)

        if n == 1:
            return JudgeResult(
                judgements=[FrontierJudgement(0, 1.0, "only candidate")],
                fell_back_to_uniform=False,
            )

        # Render costmap — skip gracefully if grid is unavailable
        if grid is None:
            logger.info("VLMFrontierJudge: no costmap available, falling back to uniform scores")
            return self._uniform_result(n)

        try:
            png_bytes = self.render_costmap_with_frontiers(
                grid, candidates, robot_pos, trail
            )
        except Exception as exc:
            logger.warning("VLMFrontierJudge: render failed: %s", exc)
            return self._uniform_result(n)

        # Build prompt
        prompt = (
            self._SYSTEM_PROMPT
            + "\n\n"
            + self._USER_PROMPT_TEMPLATE.format(n=n)
        )

        # Call VLM
        try:
            raw_response = self._call_vlm(png_bytes, prompt)
        except Exception as exc:
            logger.warning("VLMFrontierJudge: VLM call failed: %s", exc)
            return self._uniform_result(n)

        # Parse response
        judgements = self._parse_response(raw_response, n)
        if judgements is None:
            logger.warning(
                "VLMFrontierJudge: failed to parse VLM response, using uniform scores. "
                "Raw: %.200s", raw_response
            )
            result = self._uniform_result(n)
            result.vlm_raw_response = raw_response
            return result

        return JudgeResult(
            judgements=sorted(judgements, key=lambda j: j.confidence, reverse=True),
            vlm_raw_response=raw_response,
            fell_back_to_uniform=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_vlm(self, png_bytes: bytes, prompt: str) -> str:
        """POST image + prompt to the Qwen3-VL server.

        Uses the same ``/v1/image-inference-base64`` endpoint as
        ``Qwen3LocalVlModel.query()``.
        """
        import httpx

        img_b64 = base64.b64encode(png_bytes).decode("utf-8")

        payload: dict[str, Any] = {
            "image_base64": img_b64,
            "prompt": prompt,
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "top_p": 0.9,
        }

        url = f"{self.vlm_base_url}/v1/image-inference-base64"
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()

        data = resp.json()
        if data.get("status") != "success":
            error = data.get("error") or data.get("detail") or str(data)
            raise RuntimeError(f"Qwen3-VL server error: {error}")

        return str(data["response"])

    def _parse_response(
        self, raw: str, expected_n: int
    ) -> list[FrontierJudgement] | None:
        """Extract frontier judgements from the VLM's JSON response.

        Handles:
        - Pure JSON arrays
        - JSON embedded in markdown code fences (```json ... ```)
        - Partial responses with fewer than ``expected_n`` entries

        Args:
            raw: Raw VLM response text.
            expected_n: Expected number of frontier entries.

        Returns:
            List of FrontierJudgement, or None on unrecoverable parse failure.
        """
        # Strip thinking tags (Qwen3 CoT models emit <think>...</think>)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Try to extract JSON array
        json_str = self._extract_json_array(raw)
        if json_str is None:
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, list):
            return None

        judgements: list[FrontierJudgement] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            raw_id = entry.get("id")
            raw_conf = entry.get("confidence", 0.5)
            reason = str(entry.get("reason", ""))
            if raw_id is None:
                continue
            try:
                idx = int(raw_id) - 1  # 1-indexed in prompt, 0-indexed internally
                conf = float(raw_conf)
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                continue
            if 0 <= idx < expected_n:
                judgements.append(FrontierJudgement(idx, conf, reason))

        if not judgements:
            return None

        # Fill in any missing frontiers with the minimum confidence
        existing_ids = {j.frontier_index for j in judgements}
        if len(existing_ids) < expected_n:
            min_conf = min(j.confidence for j in judgements) * 0.5
            for i in range(expected_n):
                if i not in existing_ids:
                    judgements.append(FrontierJudgement(i, min_conf, "not mentioned"))

        # Normalise so confidences sum to 1
        total = sum(j.confidence for j in judgements)
        if total > 0:
            for j in judgements:
                j.confidence = j.confidence / total

        return judgements

    @staticmethod
    def _extract_json_array(text: str) -> str | None:
        """Find the first JSON array in text, stripping markdown fences."""
        # Try markdown code fence
        fence_match = re.search(r"```(?:json)?\s*(\[.*?])\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1)

        # Try bare JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            return text[start : end + 1]

        return None

    def _uniform_result(self, n: int) -> JudgeResult:
        """Return equal confidence scores for all ``n`` frontiers."""
        uniform_conf = 1.0 / n
        return JudgeResult(
            judgements=[
                FrontierJudgement(i, uniform_conf, "fallback uniform")
                for i in range(n)
            ],
            fell_back_to_uniform=True,
        )


def blend_scores(
    geometric_scores: dict[int, float],
    vlm_confidences: dict[int, float],
    geo_weight: float = 0.6,
    vlm_weight: float = 0.4,
) -> dict[int, float]:
    """Combine geometric frontier scores with VLM confidence scores.

    Both score dicts are normalised to [0, 1] before blending so neither
    dominates due to scale differences.

    Args:
        geometric_scores: {frontier_index: geometric_score} from the scorer.
        vlm_confidences: {frontier_index: vlm_confidence} from the judge.
        geo_weight: Weight for geometric scores (default 0.6).
        vlm_weight: Weight for VLM scores (default 0.4).

    Returns:
        Blended {frontier_index: fused_score} dict.
    """
    all_indices = set(geometric_scores) | set(vlm_confidences)
    if not all_indices:
        return {}

    # Normalise geometric scores
    geo_vals = [geometric_scores.get(i, 0.0) for i in all_indices]
    geo_max = max(geo_vals) if max(geo_vals) > 0 else 1.0
    geo_norm = {i: geometric_scores.get(i, 0.0) / geo_max for i in all_indices}

    # VLM confidences are already normalised (sum to 1), convert to per-item scale
    vlm_max = max(vlm_confidences.values()) if vlm_confidences else 1.0
    vlm_norm = {i: vlm_confidences.get(i, 0.0) / max(vlm_max, 1e-9) for i in all_indices}

    fused = {
        i: geo_weight * geo_norm[i] + vlm_weight * vlm_norm[i]
        for i in all_indices
    }
    return fused


__all__ = [
    "VLMFrontierJudge",
    "FrontierJudgement",
    "JudgeResult",
    "blend_scores",
]
