"""In-process XVLA policy adapter matching OpenPI's ``infer(obs, noise)`` API.

Uses the official Libero packing/postprocess from
``xvla/evaluation/libero/action_processor.py`` (same contract as
``libero_client.ClientModel``), but calls ``XVLA.generate_actions`` directly
instead of the HTTP deploy server.
"""
from __future__ import annotations

import math
import pathlib
import sys
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

_XVLA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "xvla"
if str(_XVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_XVLA_ROOT))

from evaluation.libero.action_processor import (  # noqa: E402
    LiberoAbsActionProcessor,
    flip_agentview,
)
from models.modeling_xvla import XVLA  # noqa: E402
from models.processing_xvla import XVLAProcessor  # noqa: E402


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Robosuite-compatible quaternion (x,y,z,w) → axis-angle."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return np.asarray(x.detach().cpu().numpy())
    return np.asarray(x)


def _as_writable_float32(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr


class XVLAPolicy:
    """PyTorch XVLA wrapped for DSRL / train_utils_sim ``agent_dp.infer``."""

    def __init__(
        self,
        model: XVLA,
        processor: XVLAProcessor,
        *,
        device: str | torch.device = "cuda",
        domain_id: int = 3,
        steps: int = 10,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.processor = processor
        self.domain_id = int(domain_id)
        self.steps = int(steps)
        self._action_processor = LiberoAbsActionProcessor()

        self.action_horizon = int(model.num_actions)
        self.action_dim = int(model.action_space.dim_action)
        # Env-facing Libero action dim after rot6d → axis-angle postprocess.
        self.env_action_dim = 7

        self._proprio: Optional[np.ndarray] = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "2toINF/X-VLA-Libero",
        *,
        processor_path: str | None = None,
        device: str | torch.device = "cuda",
        domain_id: int = 3,
        steps: int = 10,
        torch_dtype: torch.dtype = torch.float32,
    ) -> "XVLAPolicy":
        processor_path = processor_path or model_path
        model = XVLA.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        processor = XVLAProcessor.from_pretrained(processor_path)
        return cls(
            model,
            processor,
            device=device,
            domain_id=domain_id,
            steps=steps,
        )

    def reset(self) -> None:
        """Clear stateful proprio (call at each episode start)."""
        self._proprio = None

    def infer(self, obs: dict, noise: Any = None, proprio_from_step: int | None = None) -> dict:
        """Run XVLA inference.

        Args:
            obs: Libero-oriented dict. Accepted keys (any one style):
              - Raw env: ``agentview_image``, ``robot0_eye_in_hand_image``,
                optional ``robo_pos``/``robo_ori``, or eef pose keys.
              - Prepared: ``image0``, ``image1``, ``robo_pos``, ``robo_ori``, ``prompt``.
              - OpenPI-like: ``observation/image``, ``observation/wrist_image``,
                ``observation/state`` (8-D: pos3+aa3+grip), ``prompt``.
            noise: optional ``(T, D)`` / ``(B, T, D)`` array. Padded/truncated to
                ``(B, action_horizon, action_dim)``. If ``None``, XVLA samples
                its own noise.
            proprio_from_step: which predicted timestep's abs pose to write into
                stateful proprio after this call. Defaults to the last step
                (``-1``). When only the first ``K`` actions will be executed,
                pass ``K - 1`` so proprio matches the robot, not the unused tail.

        Returns:
            ``{"actions": np.ndarray}`` with shape ``(T, 7)`` if B==1 else ``(B, T, 7)``.
        """
        packed = self._pack_obs(obs)
        B = packed["batch_size"]

        images_batch = []
        for i in range(B):
            main = flip_agentview(packed["image0"][i])
            wrist = packed["image1"][i]
            images_batch.append(
                [Image.fromarray(np.ascontiguousarray(main)), Image.fromarray(np.ascontiguousarray(wrist))]
            )

        prompts = packed["prompt"]
        if isinstance(prompts, str):
            language = [prompts] * B
        else:
            language = list(prompts)

        inputs = self.processor(images_batch, language)
        device = self.device
        dtype = next(self.model.parameters()).dtype

        def to_model(t: torch.Tensor) -> torch.Tensor:
            if t.is_floating_point():
                return t.to(device=device, dtype=dtype)
            return t.to(device=device)

        model_inputs = {k: to_model(v) for k, v in inputs.items()}

        proprio = self._build_proprio(packed["robo_pos"], packed["robo_ori"], B)
        model_inputs["proprio"] = to_model(torch.as_tensor(proprio))
        model_inputs["domain_id"] = torch.full((B,), self.domain_id, dtype=torch.long, device=device)

        noise_t = self._prepare_noise(noise, B, device, dtype)

        with torch.no_grad():
            raw = self.model.generate_actions(
                **model_inputs,
                steps=self.steps,
                noise=noise_t,
            )
        raw_np = _to_numpy(raw).astype(np.float32)  # (B, T, 20)

        # Stateful proprio from the last *executed* abs pose (ClientModel uses -1
        # because it drains the full chunk before the next query).
        if B == 1:
            if self._proprio is None:
                self._proprio = proprio[0].copy()
            step_idx = -1 if proprio_from_step is None else int(proprio_from_step)
            self._proprio[:9] = raw_np[0, step_idx, :9].copy()

        actions = self._postprocess_actions(raw_np)  # (B, T, 7)
        if B == 1:
            actions = actions[0]
        return {"actions": actions}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_proprio(
        self,
        robo_pos: np.ndarray,
        robo_ori: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """Build 20-D proprio; use stateful copy for sequential B=1 rollouts."""
        if batch_size == 1 and self._proprio is not None:
            return self._proprio[None].astype(np.float32)

        # [pos3, ori6d, grip0] + zeros → 20
        proprio = np.concatenate(
            [
                robo_pos.astype(np.float32),
                robo_ori.astype(np.float32),
                np.zeros((batch_size, 1), dtype=np.float32),
            ],
            axis=-1,
        )
        proprio = np.concatenate([proprio, np.zeros_like(proprio)], axis=-1)
        assert proprio.shape == (batch_size, 20), proprio.shape
        return proprio

    def _prepare_noise(
        self,
        noise: Any,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if noise is None:
            return None
        noise_np = _as_writable_float32(noise)
        if noise_np.ndim == 2:
            noise_np = noise_np[None, ...]
        if noise_np.ndim != 3:
            raise ValueError(f"noise must be (T,D) or (B,T,D); got {noise_np.shape}")
        if noise_np.shape[0] == 1 and batch_size > 1:
            noise_np = np.repeat(noise_np, batch_size, axis=0)
        if noise_np.shape[0] != batch_size:
            raise ValueError(
                f"noise batch {noise_np.shape[0]} != obs batch {batch_size}"
            )

        T_need, D_need = self.action_horizon, self.action_dim
        T_cur, D_cur = noise_np.shape[1], noise_np.shape[2]

        if T_cur < T_need:
            pad = np.repeat(noise_np[:, -1:, :], T_need - T_cur, axis=1)
            noise_np = np.concatenate([noise_np, pad], axis=1)
        elif T_cur > T_need:
            noise_np = noise_np[:, :T_need, :]

        if D_cur < D_need:
            pad = np.zeros((batch_size, T_need, D_need - D_cur), dtype=np.float32)
            noise_np = np.concatenate([noise_np, pad], axis=-1)
        elif D_cur > D_need:
            noise_np = noise_np[:, :, :D_need]

        return torch.from_numpy(noise_np).to(device=device, dtype=dtype)

    def _postprocess_actions(self, raw: np.ndarray) -> np.ndarray:
        """(B, T, 20) ee6d → (B, T, 7) abs Libero actions with discrete gripper."""
        B, T, _ = raw.shape
        out = np.zeros((B, T, 7), dtype=np.float32)
        for b in range(B):
            # First arm: pos3 + rot6d + grip1
            arm = raw[b, :, :10]
            aa = self._action_processor.Rotate6D_to_AxisAngle(arm[:, 3:9])
            grip = arm[:, 9:10]
            act = np.concatenate([arm[:, :3], aa, grip], axis=-1)
            act[:, -1] = np.where(act[:, -1] > 0.5, 1.0, -1.0)
            out[b] = act.astype(np.float32)
        return out

    def _pack_obs(self, obs: dict) -> dict:
        """Normalize heterogeneous obs dicts into batched image/pose/prompt."""
        # Prepared / raw style
        if "image0" in obs or "agentview_image" in obs or "observation/image" in obs:
            pass
        else:
            raise KeyError(
                f"Unrecognized obs keys for XVLAPolicy: {list(obs.keys())}"
            )

        if "image0" in obs:
            image0 = _to_numpy(obs["image0"])
            image1 = _to_numpy(obs["image1"])
            robo_pos = _to_numpy(obs["robo_pos"]).astype(np.float32)
            robo_ori = _to_numpy(obs["robo_ori"]).astype(np.float32)
            prompt = obs.get("prompt", "")
        elif "agentview_image" in obs:
            image0 = _to_numpy(obs["agentview_image"])
            image1 = _to_numpy(obs["robot0_eye_in_hand_image"])
            if "robo_pos" in obs and "robo_ori" in obs:
                robo_pos = _to_numpy(obs["robo_pos"]).astype(np.float32)
                robo_ori = _to_numpy(obs["robo_ori"]).astype(np.float32)
            else:
                robo_pos = _to_numpy(obs["robot0_eef_pos"]).astype(np.float32)
                aa = _quat2axisangle(_to_numpy(obs["robot0_eef_quat"]))
                robo_ori = self._action_processor.AxisAngle_to_Rotate6D(
                    np.asarray(aa, dtype=np.float64)
                ).astype(np.float32)
            prompt = obs.get("prompt", obs.get("language_instruction", ""))
        else:
            # OpenPI-like / DSRL replay keys
            image0 = _to_numpy(obs["observation/image"])
            image1 = _to_numpy(obs["observation/wrist_image"])
            state = _to_numpy(obs["observation/state"]).astype(np.float32)
            prompt = obs.get("prompt", "")
            if state.ndim == 1:
                robo_pos = state[:3]
                robo_ori = self._action_processor.AxisAngle_to_Rotate6D(
                    state[3:6].astype(np.float64)
                ).astype(np.float32)
            else:
                robo_pos = state[:, :3]
                robo_ori = self._action_processor.AxisAngle_to_Rotate6D(
                    state[:, 3:6].astype(np.float64)
                ).astype(np.float32)

        # Ensure batch dims: images (B,H,W,C), pos (B,3), ori (B,6)
        if image0.ndim == 3:
            image0 = image0[None, ...]
            image1 = image1[None, ...]
        if robo_pos.ndim == 1:
            robo_pos = robo_pos[None, ...]
            robo_ori = robo_ori[None, ...]

        image0 = np.ascontiguousarray(image0)
        image1 = np.ascontiguousarray(image1)
        if image0.dtype != np.uint8:
            image0 = np.clip(image0, 0, 255).astype(np.uint8)
        if image1.dtype != np.uint8:
            image1 = np.clip(image1, 0, 255).astype(np.uint8)

        return {
            "image0": image0,
            "image1": image1,
            "robo_pos": robo_pos.astype(np.float32),
            "robo_ori": robo_ori.astype(np.float32),
            "prompt": prompt,
            "batch_size": int(image0.shape[0]),
        }
