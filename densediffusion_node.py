from __future__ import annotations
import itertools
import logging
import math
import os
from typing import Literal
from dataclasses import dataclass

import torch

import comfy
from comfy.model_patcher import ModelPatcher

from .enums import StableDiffusionVersion

_dd_logger = logging.getLogger("DenseDiffusion")
_DD_DEBUG = os.environ.get("DENSEDIFFUSION_DEBUG", "").strip().lower() in ("1", "true", "yes")
_dd_debug_logged_for: set = set()  # tracks prompt_ids already logged


ComfyUIConditioning = list  # Dummy type definitions for ComfyUI


def get_sd_version(model: ModelPatcher) -> StableDiffusionVersion:
    """Check if the model is a Stable Diffusion XL model.
    TODO: Make this a part of comfy.model_patcher.ModelPatcher
    """
    is_sdxl = isinstance(
        model.model,
        (
            comfy.model_base.SDXL,
            comfy.model_base.SDXLRefiner,
            comfy.model_base.SDXL_instructpix2pix,
        ),
    )
    return StableDiffusionVersion.SDXL if is_sdxl else StableDiffusionVersion.SD1x

@dataclass
class DenseDiffusionConditioning:
    # Text embeddings
    cond: torch.Tensor
    # The mask to apply. Shape: [H, W]
    mask: torch.Tensor
    pooled_output: torch.Tensor | None = None


class OmostDenseDiffusionCrossAttention(torch.nn.Module):
    @staticmethod
    def calc_mask_cond(
        q: torch.Tensor,
        extra_options: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dd_conds: list[DenseDiffusionConditioning] = extra_options.get(
            "dense_diffusion_cond", []
        )

        cond_or_uncond: list[Literal[0, 1]] = extra_options["cond_or_uncond"]
        assert len(cond_or_uncond) in (1, 2)

        batch_size = q.size(0)
        cond_batch_size = batch_size // len(cond_or_uncond) if 0 in cond_or_uncond else 0
        uncond_batch_size = q.size(0) - cond_batch_size

        if dd_conds and cond_batch_size > 0:
            _, _, latent_height, latent_width = extra_options["original_shape"]
            H, W = OmostDenseDiffusionCrossAttention.calc_hidden_state_shape(
                q.size(2), latent_height, latent_width
            )
            masks = []
            
            for dd_cond in dd_conds:
                mask = dd_cond.mask
                # Handle different mask shapes
                if len(mask.shape) == 2:  # [H, W]
                    mask = mask.unsqueeze(0)  # [1, H, W]
                
                # Ensure batch dimension matches cond_batch_size
                if mask.size(0) == 1:
                    mask = mask.repeat(cond_batch_size, 1, 1)
                elif mask.size(0) != cond_batch_size:
                    mask = mask[:cond_batch_size]  # Take first cond_batch_size masks
                
                # Resize mask in float16: nearest-exact upsample kernels are not
                # implemented for float8 dtypes (e.g. Float8_e4m3fn).
                m = torch.nn.functional.interpolate(
                    mask.to(dtype=torch.float16).unsqueeze(1),  # [B, 1, H, W]
                    (H, W),
                    mode="nearest-exact"
                ).flatten(1)  # [B, H*W]
                
                # Repeat for each embedding dimension
                m = m.unsqueeze(2).repeat(1, 1, dd_cond.cond.size(1))  # [B, H*W, emb_dim]
                masks.append(m)
            
            masks = torch.cat(masks, dim=2)  # Concatenate along embedding dimension
            
            mask_bool = masks > 0.5
            # Prevent Inf scale values when any conditioning channel has near-zero masked area.
            mask_denom = torch.sum(masks, dim=1, keepdim=True).clamp_min(1e-6)
            mask_scale = (H * W) / mask_denom
            
            # Reshape for attention computation
            mask_bool = mask_bool.unsqueeze(1).repeat(1, q.size(1), 1, 1)
            mask_scale = mask_scale.unsqueeze(1).repeat(1, q.size(1), 1, 1)

            # Handle unconditional part
            if uncond_batch_size > 0:
                assert len(cond_or_uncond) == 2
                uncond_first = cond_or_uncond.index(1) == 0
                
                uncond_mask_bool = torch.ones_like(mask_bool[:uncond_batch_size])
                uncond_mask_scale = torch.ones_like(mask_scale[:uncond_batch_size])
                
                if uncond_first:
                    mask_bool = torch.cat([uncond_mask_bool, mask_bool], dim=0)
                    mask_scale = torch.cat([uncond_mask_scale, mask_scale], dim=0)
                else:
                    mask_bool = torch.cat([mask_bool, uncond_mask_bool], dim=0)
                    mask_scale = torch.cat([mask_scale, uncond_mask_scale], dim=0)
                    
            return mask_bool, mask_scale
        return None, None

    @staticmethod
    def calc_hidden_state_shape(
        sequence_length: int, H: int, W: int
    ) -> tuple[int, int]:
        ratio = W / H
        mask_h = int(round(math.sqrt(sequence_length / ratio)))
        mask_w = int(round(math.sqrt(sequence_length * ratio)))
        return mask_h, mask_w

    @staticmethod
    def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_bool: torch.Tensor | None = None,
        mask_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Modified scaled dot product attention that applies mask_bool
        and mask_scale on q@k before softmax calculation.
        """
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_weight = query @ key.transpose(-2, -1) * scale_factor

        def _align_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
            # Ensure the mask has the same rank as attention weights.
            while mask.ndim < attn_weight.ndim:
                mask = mask.unsqueeze(0)

            # Expand/repeat each dimension to match attention shape.
            for dim in range(attn_weight.ndim):
                current = mask.size(dim)
                target = attn_weight.size(dim)
                if current == target:
                    continue
                if current == 1:
                    expand_shape = list(mask.shape)
                    expand_shape[dim] = target
                    mask = mask.expand(*expand_shape)
                    continue
                if target % current == 0:
                    mask = mask.repeat_interleave(target // current, dim=dim)
                    continue

                raise RuntimeError(
                    f"DenseDiffusion {name} shape {tuple(mask.shape)} cannot align with attention shape {tuple(attn_weight.shape)}"
                )

            return mask

        if mask_scale is not None:
            mask_scale = _align_mask(
                mask_scale.to(device=attn_weight.device, dtype=attn_weight.dtype),
                "mask_scale",
            )
            mask_scale = torch.nan_to_num(mask_scale, nan=1.0, posinf=1.0, neginf=1.0)
            attn_weight = attn_weight * mask_scale
        if mask_bool is not None:
            mask_bool = _align_mask(
                mask_bool.to(device=attn_weight.device, dtype=torch.bool),
                "mask_bool",
            )

            # Avoid all -inf rows (which become NaN after softmax) by disabling masking
            # for query rows that would otherwise be fully masked.
            fully_masked = mask_bool.logical_not().all(dim=-1, keepdim=True)
            if fully_masked.any():
                mask_bool = mask_bool | fully_masked

            attn_weight.masked_fill_(mask_bool.logical_not(), float("-inf"))

        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.nan_to_num(attn_weight, nan=0.0, posinf=0.0, neginf=0.0)
        return attn_weight @ value

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, extra_options: dict
    ):
        """
        y=softmax(modify(q@k))@v
        where modify() is a complicated non-linear function with many normalizations
        and tricks to change the score's distributions.

        This function implements the `modify` function used in Omost, instead
        of the original DenseDiffusion repo.

        https://github.com/lllyasviel/Omost/blob/731e74922fc6be91171688574d07624f93d3b658/lib_omost/pipeline.py#L129-L173
        """
        heads: int = extra_options["n_heads"]

        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )
        mask_bool, mask_scale = self.calc_mask_cond(q, extra_options)

        if _DD_DEBUG and mask_bool is not None:
            prompt_id = extra_options.get("prompt_id", "unknown")
            if prompt_id not in _dd_debug_logged_for:
                _dd_debug_logged_for.add(prompt_id)
                # Keep set bounded
                if len(_dd_debug_logged_for) > 50:
                    _dd_debug_logged_for.clear()
                    _dd_debug_logged_for.add(prompt_id)
                with torch.no_grad():
                    mb = mask_bool[0]  # first batch item, shape [heads, H*W, total_T]
                    total_positions = mb.shape[1]
                    # Per-head average across heads → [H*W, total_T]
                    mb_any_head = mb.float().mean(dim=0) > 0.5
                    true_per_pos = mb_any_head.sum(dim=-1)  # tokens True per position
                    fully_true = (true_per_pos == mb_any_head.shape[-1]).sum().item()
                    fully_false = (true_per_pos == 0).sum().item()
                    partial = total_positions - fully_true - fully_false
                    coverage = mb_any_head.float().mean(dim=0)  # per-token coverage
                    _dd_logger.info(
                        "[DenseDiffusion DIAG] prompt=%s | q=%s mask_bool=%s mask_scale=%s | "
                        "positions: fully_open=%d partial=%d fully_masked=%d / %d | "
                        "per_token_coverage(first5)=%s",
                        prompt_id, tuple(q.shape), tuple(mask_bool.shape),
                        tuple(mask_scale.shape) if mask_scale is not None else None,
                        fully_true, partial, fully_false, total_positions,
                        [f"{v:.3f}" for v in coverage[:5].tolist()],
                    )

        out = self.scaled_dot_product_attention(
            q, k, v, mask_bool=mask_bool, mask_scale=mask_scale
        )
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class DenseDiffusionApplyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "apply"
    CATEGORY = "DenseDiffusion"
    DESCRIPTION = "Apply DenseDiffusion model."

    def apply(self, model: ModelPatcher) -> tuple[ModelPatcher]:
        work_model: ModelPatcher = model.clone()
        # TODO: Make patching cross-attn easier/more composable with Unified ModelPatcher.
        # Current approach does not compose with IPAdapter.
        sd_version: StableDiffusionVersion = get_sd_version(work_model)
        input_ids, output_ids, middle_ids = sd_version.transformer_ids
        for transformer_id in itertools.chain(input_ids, output_ids, middle_ids):
            work_model.set_model_attn2_replace(
                OmostDenseDiffusionCrossAttention(),
                block_name=transformer_id.block_type.value,
                number=transformer_id.block_id,
                # transformer_index param here specifies the depth index of the transformer
                transformer_index=transformer_id.block_index,
            )

        dd_conds: list[DenseDiffusionConditioning] = work_model.model_options[
            "transformer_options"
        ].get("dense_diffusion_cond", [])
        assert dd_conds, "No DenseDiffusion conditioning found!"

        # Move all mask to unet device to avoid move during inference.
        for dd_cond in dd_conds:
            dd_cond.mask = dd_cond.mask.to(model.load_device, model.model_dtype())

        cond = [
            [
                # cond
                torch.cat([dd_cond.cond for dd_cond in dd_conds], dim=1),
                # pooled_output
                {"pooled_output": dd_conds[0].pooled_output},
            ]
        ]
        return (work_model, cond)


class DenseDiffusionAddCondNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "append"
    CATEGORY = "DenseDiffusion"
    DESCRIPTION = "Set a regional prompt for DenseDiffusion."

    def append(
        self,
        model: ModelPatcher,
        conditioning: ComfyUIConditioning,
        strength: float,
        mask: torch.Tensor=None,
    ) -> tuple[ModelPatcher]:
        work_model: ModelPatcher = model.clone()
        work_model.model_options["transformer_options"].setdefault(
            "dense_diffusion_cond", []
        )
        assert len(conditioning) == 1
        cond, extra_fields = conditioning[0]
        assert isinstance(extra_fields, dict)
        assert "pooled_output" in extra_fields
        
        if mask == None:
            mask = torch.full((1, 512, 512), 1.0, dtype=torch.float32, device="cpu")

        work_model.model_options["transformer_options"]["dense_diffusion_cond"].append(
            DenseDiffusionConditioning(
                cond=cond,
                mask=mask.squeeze() * strength,
                pooled_output=extra_fields["pooled_output"],
            )
        )
        return (work_model,)


NODE_CLASS_MAPPINGS = {
    "DenseDiffusionApplyNode": DenseDiffusionApplyNode,
    "DenseDiffusionAddCondNode": DenseDiffusionAddCondNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DenseDiffusionApplyNode": "DenseDiffusion Apply",
    "DenseDiffusionAddCondNode": "DenseDiffusion Add Cond",
}
