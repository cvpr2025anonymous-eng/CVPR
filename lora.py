import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from safetensors import safe_open
from safetensors.torch import save_file


class LoRALinear(nn.Module):


    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"base_layer must be nn.Linear, got {type(base_layer)}")
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        self.reset_parameters()


        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> Tensor:

        delta_w = torch.matmul(self.lora_B.weight, self.lora_A.weight)
        delta_w = delta_w.to(dtype=self.base.weight.dtype, device=self.base.weight.device)
        return self.base.weight + self.scale * delta_w

    @property
    def bias(self) -> Optional[Tensor]:
        return self.base.bias

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))


class LoRA_qkv(nn.Module):


    def __init__(
        self,
        qkv: nn.Linear,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(qkv, nn.Linear):
            raise TypeError(f"Expected qkv to be nn.Linear, got {type(qkv)}")

        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v

        self.d_model = qkv.in_features
        self.rank = linear_a_q.out_features
        self.scale = alpha / self.rank

        if qkv.out_features != 3 * self.d_model:
            raise ValueError(
                f"Expected qkv.out_features == 3 * in_features, "
                f"got out_features={qkv.out_features}, in_features={qkv.in_features}"
            )


        self.qkv.weight.requires_grad = False
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.qkv(x)

        if qkv.shape[-1] != 3 * self.d_model:
            raise ValueError(
                f"Expected last dim {3 * self.d_model}, got {qkv.shape[-1]}"
            )

        q_delta = self.linear_b_q(self.linear_a_q(x)) * self.scale
        v_delta = self.linear_b_v(self.linear_a_v(x)) * self.scale

        original_shape = qkv.shape
        qkv = qkv.view(*original_shape[:-1], 3, self.d_model)
        qkv[..., 0, :] = qkv[..., 0, :] + q_delta
        qkv[..., 2, :] = qkv[..., 2, :] + v_delta
        qkv = qkv.view(*original_shape)

        return qkv


class LoRA_sam(nn.Module):


    def __init__(
        self,
        sam_model,
        rank: int = 4,
        alpha: float = 1.0,
        lora_layer: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.sam = sam_model
        self.rank = rank
        self.alpha = alpha

        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        if not hasattr(self.sam, "image_encoder"):
            raise AttributeError("sam_model must have `image_encoder`")
        if not hasattr(self.sam.image_encoder, "blocks"):
            raise AttributeError("sam_model.image_encoder must have `blocks`")

        if lora_layer is None:
            self.lora_layer = list(range(len(self.sam.image_encoder.blocks)))
        else:
            self.lora_layer = list(lora_layer)

        self.A_weights = nn.ModuleList()
        self.B_weights = nn.ModuleList()


        for p in self.sam.image_encoder.parameters():
            p.requires_grad = False

        for layer_idx, blk in enumerate(self.sam.image_encoder.blocks):
            if layer_idx not in self.lora_layer:
                continue

            if not hasattr(blk, "attn") or not hasattr(blk.attn, "qkv"):
                raise AttributeError(f"Block {layer_idx} does not have attn.qkv")

            qkv_linear = blk.attn.qkv
            if not isinstance(qkv_linear, nn.Linear):
                raise TypeError(
                    f"Block {layer_idx} attn.qkv must be nn.Linear, got {type(qkv_linear)}"
                )

            d_model = qkv_linear.in_features

            a_q = nn.Linear(d_model, self.rank, bias=False)
            b_q = nn.Linear(self.rank, d_model, bias=False)
            a_v = nn.Linear(d_model, self.rank, bias=False)
            b_v = nn.Linear(self.rank, d_model, bias=False)

            nn.init.kaiming_uniform_(a_q.weight, a=np.sqrt(5))
            nn.init.zeros_(b_q.weight)
            nn.init.kaiming_uniform_(a_v.weight, a=np.sqrt(5))
            nn.init.zeros_(b_v.weight)

            self.A_weights.append(a_q)
            self.B_weights.append(b_q)
            self.A_weights.append(a_v)
            self.B_weights.append(b_v)

            blk.attn.qkv = LoRA_qkv(
                qkv=qkv_linear,
                linear_a_q=a_q,
                linear_b_q=b_q,
                linear_a_v=a_v,
                linear_b_v=b_v,
                alpha=self.alpha,
            )

    def forward(self, *args, **kwargs):
        return self.sam(*args, **kwargs)

    def save_lora_parameters(self, filename: str) -> None:
        tensors = {}
        for i, m in enumerate(self.A_weights):
            tensors[f"sam_w_a_{i:03d}"] = m.weight.detach().cpu()
        for i, m in enumerate(self.B_weights):
            tensors[f"sam_w_b_{i:03d}"] = m.weight.detach().cpu()
        save_file(tensors, filename)

    def load_lora_parameters(self, filename: str) -> None:
        with safe_open(filename, framework="pt", device="cpu") as f:
            for i, m in enumerate(self.A_weights):
                key = f"sam_w_a_{i:03d}"
                m.weight = nn.Parameter(
                    f.get_tensor(key).to(m.weight.device, dtype=m.weight.dtype)
                )
            for i, m in enumerate(self.B_weights):
                key = f"sam_w_b_{i:03d}"
                m.weight = nn.Parameter(
                    f.get_tensor(key).to(m.weight.device, dtype=m.weight.dtype)
                )


class LoRA_CLIP_Text(nn.Module):


    def __init__(
        self,
        clip_backbone,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.clip_backbone = clip_backbone
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lora_modules = nn.ModuleList()

        if not hasattr(self.clip_backbone, "transformer"):
            raise AttributeError("clip_backbone must have `transformer`")
        if not hasattr(self.clip_backbone.transformer, "resblocks"):
            raise AttributeError("clip_backbone.transformer must have `resblocks`")


        for p in self.clip_backbone.parameters():
            p.requires_grad = False


        if hasattr(self.clip_backbone, "visual"):
            for p in self.clip_backbone.visual.parameters():
                p.requires_grad = False


        for block in self.clip_backbone.transformer.resblocks:



            if isinstance(block.attn.out_proj, nn.Linear):
                block.attn.out_proj = LoRALinear(
                    block.attn.out_proj,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=0.0,
                )
                self.lora_modules.append(block.attn.out_proj)

            if hasattr(block.mlp, "c_fc") and isinstance(block.mlp.c_fc, nn.Linear):
                block.mlp.c_fc = LoRALinear(
                    block.mlp.c_fc,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                self.lora_modules.append(block.mlp.c_fc)

            if hasattr(block.mlp, "c_proj") and isinstance(block.mlp.c_proj, nn.Linear):
                block.mlp.c_proj = LoRALinear(
                    block.mlp.c_proj,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                self.lora_modules.append(block.mlp.c_proj)

    def save_lora_parameters(self, filename: str) -> None:
        tensors = {}
        for i, m in enumerate(self.lora_modules):
            tensors[f"clip_lora_A_{i:03d}"] = m.lora_A.weight.detach().cpu()
            tensors[f"clip_lora_B_{i:03d}"] = m.lora_B.weight.detach().cpu()
        save_file(tensors, filename)

    def load_lora_parameters(self, filename: str) -> None:
        with safe_open(filename, framework="pt", device="cpu") as f:
            for i, m in enumerate(self.lora_modules):
                m.lora_A.weight = nn.Parameter(
                    f.get_tensor(f"clip_lora_A_{i:03d}").to(
                        m.lora_A.weight.device, dtype=m.lora_A.weight.dtype
                    )
                )
                m.lora_B.weight = nn.Parameter(
                    f.get_tensor(f"clip_lora_B_{i:03d}").to(
                        m.lora_B.weight.device, dtype=m.lora_B.weight.dtype
                    )
                )