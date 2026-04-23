import os
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Any, Dict, Optional, Tuple, Type

try:
    from common import LayerNorm2d
    from clip_saist import build_clip_backbone, SRCLIP
except ImportError:
    from .common import LayerNorm2d
    from .clip_saist import build_clip_backbone, SRCLIP


class MaskPromptFeatureBlock(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        stride: int = 1,
        use_layernorm: bool = True,
        use_spatial_attention: bool = False,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.use_spatial_attention = use_spatial_attention
        self.conv1 = nn.Conv2d(
            in_chans,
            out_chans,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = LayerNorm2d(out_chans) if use_layernorm else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_chans,
            out_chans,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = LayerNorm2d(out_chans) if use_layernorm else nn.Identity()
        self.conv3 = nn.Conv2d(
            out_chans,
            out_chans,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm3 = LayerNorm2d(out_chans) if use_layernorm else nn.Identity()
        self.act = activation()
        if use_spatial_attention:
            self.spatial_att = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        if stride != 1 or in_chans != out_chans:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        if self.use_spatial_attention:
            attn_input = torch.cat(
                [
                    x.mean(dim=1, keepdim=True),
                    x.amax(dim=1, keepdim=True),
                ],
                dim=1,
            )
            x = x * torch.sigmoid(self.spatial_att(attn_input))
        return self.act(x + identity)


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        sr_clip_ckpt_path: str = "../../ViT-B-32.pt",
        sr_clip_image_size: int = 224,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.sr_clip_image_size = sr_clip_image_size

        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

        self.no_mask_embed = nn.Embedding(1, embed_dim)

        self.mask_prompt_stem = nn.Sequential(
            MaskPromptFeatureBlock(3, 16, stride=1, use_layernorm=False, activation=activation),
            MaskPromptFeatureBlock(16, 16, stride=1, use_layernorm=False, activation=activation),
        )
        self.mask_prompt_level0 = MaskPromptFeatureBlock(
            16,
            16,
            stride=1,
            use_layernorm=True,
            use_spatial_attention=False,
            activation=activation,
        )
        self.mask_prompt_level1 = MaskPromptFeatureBlock(
            16,
            32,
            stride=2,
            use_layernorm=True,
            use_spatial_attention=False,
            activation=activation,
        )
        self.mask_prompt_level2 = MaskPromptFeatureBlock(
            32,
            64,
            stride=2,
            use_layernorm=True,
            use_spatial_attention=True,
            activation=activation,
        )

        ckpt_path = sr_clip_ckpt_path
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)
        state_dict = torch.jit.load(ckpt_path, map_location="cpu").state_dict()
        clip_backbone = build_clip_backbone(state_dict)

        self.sr_clip = SRCLIP(
            clip_backbone=clip_backbone,
            prompt_dim=clip_backbone.prompt_dim,
            decoder_dim=embed_dim,
        )

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        points = points + 0.5

        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_mask_inputs(self, masks: torch.Tensor) -> torch.Tensor:
        return self.mask_downscaling(masks)

    def _build_mask_prompt_dense_embeddings(
        self,
        prompt_image: torch.Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mask_prompt_1024_dense_embeddings = self.mask_prompt_level0(
            self.mask_prompt_stem(prompt_image)
        )
        mask_prompt_512_dense_embeddings = self.mask_prompt_level1(
            mask_prompt_1024_dense_embeddings
        )
        mask_prompt_256_dense_embeddings = self.mask_prompt_level2(
            mask_prompt_512_dense_embeddings
        )

        return (
            mask_prompt_256_dense_embeddings,
            mask_prompt_512_dense_embeddings,
            mask_prompt_1024_dense_embeddings,
        )

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        scenes: Optional[torch.Tensor] = None,
    ) -> int:
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif scenes is not None and torch.is_tensor(scenes):
            return scenes.shape[0] if scenes.dim() == 4 else 1
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def _prepare_scene_image(self, scenes: torch.Tensor) -> torch.Tensor:
        if scenes is None:
            raise ValueError("scenes must not be None when using SR-CLIP PromptEncoder.")

        if not torch.is_tensor(scenes):
            raise TypeError(f"scenes must be a torch.Tensor, got {type(scenes)}")

        if scenes.dim() == 3:
            scenes = scenes.unsqueeze(0)
        elif scenes.dim() == 4:
            pass
        else:
            raise ValueError(f"scenes must have dim 3 or 4, got shape {scenes.shape}")

        if scenes.shape[1] != 3:
            raise ValueError(f"scenes must have 3 channels, got shape {scenes.shape}")

        scenes = F.interpolate(
            scenes,
            size=(self.sr_clip_image_size, self.sr_clip_image_size),
            mode="bilinear",
            align_corners=False,
        )

        return scenes

    def _prepare_captions(self, captions: torch.Tensor, device: torch.device) -> torch.Tensor:
        if captions is None:
            raise ValueError("captions must not be None when using SR-CLIP PromptEncoder.")

        if not torch.is_tensor(captions):
            raise TypeError(
                f"captions must be a torch.Tensor of token ids, got {type(captions)}"
            )

        captions = captions.to(device)

        if captions.dim() == 1:
            captions = captions.unsqueeze(0)
        elif captions.dim() == 2:
            pass
        else:
            raise ValueError(
                f"captions must have shape [L] or [K, L], got {captions.shape}"
            )

        return captions

    def _prepare_prompt_image(
        self,
        prompt_image: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if prompt_image is None:
            raise ValueError(
                "prompt_image must not be None when using the mask-prompt dense branch."
            )

        if not torch.is_tensor(prompt_image):
            raise TypeError(f"prompt_image must be a torch.Tensor, got {type(prompt_image)}")

        prompt_image = prompt_image.to(device)
        if prompt_image.dim() == 3:
            prompt_image = prompt_image.unsqueeze(0)
        elif prompt_image.dim() != 4:
            raise ValueError(
                f"prompt_image must have dim 3 or 4, got shape {prompt_image.shape}"
            )

        if prompt_image.shape[1] != 3:
            raise ValueError(
                f"prompt_image must have 3 channels, got shape {prompt_image.shape}"
            )

        return prompt_image

    def forward(
        self,
        scenes: Optional[torch.Tensor],
        prompt_image: Optional[torch.Tensor],
        captions: Optional[torch.Tensor],
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Dict[str, torch.Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        device = self._get_device()
        self.sr_clip.to(device)

        prompt_image = self._prepare_prompt_image(prompt_image, device)
        (
            mask_prompt_256_dense_embeddings,
            mask_prompt_512_dense_embeddings,
            mask_prompt_1024_dense_embeddings,
        ) = self._build_mask_prompt_dense_embeddings(prompt_image)

        scene_images = self._prepare_scene_image(scenes).to(device)
        caption_tokens = self._prepare_captions(captions, device)

        srclip_outputs = self.sr_clip(scene_images, caption_tokens)

        text_token = srclip_outputs["text_token"]
        image_token = srclip_outputs["image_token"]

        prompt_aux: Dict[str, torch.Tensor] = {}

        bs = self._get_batch_size(points, boxes, masks, scenes)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs,
            -1,
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        )

        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim),
            device=device,
        )

        if points is not None:
            coords, labels = points
            coords = coords.to(device)
            labels = labels.to(device)
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            boxes = boxes.to(device)
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            masks = masks.to(device)
            dense_embeddings = self._embed_mask_inputs(masks)

        if text_token.shape[0] != bs:
            if text_token.shape[0] == 1:
                text_token = text_token.expand(bs, -1)
            else:
                raise ValueError(
                    f"text_token batch mismatch: token batch={text_token.shape[0]}, prompt batch={bs}"
                )

        if image_token.shape[0] != bs:
            if image_token.shape[0] == 1:
                image_token = image_token.expand(bs, -1)
            else:
                raise ValueError(
                    f"image_token batch mismatch: token batch={image_token.shape[0]}, prompt batch={bs}"
                )

        return (
            sparse_embeddings,
            dense_embeddings,
            text_token,
            image_token,
            prompt_aux,
            mask_prompt_256_dense_embeddings,
            mask_prompt_512_dense_embeddings,
            mask_prompt_1024_dense_embeddings,
        )


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0

        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device

        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5

        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(
        self,
        coords_input: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))
