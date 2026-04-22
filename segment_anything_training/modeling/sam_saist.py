import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple, Optional

from .image_encoder_saist import ImageEncoderViT
from .mask_decoder_saist import MaskDecoder
from .prompt_encoder_saist import PromptEncoder

class Sam(nn.Module):
    mask_threshold: float = 0.5
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False
        )

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def _safe_cat_optional(self, values: List[Optional[Tensor]]) -> Optional[Tensor]:
        if len(values) == 0:
            return None
        if any(v is None for v in values):
            return None
        if not all(torch.is_tensor(v) for v in values):
            return None
        return torch.cat(values, dim=0)

    def _postprocess_aux_outputs(
        self,
        aux_outputs: Optional[Dict[str, Any]],
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> Dict[str, Any]:
        processed_aux_outputs: Dict[str, Any] = {}

        if aux_outputs is not None:
            for key, value in aux_outputs.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.dim() == 4
                    and ("mask" in key or "logit" in key)
                ):
                    processed_aux_outputs[key] = self.postprocess_masks(
                        value,
                        input_size=input_size,
                        original_size=original_size,
                    )
                else:
                    processed_aux_outputs[key] = value

        return processed_aux_outputs

    def _merge_aux_outputs(
        self,
        outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        aux_outputs_merged: Dict[str, Any] = {}

        if len(outputs) == 0:
            return aux_outputs_merged

        all_keys = set()
        for out in outputs:
            all_keys.update(out["aux_outputs"].keys())

        for key in all_keys:
            values = [out["aux_outputs"].get(key, None) for out in outputs]
            values = [v for v in values if v is not None]

            if len(values) == 0:
                continue

            first_value = values[0]

            if not isinstance(first_value, torch.Tensor):
                continue

            if first_value.dim() == 0:
                aux_outputs_merged[key] = torch.stack(values, dim=0)

            else:
                aux_outputs_merged[key] = torch.cat(values, dim=0)

        return aux_outputs_merged

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Dict[str, Any]]:
        input_images = torch.stack(
            [self.preprocess(x["image"]) for x in batched_input], dim=0
        )

        image_embeddings, inter_embeddings = self.image_encoder(input_images)

        multi_inter_embeddings = [
            feat.permute(0, 3, 1, 2).contiguous() for feat in inter_embeddings
        ]

        outputs: List[Dict[str, Any]] = []

        for b_idx, (image_record, curr_embedding) in enumerate(
            zip(batched_input, image_embeddings)
        ):
            curr_inter_embeddings = [
                feat[b_idx:b_idx + 1] for feat in multi_inter_embeddings
            ]

            if "point_coords" in image_record:
                points = (
                    image_record["point_coords"],
                    image_record["point_labels"],
                )
            else:
                points = None

            scenes = self.preprocess(image_record["background"])
            prompt_image = input_images[b_idx:b_idx + 1]
            captions = image_record["caption"]

            if not torch.is_tensor(captions):
                raise TypeError("image_record['caption'] must be token ids tensor.")

            (
                sparse_embeddings,
                dense_embeddings,
                text_tokens,
                image_tokens,
                prompt_aux_outputs,
                mask_prompt_256_dense_embeddings,
                mask_prompt_512_dense_embeddings,
                mask_prompt_1024_dense_embeddings,
            ) = self.prompt_encoder(
                scenes=scenes,
                prompt_image=prompt_image,
                captions=captions,
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )

            low_res_masks, sam_masks, _, aux_outputs = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                interm_embeddings=curr_inter_embeddings,
                mask_prompt_256_dense_embeddings=mask_prompt_256_dense_embeddings,
                mask_prompt_512_dense_embeddings=mask_prompt_512_dense_embeddings,
                mask_prompt_1024_dense_embeddings=mask_prompt_1024_dense_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                image_tokens=image_tokens,
                text_tokens=text_tokens,
                multimask_output=multimask_output,
            )

            aux_outputs["sam_mask"] = sam_masks

            if prompt_aux_outputs is not None:
                aux_outputs.update(prompt_aux_outputs)

            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )

            processed_aux_outputs = self._postprocess_aux_outputs(
                aux_outputs=aux_outputs,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )

            outputs.append(
                {
                    "masks": masks,
                    "text_tokens": text_tokens,
                    "image_tokens": image_tokens,
                    "aux_outputs": processed_aux_outputs,
                }
            )

        masks = torch.cat([x["masks"] for x in outputs], dim=0)

        text_tokens = self._safe_cat_optional(
            [x.get("text_tokens", None) for x in outputs]
        )
        image_tokens = self._safe_cat_optional(
            [x.get("image_tokens", None) for x in outputs]
        )

        aux_outputs_merged = self._merge_aux_outputs(outputs)

        return masks, text_tokens, image_tokens, aux_outputs_merged

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        target_size = (self.image_encoder.img_size, self.image_encoder.img_size)
        if masks.shape[-2:] != target_size:
            masks = F.interpolate(
                masks,
                target_size,
                mode="bilinear",
                align_corners=False,
            )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks,
            original_size,
            mode="bilinear",
            align_corners=False,
        )
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected image tensor [C,H,W], got shape {tuple(x.shape)}")

        if x.shape[0] not in (1, 3):
            raise ValueError(f"Only 1-channel or 3-channel inputs are supported, got {x.shape[0]} channels")

        mean = self.pixel_mean
        std = self.pixel_std.clamp_min(1e-6)

        if mean.shape[0] not in (1, 3) or std.shape[0] not in (1, 3):
            raise ValueError("pixel_mean and pixel_std must have length 1 or 3")

        if x.shape[0] == 1 and mean.shape[0] == 3:
            mean = mean.mean(dim=0, keepdim=True)
            std = std.mean(dim=0, keepdim=True)
        elif x.shape[0] == 3 and mean.shape[0] == 1:
            mean = mean.expand(3, -1, -1)
            std = std.expand(3, -1, -1)

        x = (x - mean) / std

        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)

        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
