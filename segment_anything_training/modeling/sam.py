





import torch
from torch import nn, Tensor
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


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

        for n, p in self.named_parameters():
            p.requires_grad = False


        for name, param in self.named_parameters():
            if "mask_decoder" in name:
                param.requires_grad = True

        for name, param in self.named_parameters():
            if "prompt_encoder" in name:
                param.requires_grad = True





        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device


    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> [Tensor, Tensor]:







        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings, inter_embeddings = self.image_encoder(input_images)
        inter_embeddings = torch.stack([inter_embedding for inter_embedding in inter_embeddings], dim=0)
        inter_embeddings = inter_embeddings[0].permute(0, 3, 1, 2)





        outputs = []
        for image_record, curr_embedding, inter_embedding in zip(batched_input, image_embeddings, inter_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            scenes = self.preprocess(image_record['background'])
            captions = image_record['caption']
            sparse_embeddings, dense_embeddings, text_tokens, image_tokens = self.prompt_encoder(
                scenes=scenes,
                captions=captions,
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                interm_embeddings=inter_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                image_tokens=image_tokens,
                text_tokens=text_tokens,
                multimask_output=multimask_output
            )

            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )



            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "encoder_embedding": curr_embedding.unsqueeze(0),
                    "image_pe": self.prompt_encoder.get_dense_pe(),
                    "sparse_embeddings": sparse_embeddings,
                    "dense_embeddings": dense_embeddings,
                    "text_tokens": text_tokens,
                    "image_tokens": image_tokens
                }
            )

        masks = torch.cat([x["masks"] for x in outputs], dim=0)
        image_tokens = torch.cat([x["image_tokens"] for x in outputs], dim=0)
        text_tokens = torch.cat([x["text_tokens"] for x in outputs], dim=0)
        return masks, text_tokens, image_tokens


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:


        x = (x - self.pixel_mean) / self.pixel_std


        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
