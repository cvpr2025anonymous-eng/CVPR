# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        # 冻结所有参数
        for n, p in self.named_parameters():
            p.requires_grad = False

        # 选择性解冻某些层
        for name, param in self.named_parameters():
            if "mask_decoder" in name:  # 仅解冻 mask_decoder 的参数
                param.requires_grad = True

        for name, param in self.named_parameters():
            if "prompt_encoder" in name:  # 仅解冻 mask_decoder 的参数
                param.requires_grad = True

        # for name, param in self.named_parameters():
        #     if "prompt_encoder" in name:  # 仅解冻 mask_decoder 的参数
        #         param.requires_grad = True

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    # @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> [Tensor, Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """

        # batched_input是一个包含多个输入字典的列表，每个字典中有一个键
        # "image"对应着图像数据。
        # 使用列表推导式[self.preprocess(x["image"]) for x in batched_input]
        # 遍历 batched_input，对每个图像 x["image"] 调用 self.preprocess 方法进行预处理。
        # torch.stack(..., dim=0)将预处理后的图像列表沿着第一个维度（即批次维度）堆叠成一个单一的张量input_images。
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings, inter_embeddings = self.image_encoder(input_images)
        inter_embeddings = torch.stack([inter_embedding for inter_embedding in inter_embeddings], dim=0)
        inter_embeddings = inter_embeddings[0].permute(0, 3, 1, 2)   # early-layer ViT feature, after 1st global attention block in ViT
        # print(inter_embeddings.shape)
        # 遍历batched_input和image_embeddings两个集合
        # 使用zip(batched_input, image_embeddings)将batched_input和image_embeddings中的元素配对，分别赋值给image_record和curr_embedding
        # zip是Python中的一个内置函数，用于将多个可迭代对象（如列表、元组等）“配对”在一起。
        # 它的作用是将传入的可迭代对象按元素位置进行组合，返回一个迭代器，其中每个元素是一个元组，包含来自各个可迭代对象的对应元素。
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

            # masks = masks > self.mask_threshold

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
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
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
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
