import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from .common import LayerNorm2d
import torch
from copy import deepcopy


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:

        super().__init__()
        self.transformer_dim = transformer_dim



        self.transformer = deepcopy(transformer)
        self.transformer_clip = deepcopy(transformer)
        self.sigmoid = nn.Sigmoid()

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.output_background_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.scene_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.clip_mlp = MLP(transformer_dim * 4, transformer_dim, transformer_dim, 2)
        self.scene_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
            nn.Conv2d(768, transformer_dim, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=3, stride=1, padding=1))

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        self.scene_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

        self.mask_fusion = nn.Sequential(
            nn.Conv2d(4, 3, 3, 1, 1),
            LayerNorm2d(3),
            nn.GELU(),
            nn.Conv2d(3, 1, 3, 1, 1))

    def forward(
        self,
        image_embeddings: torch.Tensor,
        interm_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        scene_features = self.compress_vit_feat(interm_embeddings)

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            scene_features=scene_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            image_tokens=image_tokens,
            text_tokens=text_tokens,
        )


        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = self.mask_fusion(masks)
        iou_pred = iou_pred[:, mask_slice]


        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        scene_features: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:





        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.scene_token.weight], dim=0)

        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)

        out_tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)



        src = torch.repeat_interleave(image_embeddings, out_tokens.shape[0], dim=0)

        src = src + dense_prompt_embeddings

        pos_src = torch.repeat_interleave(image_pe, out_tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, out_tokens)


        clip_tokens = torch.cat([self.iou_token.weight, text_tokens.reshape(-1, self.transformer_dim)], dim=0)

        clip_tokens = clip_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)

        scene = torch.repeat_interleave(scene_features, clip_tokens.shape[0], dim=0)


        scene = scene + dense_prompt_embeddings

        pos_src = torch.repeat_interleave(image_pe, out_tokens.shape[0], dim=0)
        b, c, h, w = scene.shape


        hs_background, background = self.transformer_clip(scene, pos_src, clip_tokens)



        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]
        clip_tokens_out = hs[:, 1: self.num_mask_tokens, :]


        background = background.transpose(1, 2).view(b, c, h, w)
        background = self.output_upscaling(background)

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        upscaled_embedding_scene = self.scene_upscaling(scene_features)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.scene_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in[:, 0:4] @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        scene = (hyper_in[:, 4:] @ upscaled_embedding_scene.view(b, c, h * w)).view(b, -1, h, w)
        scene = self.sigmoid(scene)

        hyper_background_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_background_list.append(self.output_background_mlps[i](clip_tokens_out[:, i, :]))
        hyper_background = torch.stack(hyper_background_list, dim=1)

        b, c, h, w = background.shape
        background = (hyper_background @ background.view(b, c, h * w)).view(b, -1, h, w)
        background = self.sigmoid(background)

        masks = (masks * scene - masks * background) + masks


        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred




class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
