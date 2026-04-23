import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type, Union
from copy import deepcopy
from .common import LayerNorm2d

class MultiLevelSceneAdapter(nn.Module):
    def __init__(self, vit_dim=768, transformer_dim=256, inter_num_levels=4):
        super().__init__()
        if isinstance(vit_dim, (list, tuple)):
            self.level_input_dims = [int(dim) for dim in vit_dim]
            if inter_num_levels is None:
                self.inter_num_levels = len(self.level_input_dims)
            else:
                self.inter_num_levels = min(inter_num_levels, len(self.level_input_dims))
            self.level_input_dims = self.level_input_dims[: self.inter_num_levels]
        else:
            self.inter_num_levels = inter_num_levels
            self.level_input_dims = [int(vit_dim) for _ in range(self.inter_num_levels)]

        gate_hidden_dim = max(32, transformer_dim // 4)

        self.level_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_dim, transformer_dim, kernel_size=1, bias=False),
                    LayerNorm2d(transformer_dim),
                    nn.GELU(),
                    nn.Conv2d(
                        transformer_dim,
                        transformer_dim,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    LayerNorm2d(transformer_dim),
                    nn.GELU(),
                )
                for in_dim in self.level_input_dims
            ]
        )
        self.level_score = nn.Sequential(
            nn.Linear(transformer_dim * 2, gate_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1, bias=True),
        )
        self.level_logits = nn.Parameter(torch.zeros(self.inter_num_levels))
        self.fuse_refine = nn.Sequential(
            nn.Conv2d(
                transformer_dim,
                transformer_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.Conv2d(
                transformer_dim,
                transformer_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
        )

    def forward(
        self,
        inter_feats: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
        target_size: Tuple[int, int],
    ):
        if not isinstance(inter_feats, (list, tuple)):
            inter_feats = [inter_feats]

        num_used = min(len(inter_feats), self.inter_num_levels)
        if num_used == 0:
            raise ValueError("inter_feats list is empty.")

        fused_features = []
        for i in range(num_used):
            x = self.level_adapters[i](inter_feats[i])
            if x.shape[-2:] != target_size:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            fused_features.append(x)

        level_scores = []
        for i, x in enumerate(fused_features):
            avg_pooled = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
            max_pooled = F.adaptive_max_pool2d(x, output_size=1).flatten(1)
            pooled = torch.cat([avg_pooled, max_pooled], dim=1)
            score = self.level_score(pooled).squeeze(1) + self.level_logits[i]
            level_scores.append(score)

        dynamic_weights = torch.softmax(torch.stack(level_scores, dim=1), dim=1)

        fused = torch.zeros_like(fused_features[0])
        for i, x in enumerate(fused_features):
            weight = dynamic_weights[:, i].view(-1, 1, 1, 1)
            fused = fused + x * weight

        fused = self.fuse_refine(fused)
        return fused, dynamic_weights


class UpscaleTo256Block(nn.Module):
    def __init__(
        self,
        in_chans: int,
        mid_chans: int,
        out_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_chans, mid_chans, kernel_size=2, stride=2),
            LayerNorm2d(mid_chans),
            activation(),
            nn.Conv2d(mid_chans, mid_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(mid_chans),
            activation(),
            nn.ConvTranspose2d(mid_chans, out_chans, kernel_size=2, stride=2),
            LayerNorm2d(out_chans),
            activation(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FeatureFuseBlock(nn.Module):
    def __init__(
        self,
        skip_chans: int,
        in_chans: int,
        out_chans: int,
        use_channel_attention: bool = False,
        use_spatial_attention: bool = True,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention
        attn_hidden = max(4, out_chans // 8)
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_chans, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            activation(),
        )
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
            activation(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_chans * 2, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
            activation(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
            activation(),
        )
        if use_channel_attention:
            self.channel_att_mlp = nn.Sequential(
                nn.Conv2d(out_chans, attn_hidden, kernel_size=1, bias=False),
                activation(),
                nn.Conv2d(attn_hidden, out_chans, kernel_size=1, bias=False),
            )
        if use_spatial_attention:
            self.spatial_att = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.out_refine = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
            activation(),
        )

    def forward(self, skip_feat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if skip_feat.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"FeatureFuseBlock shape mismatch: "
                f"skip={tuple(skip_feat.shape)}, x={tuple(x.shape)}"
            )
        skip_feat = self.skip_proj(skip_feat)
        x = self.in_proj(x)
        fused = self.fuse(torch.cat([skip_feat, x], dim=1))

        if self.use_channel_attention:
            ch_att = torch.sigmoid(
                self.channel_att_mlp(F.adaptive_avg_pool2d(fused, output_size=1))
                + self.channel_att_mlp(F.adaptive_max_pool2d(fused, output_size=1))
            )
            fused = fused * ch_att

        if self.use_spatial_attention:
            sp_att_input = torch.cat(
                [
                    fused.mean(dim=1, keepdim=True),
                    fused.amax(dim=1, keepdim=True),
                ],
                dim=1,
            )
            sp_att = torch.sigmoid(self.spatial_att(sp_att_input))
            fused = fused * sp_att

        return self.out_refine(fused + x)


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
        vit_dim: int = 768,
        inter_num_levels: int = 4,
    ) -> None:
        super().__init__()
        del num_multimask_outputs

        self.transformer_dim = transformer_dim
        self.upscaled_dim = transformer_dim // 8
        self.mask_decoder_dim_256 = self.upscaled_dim
        self.mask_decoder_dim_512 = self.upscaled_dim
        self.mask_decoder_dim_1024 = self.upscaled_dim // 2

        self.foreground_transformer = deepcopy(transformer)
        self.background_transformer = deepcopy(transformer)

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_token = nn.Embedding(1, transformer_dim)
        self.background_token = nn.Embedding(1, transformer_dim)

        mid_chans = transformer_dim // 4

        self.output_upscaling = UpscaleTo256Block(
            in_chans=transformer_dim,
            mid_chans=mid_chans,
            out_chans=self.upscaled_dim,
            activation=activation,
        )

        self.background_upscaling = UpscaleTo256Block(
            in_chans=transformer_dim,
            mid_chans=mid_chans,
            out_chans=self.upscaled_dim,
            activation=activation,
        )

        self.mask_feature_refine = nn.Sequential(
            nn.Conv2d(self.upscaled_dim * 2, self.mask_decoder_dim_256, kernel_size=1, bias=False),
            LayerNorm2d(self.mask_decoder_dim_256),
            activation(),
            nn.Conv2d(
                self.mask_decoder_dim_256,
                self.mask_decoder_dim_256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(self.mask_decoder_dim_256),
            activation(),
            nn.Conv2d(
                self.mask_decoder_dim_256,
                self.mask_decoder_dim_256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(self.mask_decoder_dim_256),
            activation(),
        )

        self.mask_feature_up_512 = nn.Sequential(
            nn.ConvTranspose2d(
                self.mask_decoder_dim_256,
                self.mask_decoder_dim_512,
                kernel_size=2,
                stride=2,
            ),
            LayerNorm2d(self.mask_decoder_dim_512),
            activation(),
            nn.Conv2d(
                self.mask_decoder_dim_512,
                self.mask_decoder_dim_512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(self.mask_decoder_dim_512),
            activation(),
        )
        self.mask_feature_up_1024 = nn.Sequential(
            nn.ConvTranspose2d(
                self.mask_decoder_dim_512,
                self.mask_decoder_dim_1024,
                kernel_size=2,
                stride=2,
            ),
            LayerNorm2d(self.mask_decoder_dim_1024),
            activation(),
            nn.Conv2d(
                self.mask_decoder_dim_1024,
                self.mask_decoder_dim_1024,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(self.mask_decoder_dim_1024),
            activation(),
        )

        self.mask_up_256 = FeatureFuseBlock(
            skip_chans=64,
            in_chans=self.mask_decoder_dim_256,
            out_chans=self.mask_decoder_dim_256,
            use_channel_attention=True,
            use_spatial_attention=True,
            activation=activation,
        )

        self.mask_up_512 = FeatureFuseBlock(
            skip_chans=32,
            in_chans=self.mask_decoder_dim_512,
            out_chans=self.mask_decoder_dim_512,
            use_channel_attention=False,
            use_spatial_attention=False,
            activation=activation,
        )
        self.mask_up_1024 = FeatureFuseBlock(
            skip_chans=16,
            in_chans=self.mask_decoder_dim_1024,
            out_chans=self.mask_decoder_dim_1024,
            use_channel_attention=False,
            use_spatial_attention=False,
            activation=activation,
        )

        self.final_mask_hypernet_mlp = MLP(
            transformer_dim,
            transformer_dim,
            self.mask_decoder_dim_1024,
            3,
        )
        self.sam_mask_hypernet_mlp = MLP(
            transformer_dim,
            transformer_dim,
            self.mask_decoder_dim_256,
            3,
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            4,
            iou_head_depth,
        )

        self.scene_adapter = MultiLevelSceneAdapter(
            vit_dim=vit_dim,
            transformer_dim=transformer_dim,
            inter_num_levels=inter_num_levels,
        )

        self.depth_predictor = nn.Sequential(
            nn.Conv2d(
                self.upscaled_dim,
                self.upscaled_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(self.upscaled_dim),
            activation(),
            nn.Conv2d(
                self.upscaled_dim,
                self.upscaled_dim // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(self.upscaled_dim // 2),
            activation(),
            nn.Conv2d(self.upscaled_dim // 2, 1, kernel_size=1),
        )

        self.log_beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        image_embeddings: torch.Tensor,
        interm_embeddings: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
        mask_prompt_256_dense_embeddings: torch.Tensor,
        mask_prompt_512_dense_embeddings: torch.Tensor,
        mask_prompt_1024_dense_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        multimask_output: bool,
    ):
        del multimask_output

        final_mask, sam_mask, iou_pred, aux_outputs = self.predict_masks(
            image_embeddings=image_embeddings,
            interm_embeddings=interm_embeddings,
            mask_prompt_256_dense_embeddings=mask_prompt_256_dense_embeddings,
            mask_prompt_512_dense_embeddings=mask_prompt_512_dense_embeddings,
            mask_prompt_1024_dense_embeddings=mask_prompt_1024_dense_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            image_tokens=image_tokens,
            text_tokens=text_tokens,
        )

        return final_mask, sam_mask, iou_pred[:, :1], aux_outputs

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        interm_embeddings: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
        mask_prompt_256_dense_embeddings: torch.Tensor,
        mask_prompt_512_dense_embeddings: torch.Tensor,
        mask_prompt_1024_dense_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ):
        batch_size = image_embeddings.shape[0]
        target_size = image_embeddings.shape[-2:]

        scene_dense_embeddings, scene_level_weights = self.scene_adapter(
            interm_embeddings,
            target_size=target_size,
        )

        sam_output_tokens = torch.cat(
            [
                self.iou_token.weight,
                self.mask_token.weight,
            ],
            dim=0,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        sam_tokens = torch.cat((sam_output_tokens, sparse_prompt_embeddings), dim=1)

        sam_src = image_embeddings + scene_dense_embeddings + dense_prompt_embeddings
        sam_pos_src = image_pe.expand(batch_size, -1, -1, -1)

        b, c, h, w = sam_src.shape
        sam_hs, sam_src_seq = self.foreground_transformer(
            sam_src,
            sam_pos_src,
            sam_tokens,
        )

        iou_token_out = sam_hs[:, 0, :]
        mask_token_out = sam_hs[:, 1, :]
        sam_src_feat = sam_src_seq.transpose(1, 2).view(b, c, h, w)
        sam_decode_features = self.output_upscaling(sam_src_feat)

        background_prompt_tokens = torch.cat(
            [
                self.background_token.weight.unsqueeze(0).expand(batch_size, -1, -1),
                torch.stack([text_tokens, image_tokens], dim=1),
            ],
            dim=1,
        )

        bg_src = scene_dense_embeddings + dense_prompt_embeddings
        _, bg_src_seq = self.background_transformer(
            bg_src,
            sam_pos_src,
            background_prompt_tokens,
        )

        bg_src_feat = bg_src_seq.transpose(1, 2).view(b, c, h, w)
        background_features = self.background_upscaling(bg_src_feat)

        depth_map = F.softplus(self.depth_predictor(sam_decode_features))
        beta = F.softplus(self.log_beta) + 1e-6
        inverse_transmission = torch.exp(beta * depth_map).clamp(min=1.0, max=10.0)

        recovered_foreground = (
            sam_decode_features * inverse_transmission
            - background_features * (inverse_transmission - 1.0)
        )
        
        mask_feature_256 = self.mask_feature_refine(
            torch.cat([recovered_foreground, sam_decode_features], dim=1)
        )

        mask_feature_256 = self.mask_up_256(
            mask_prompt_256_dense_embeddings,
            mask_feature_256,
        )

        b_fg, c_fg, h_fg, w_fg = mask_feature_256.shape
        sam_hyper = self.sam_mask_hypernet_mlp(mask_token_out).unsqueeze(1)
        sam_mask = (
            sam_hyper @ mask_feature_256.view(b_fg, c_fg, h_fg * w_fg)
        ).view(b_fg, 1, h_fg, w_fg)

        mask_feature_512 = self.mask_feature_up_512(mask_feature_256)
        mask_feature_512 = self.mask_up_512(
            mask_prompt_512_dense_embeddings,
            mask_feature_512,
        )

        mask_feature_1024 = self.mask_feature_up_1024(mask_feature_512)
        mask_feature_1024 = self.mask_up_1024(
            mask_prompt_1024_dense_embeddings,
            mask_feature_1024,
        )

        b_1024, c_1024, h_1024, w_1024 = mask_feature_1024.shape
        final_hyper = self.final_mask_hypernet_mlp(mask_token_out).unsqueeze(1)
        final_mask = (
            final_hyper @ mask_feature_1024.view(b_1024, c_1024, h_1024 * w_1024)
        ).view(b_1024, 1, h_1024, w_1024)

        iou_pred = self.iou_prediction_head(iou_token_out)
        aux_outputs = {
            "sam_mask": sam_mask,
            "mask_feature_256_mean": mask_feature_256.abs().mean(dim=(1, 2, 3)),
            "mask_feature_512_mean": mask_feature_512.abs().mean(dim=(1, 2, 3)),
            "mask_feature_1024_mean": mask_feature_1024.abs().mean(dim=(1, 2, 3)),
            "scene_level_weights": scene_level_weights,
        }
        return final_mask, sam_mask, iou_pred, aux_outputs


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
        hidden_dims = [hidden_dim] * (num_layers - 1)

        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip(
                [input_dim] + hidden_dims,
                hidden_dims + [output_dim],
            )
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        if self.sigmoid_output:
            x = torch.sigmoid(x)

        return x
