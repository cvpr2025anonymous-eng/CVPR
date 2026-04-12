from collections import OrderedDict
from typing import Tuple, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
try:
    from torchvision.transforms import Compose, Resize
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from torchvision.transforms import Compose, Resize
    BICUBIC = Image.BICUBIC

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = None
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.AvgPool2d(stride)), ('0', nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):

    def __init__(self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spatial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(query=x[:1], key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x.squeeze(0)

class ModifiedResNet(nn.Module):

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(inp):
            inp = self.relu1(self.bn1(self.conv1(inp)))
            inp = self.relu2(self.bn2(self.conv2(inp)))
            inp = self.relu3(self.bn3(self.conv3(inp)))
            inp = self.avgpool(inp)
            return inp
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x

class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        bq, nq, cq = q.shape
        bk, nk, ck = k.shape
        assert bq == bk and cq == ck and (k.shape == v.shape)
        q = self.q_proj(q).reshape(bq, nq, self.num_heads, cq // self.num_heads)
        k = self.k_proj(k).reshape(bk, nk, self.num_heads, ck // self.num_heads)
        v = self.v_proj(v).reshape(bk, nk, self.num_heads, ck // self.num_heads)
        attn = torch.einsum('bnhc,bmhc->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum('bhnm,bmhc->bnhc', attn, v).reshape(bq, nq, cq)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class TextToText(nn.Module):

    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = Attention(dim, nhead, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, text):
        x = text + self.self_attn(self.norm1(text), self.norm1(text), self.norm1(text))
        x = x + self.mlp(self.norm2(x))
        return x

class VisualToText(nn.Module):

    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = Attention(dim, nhead, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, text, visual):
        x = text + self.cross_attn(self.norm1(text), visual, visual)
        x = x + self.mlp(self.norm2(x))
        return x

class VisualToVisual(nn.Module):

    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = Attention(dim, nhead, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, visual):
        x = visual + self.self_attn(self.norm1(visual), self.norm1(visual), self.norm1(visual))
        x = x + self.mlp(self.norm2(x))
        return x

class TextToVisual(nn.Module):

    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = Attention(dim, nhead, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, visual, text):
        x = visual + self.cross_attn(self.norm1(visual), text, text)
        x = x + self.mlp(self.norm2(x))
        return x

class TextDecoder(nn.Module):

    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.text_to_text = TextToText(dim, nhead, dropout)
        self.visual_to_text = VisualToText(dim, nhead, dropout)

    def forward(self, visual, text):
        text = self.text_to_text(text)
        text = self.visual_to_text(text, visual)
        return text

class VisualDecoder(nn.Module):

    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.visual_to_visual = VisualToVisual(dim, nhead, dropout)
        self.text_to_visual = TextToVisual(dim, nhead, dropout)

    def forward(self, visual, text):
        visual = self.visual_to_visual(visual)
        visual = self.text_to_visual(visual, text)
        return visual

class TokenExpanderMLP(nn.Module):

    def __init__(self, dim: int, num_tokens: int, hidden_dim: Optional[int]=None):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        hidden_dim = hidden_dim or dim * 2
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_tokens * dim))
        self.slot_embedding = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.delta_scale = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.slot_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            if x.shape[1] != 1:
                raise ValueError(f'TokenExpanderMLP expects a single-token input, but got shape {tuple(x.shape)}')
            x = x[:, 0]
        elif x.dim() != 2:
            raise ValueError(f'TokenExpanderMLP expects [B, D] or [B, 1, D], but got shape {tuple(x.shape)}')
        b, d = x.shape
        if d != self.dim:
            raise ValueError(f'TokenExpanderMLP dim mismatch: expected {self.dim}, got {d}')
        base = x.unsqueeze(1).expand(-1, self.num_tokens, -1)
        delta = self.mlp(self.norm(x)).view(b, self.num_tokens, self.dim)
        return base + self.slot_embedding + self.delta_scale * delta

class VisionTransformer(nn.Module):

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** (-0.5)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width=width, layers=layers, heads=heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.proj_visual = nn.Parameter(scale * torch.randn(output_dim, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(b, 1, x.shape[-1], device=x.device), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        x = x @ self.proj
        x_scene = x[:, 0]
        x_visual = x[:, 1:] @ self.proj_visual
        x_visual = x_visual.reshape(b, h * w, -1)
        return (x_scene, x_visual)

class CLIPBackbone(nn.Module):

    def __init__(self, embed_dim: int, image_resolution: int, vision_layers, vision_width: int, vision_patch_size: int, context_length: int, vocab_size: int, transformer_width: int, transformer_heads: int, transformer_layers: int):
        super().__init__()
        self.context_length = context_length
        self.prompt_dim = embed_dim * 2
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers, output_dim=embed_dim, heads=vision_heads, input_resolution=image_resolution, width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)
        self.projection = nn.Parameter(torch.empty(transformer_width, self.prompt_dim))
        learned_ctx_len = max(1, min(16, self.context_length - 1))
        self.contexts = nn.Parameter(torch.randn(1, learned_ctx_len, transformer_width))
        self.context_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, transformer_width))
        self.visual_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, self.prompt_dim))
        self.scene_prompt_proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, self.prompt_dim))
        self.num_scene_prompt_tokens = 4
        self.num_text_prompt_tokens = 4
        self.scene_token_expander = TokenExpanderMLP(dim=self.prompt_dim, num_tokens=self.num_scene_prompt_tokens)
        self.text_token_expander = TokenExpanderMLP(dim=self.prompt_dim, num_tokens=self.num_text_prompt_tokens)
        self.object_object_refine = VisualToVisual(dim=self.prompt_dim, nhead=transformer_heads, dropout=0.1)
        self.scene_object_refine = VisualToText(dim=self.prompt_dim, nhead=transformer_heads, dropout=0.1)
        self.text_decoder = TextDecoder(dim=self.prompt_dim, nhead=transformer_heads, dropout=0.1)
        self.visual_decoder = VisualDecoder(dim=self.prompt_dim, nhead=transformer_heads, dropout=0.1)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.projection, std=0.02)
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** (-0.5)
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith('bn3.weight'):
                        nn.init.zeros_(param)
        proj_std = self.transformer.width ** (-0.5) * (2 * self.transformer.layers) ** (-0.5)
        attn_std = self.transformer.width ** (-0.5)
        fc_std = (2 * self.transformer.width) ** (-0.5)
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        if hasattr(self.visual, 'conv1'):
            return self.visual.conv1.weight.dtype
        return torch.float32

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text, context):
        x_text = self.token_embedding(text)
        k, n1, c = x_text.shape
        b, n2, c_ctx = context.shape
        if c_ctx != c:
            raise ValueError(f'context last dim ({c_ctx}) must match token embedding dim ({c})')
        if n1 != self.context_length:
            raise ValueError(f'text sequence length ({n1}) must equal CLIP context_length ({self.context_length})')
        x = x_text.reshape(1, k, n1, c).expand(b, k, n1, c).clone()
        usable_ctx = min(n2, n1 - 1)
        if usable_ctx > 0:
            context = context[:, None, :usable_ctx, :].expand(b, k, usable_ctx, c)
            x[:, :, 1:1 + usable_ctx, :] = context
        eos_idx = text.argmax(dim=-1)
        eos_idx = eos_idx.reshape(1, k).expand(b, k).reshape(-1)
        x = x.reshape(b * k, n1, c)
        x = x + self.positional_embedding[:n1]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), eos_idx] @ self.projection
        return x

    def forward_features(self, image, text):
        scene_features, object_features = self.encode_image(image)
        if scene_features.dim() == 2:
            scene_features = scene_features.unsqueeze(1)
        b = object_features.shape[0]
        eps = 1e-08
        object_tokens = self.visual_proj(object_features)
        object_tokens = self.object_object_refine(object_tokens)
        scene_prompt = self.scene_prompt_proj(scene_features)
        scene_slots = self.scene_token_expander(scene_prompt)
        scene_slots = self.scene_object_refine(scene_slots, object_tokens)
        visual = torch.cat([scene_slots, object_tokens], dim=1)
        scene_context = self.context_proj(scene_features)
        context = torch.cat([scene_context, self.contexts.expand(b, -1, -1)], dim=1)
        text_features = self.encode_text(text, context).view(b, -1, self.prompt_dim)
        if text_features.shape[1] == 1:
            text_features = self.text_token_expander(text_features)
        tprompt = self.text_decoder(visual, text_features)
        vprompt = self.visual_decoder(visual, text_features)
        tprompt = F.normalize(tprompt, dim=-1, eps=eps)
        vprompt = F.normalize(vprompt, dim=-1, eps=eps)
        return (tprompt, vprompt)

    def forward(self, image, text):
        return self.forward_features(image, text)

class SRCLIP(nn.Module):

    def __init__(self, clip_backbone: CLIPBackbone, prompt_dim: int=1024, decoder_dim: int=256):
        super().__init__()
        self.clip_backbone = clip_backbone
        self.prompt_dim = prompt_dim
        self.decoder_dim = decoder_dim
        self.text_adapter = nn.Sequential(nn.LayerNorm(prompt_dim), nn.Linear(prompt_dim, prompt_dim), nn.GELU(), nn.Linear(prompt_dim, decoder_dim))
        self.visual_adapter = nn.Sequential(nn.LayerNorm(prompt_dim), nn.Linear(prompt_dim, prompt_dim), nn.GELU(), nn.Linear(prompt_dim, decoder_dim))

    def forward(self, image, text):
        tprompt, vprompt = self.clip_backbone(image, text)
        text_prompt = tprompt.mean(dim=1)
        visual_prompt = vprompt.mean(dim=1)
        text_token = self.text_adapter(text_prompt)
        image_token = self.visual_adapter(visual_prompt)
        return {'text_token': text_token, 'image_token': image_token, 'TPrompt': tprompt, 'VPrompt': vprompt}

def convert_weights(model: nn.Module):

    def _convert_weights_to_fp16(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            layer.weight.data = layer.weight.data.half()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.half()
        if isinstance(layer, nn.MultiheadAttention):
            for attr in [*[f'{s}_proj_weight' for s in ['in', 'q', 'k', 'v']], 'in_proj_bias', 'bias_k', 'bias_v']:
                tensor = getattr(layer, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ['text_projection', 'proj']:
            if hasattr(layer, name):
                attr = getattr(layer, name)
                if attr is not None:
                    attr.data = attr.data.half()
    model.apply(_convert_weights_to_fp16)

def _convert_image_to_rgb(image):
    return image.convert('RGB')

def _transform(n_px):
    return Compose([Resize(n_px, interpolation=BICUBIC)])

def build_clip_backbone(state_dict: dict):
    vit = 'visual.proj' in state_dict
    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith('visual.') and k.endswith('.attn.in_proj_weight')])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round((state_dict['visual.positional_embedding'].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set((k.split('.')[2] for k in state_dict if k.startswith(f'visual.layer{b}')))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict['visual.layer1.0.conv1.weight'].shape[0]
        output_width = round((state_dict['visual.attnpool.positional_embedding'].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict['visual.attnpool.positional_embedding'].shape[0]
        image_resolution = output_width * 32
    if 'text_projection' in state_dict:
        embed_dim = state_dict['text_projection'].shape[1]
    else:
        proj_out = state_dict['projection'].shape[1]
        embed_dim = proj_out // 2
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = max(1, transformer_width // 64)
    transformer_layers = len(set((k.split('.')[2] for k in state_dict if k.startswith('transformer.resblocks'))))
    model = CLIPBackbone(embed_dim=embed_dim, image_resolution=image_resolution, vision_layers=vision_layers, vision_width=vision_width, vision_patch_size=vision_patch_size, context_length=context_length, vocab_size=vocab_size, transformer_width=transformer_width, transformer_heads=transformer_heads, transformer_layers=transformer_layers)
    for key in ['input_resolution', 'context_length', 'vocab_size']:
        if key in state_dict:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    return model.eval()
