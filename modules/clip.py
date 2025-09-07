"""
Implementation of a Hybrid CLIP model combining STOP and TempMe.
- Base architecture and patching mechanism from TempMe.
- Pixel-space Spatio-Temporal Prompting from STOP.
"""
from collections import OrderedDict
from typing import Tuple, Union, List
import hashlib
import os
import urllib
import warnings
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
import math
import logging

# --- Imports for our hybrid model ---
from .temporal_prompting import get_TemporalPrompt # From STOP
from .module_transformer import Transformer as TextTransformer # From TempMe, aliased for clarity
from .utils import log_info

logger = logging.getLogger(__name__)

# ===================================================================================
# Foundational Building Blocks (largely from TempMe's module_clip.py)
# ===================================================================================

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Attention(nn.Module):
    """ Custom Attention module from TempMe, designed to be patched by ToMeAttention. """
    def __init__(self, embed_dim, num_heads, lora_dim=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = float(self.head_dim) ** -0.5
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        # LoRA parameters (can be ignored if lora_dim is 0)
        self.lora_dim = lora_dim
        if lora_dim > 0:
            self.TVPt_LoRA_a = nn.Parameter(torch.zeros(lora_dim, embed_dim))
            nn.init.kaiming_uniform_(self.TVPt_LoRA_a, a=math.sqrt(5))
            self.TVPt_LoRA_b = nn.Parameter(torch.zeros(3 * embed_dim, lora_dim))

    def forward(self, x, attn_mask=None):
        bsz, tgt_len, embed_dim = x.size()
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        
        if self.lora_dim > 0:
            qkv_delta = F.linear(x, self.TVPt_LoRA_a)
            qkv_delta = F.linear(qkv_delta, self.TVPt_LoRA_b).chunk(3, dim=-1)
            q, k, v = q + qkv_delta[0], k + qkv_delta[1], v + qkv_delta[2]

        q = q * self.scaling
        q = q.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1))
        
        if attn_mask is not None:
             attn = attn + attn_mask
             
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        x = self.out_proj(x)
        return x, None

class ResidualAttentionBlock(nn.Module):
    """ The basic building block for the transformer, from TempMe. """
    def __init__(self, d_model: int, n_head: int, lora_dim: int, frame_num: int, attn_mask=None):
        super(ResidualAttentionBlock, self).__init__()
        self.attn = Attention(d_model, n_head, lora_dim)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        if frame_num > 0:
            self.TVPt_Video_Positional_embedding = nn.Parameter(torch.zeros(1, frame_num, 1, d_model))

    def forward(self, x, attn_mask=None):
        # This will be overridden by ToMeBlock for the visual transformer,
        # but is used as-is by the text transformer.
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)[0]
        x = x + self.mlp(self.ln_2(x))
        return x

class VisualTransformer(nn.Module):
    """ HYBRID Visual Transformer: Combines STOP's prompting and TempMe's merging. """
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, lora_dim: int, merge_layer: List[int], frame_pos: int, frame_num_list: List[int], args=None):
        super(VisualTransformer, self).__init__()
        self.args = args
        self.clip = None # This will be set by the parent CLIP model after init.

        # --- STOP Integration ---
        if self.args and hasattr(self.args, 'temporal_prompt') and self.args.temporal_prompt:
            print("INFO: Initializing STOP's TemporalPrompt module.")
            self.TemporalPrompt = get_TemporalPrompt(args=self.args)

        # --- TempMe Architecture ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        # Use nn.ModuleList to allow for the custom forward loop.
        # This Transformer is composed of ResidualAttentionBlocks, which will be patched into ToMeBlocks.
        self.transformer_blocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, lora_dim, frame_num_list[min(idx_pos, len(frame_num_list)-1)] if _l in merge_layer else 0, attn_mask=None)
            for _l, idx_pos in zip(range(layers), range(layers))
        ])

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, video_frame=-1):
        # PART 1: STOP's LOGIC (APPLIED FIRST)
        if hasattr(self, 'TemporalPrompt'):
            b, c, h, w = x.shape
            x = x.view(b // video_frame, video_frame, c, h, w)
            x = self.TemporalPrompt(x)
            x = x.view(b, c, h, w)

        # PART 2: STANDARD ViT PATCHING
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # PART 3: TEMPME's PROGRESSIVE MERGING LOOP
        self.clip._tome_info["size"] = None
        self.clip._tome_info["source"] = None
        self.clip._tome_info["cls_num"] = 1
        self.clip._tome_info["frame_num"] = video_frame
        self.clip._tome_info["token_num"] = x.shape[1]

        merge_layer_list = [int(l) for l in self.args.merge_layer.split('-')]
        merge_frame_num_list = [int(l) for l in self.args.merge_frame_num.split('-')]
        merge_token_prop_list = [float(p) / 100 for p in self.args.merge_token_proportion.split('-')]

        for i, block in enumerate(self.transformer_blocks):
            if i in merge_layer_list:
                merge_idx = merge_layer_list.index(i)
                M_frame_num = merge_frame_num_list[merge_idx]
                M_token_num_0 = int(self.clip._tome_info["token_num"] * M_frame_num * merge_token_prop_list[0])
                M_token_num_1 = int((self.clip._tome_info["token_num"] * M_frame_num - M_token_num_0) * merge_token_prop_list[1])
                x = block(x, M_frame_num=M_frame_num, M_token_num=[M_token_num_0, M_token_num_1], frame_pos=self.args.frame_pos)
            else:
                x = block(x, M_frame_num=1, M_token_num=[self.args.tome_r])
        
        # PART 4: FINAL FEATURE AGGREGATION
        b = x.shape[0] // self.clip._tome_info["frame_num"]
        n_f = self.clip._tome_info["frame_num"]
        token_len = x.shape[1]
        d_v = x.shape[2]
        cls_tokens = x.view(b, n_f, token_len, d_v)[:, :, :self.clip._tome_info["cls_num"], :]
        video_feat = cls_tokens.reshape(b, -1, d_v).mean(dim=1)
        video_feat = self.ln_post(video_feat) @ self.proj
        
        return video_feat

class CLIP(nn.Module):
    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: int, vision_width: int,
                 vision_patch_size: int, context_length: int, vocab_size: int, transformer_width: int,
                 transformer_heads: int, transformer_layers: int, lora_dim: int, merge_layer: List[int],
                 frame_pos: int, frame_num_list: List[int], args=None):
        super(CLIP, self).__init__()
        self.args = args
        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width,
            layers=vision_layers, heads=vision_heads, output_dim=embed_dim, lora_dim=lora_dim,
            merge_layer=merge_layer, frame_pos=frame_pos, frame_num_list=frame_num_list, args=args
        )
        self.visual.clip = self

        self.transformer = TextTransformer(
            width=transformer_width, layers=transformer_layers, heads=transformer_heads, lora_dim=lora_dim
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))
        
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, video_frame=-1):
        return self.visual(image.type(self.dtype), video_frame=video_frame)

    def encode_text(self, text, *args, **kwargs): # Accept extra args for compatibility
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = self.transformer(x, attn_mask=self.build_attention_mask())
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text, video_frame=-1):
        image_features = self.encode_image(image, video_frame=video_frame)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text

# ===================================================================================
# Utility and Model Loading Functions
# ===================================================================================

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        if isinstance(l, Attention):
             for attr in ["in_proj_weight", "in_proj_bias", "TVPt_LoRA_a", "TVPt_LoRA_b"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
    model.apply(_convert_weights_to_fp16)

def build_clip_model(state_dict: dict, video_frames=None, args=None, **kwargs):
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        raise ValueError("Only ViT backbones are supported for this hybrid model.")

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    
    lora_dim = getattr(args, 'lora_dim', 0)
    merge_layer = [int(l) for l in getattr(args, 'merge_layer', '9-10-11').split('-')]
    frame_pos = getattr(args, 'frame_pos', 1)
    merge_frame_num_config = [int(l) for l in getattr(args, 'merge_frame_num', '2-2-2').split('-')]

    frame_num_list = []
    current_frame_num = video_frames
    for i in range(len(merge_layer)):
        frame_num_list.append(current_frame_num)
        if i < len(merge_frame_num_config):
             current_frame_num //= merge_frame_num_config[i]

    model = CLIP(
        embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        lora_dim=lora_dim, merge_layer=merge_layer, frame_pos=frame_pos,
        frame_num_list=frame_num_list, args=args
    ).float()

    model.load_state_dict(state_dict, strict=False)
    return model, {}

def load_clip_state_dict(pretrained_clip_name="ViT-B/32", pretrained_dir=os.path.expanduser("~/models/pretrained")):
    _MODELS = {
        "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    }
    _PT_NAME = { "ViT-B/32": "ViT-B-32.pt", "ViT-B/16": "ViT-B-16.pt" }
    
    if pretrained_clip_name not in _MODELS:
        raise NotImplementedError(f'Model {pretrained_clip_name} not found.')
        
    model_path = os.path.join(pretrained_dir, _PT_NAME[pretrained_clip_name])
    
    if not os.path.exists(model_path):
        _download(_MODELS[pretrained_clip_name], pretrained_dir)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    return state_dict

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)
    expected_sha256 = url.split("/")[-2]

    if os.path.exists(download_target) and hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
        return
        
    print(f"Downloading {url} to {download_target}")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer: break
                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not match")