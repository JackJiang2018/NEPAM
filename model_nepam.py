# %%
from timm.layers import Mlp, PatchEmbed, to_2tuple, Format
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import (
    Block,
    build_model_with_cfg,
    VisionTransformer,
    checkpoint_seq,
)

# from timm.models.deit import VisionTransformer
from timm.models.registry import register_model
from typing import Callable, Optional


def upsample(x, group_size):
    x = (
        x.unsqueeze(-1)
        .unsqueeze(-3)
        .expand(-1, -1, -1, group_size[0], -1, group_size[1])
    )
    x = rearrange(x, "b c h hp w wp->b (c hp wp) (h w)")
    return x


def upsample2d(x, group_size):
    x = (
        x.unsqueeze(-1)
        .unsqueeze(-3)
        .expand(-1, -1, -1, group_size[0], -1, group_size[1])
    )
    x = rearrange(x, "b c h hp w wp->b c (h hp) (w wp)")
    return x


def upsample3d(x, group_size):
    x = (
        x.unsqueeze(2)
        .unsqueeze(4)
        .unsqueeze(6)
        .expand(-1, -1, group_size[0], -1, group_size[1], -1, group_size[2], -1)
    )
    x = rearrange(x, "b t tp h hp w wp c->b (t h w) (c tp hp wp)")
    return x


class NEPAM(nn.Module):
    def __init__(
        self,
        patch_size=(16, 16),
        group_size=(2, 2),
        img_size=(224, 224),
        merge_group_num=0,
        score_gate=None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.group_size = group_size
        self.feature_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        token_idx = self.GenTokenIdx(self.feature_size)
        self.register_buffer("token_idx2", token_idx)
        self.merge_group_num = merge_group_num
        self.channel_num = 3 * patch_size[0] * patch_size[1]
        self.score_gate = score_gate

    def PosIndexSelect(self, pos_idx, index):
        B, l = index.shape
        pos_index = index.unsqueeze(-1).expand(-1, -1, pos_idx.shape[-1])
        pos_index_col = index.unsqueeze(-2).expand(-1, l, -1)
        pos = (
            pos_idx.expand(B, -1, -1)
            .gather(dim=1, index=pos_index)
            .gather(dim=2, index=pos_index_col)
        )
        return pos

    def GenTokenIdx(self, feature_size):
        idx = torch.arange(0, feature_size[0] * feature_size[1])
        idx = rearrange(
            idx,
            "(h hp w wp)->(hp wp) (h w)",
            h=feature_size[0] // self.group_size[0],
            hp=self.group_size[0],
            wp=self.group_size[1],
        )
        return idx

    def SelectRef(self, x, group_size):
        x = x[:, :, :: group_size[0], :: group_size[1]]
        x = upsample(x, group_size)
        return x

    def SelectScore(self, score, score_gate):
        assert score.shape[0] == 1, "The batchsize of input must be 1"
        score = score[0]
        token_merge, token_keep = (
            torch.where(score <= score_gate)[0],
            torch.where(score > score_gate)[0],
        )
        token_idx_merge = torch.index_select(
            self.token_idx2[0], dim=0, index=token_merge
        )
        token_idx_keep = torch.index_select(
            self.token_idx2, dim=1, index=token_keep
        ).flatten(0)
        token_idx = torch.cat([token_idx_merge, token_idx_keep], dim=0).sort(dim=0)[0]
        return token_idx.unsqueeze(0)

    def SortAndSelect(self, score, token_num):
        B = score.shape[0]
        token_idx = torch.argsort(score, stable=True)
        token_idx_merge, token_idx_keep = (
            token_idx[:, :token_num],
            token_idx[:, token_num:],
        )
        token_idx_merge = torch.gather(
            self.token_idx2[0].unsqueeze(0).expand(B, -1),
            dim=1,
            index=token_idx_merge,
        )
        token_idx_keep = torch.gather(
            self.token_idx2.unsqueeze(0).expand(B, -1, -1),
            dim=2,
            index=token_idx_keep.unsqueeze(1).expand(-1, self.token_idx2.shape[0], -1),
        ).flatten(1)
        token_idx = torch.cat([token_idx_merge, token_idx_keep], dim=1).sort(dim=1)[0]
        return token_idx

    def forward(self, x, return_index=True, return_pos=False):
        with torch.no_grad():

            x_ref = self.SelectRef(x, self.group_size)
            x_group = rearrange(
                x,
                "b c (h hp) (w wp)->b (c hp wp) (h w)",
                hp=self.group_size[0],
                wp=self.group_size[1],
            )
            group_score = (x_group - x_ref).abs().mean(dim=1).flatten(1)
            if self.score_gate is not None:
                token_idx = self.SelectScore(group_score, self.score_gate[0])
            else:
                token_idx = self.SortAndSelect(group_score, self.merge_group_num)

            x = torch.gather(
                x.flatten(2),
                dim=2,
                index=token_idx.unsqueeze(1).expand(-1, self.channel_num, -1),
            ).transpose(1, 2)
        if return_index:
            return x, token_idx
        else:
            return x


class NepamAblation(NEPAM):
    def __init__(
        self,
        patch_size=(16, 16),
        group_size=(2, 2),
        img_size=(224, 224),
        merge_group_num=0,
        score_gate=None,
        merge_method="keep1",
        distance="manhattan",
        token_pos=(0, 0),
    ) -> None:
        super().__init__(patch_size, group_size, img_size, merge_group_num, score_gate)
        self.token_pos = token_pos
        self.merge_method = merge_method

        if distance == "cosine":
            self.dist_func = self.CosSim
        elif distance == "manhattan":
            self.dist_func = self.P1NormDist
        elif distance == "euclidean":
            self.dist_func = self.P2NormDist

        if merge_method == "keep1":
            self.merge_module = self.SelectToken
        elif merge_method == "avg":
            self.merge_module = self.MergeToken
        elif merge_method == "conv2d":
            # This method needs to finetune
            pass
        elif merge_method == "dwconv2d":
            # This method needs to finetune
            pass
        else:
            print(f"The method of {merge_method} is not supported")

    def P1NormDist(self, in1, in2):
        # dist = torch.cdist(in1, in2, p=1)
        dist = (in1 - in2).abs().mean(dim=1)
        return dist

    def P2NormDist(self, in1, in2):
        # dist = torch.cdist(in1, in2, p=2)
        dist = ((in1 - in2) ** 2).mean(dim=1)
        return dist

    def CosSim(self, in1, in2):
        sim = F.cosine_similarity(in1, in2, 1)
        return sim

    def SortIndex(self, score, merge_groups):
        B = score.shape[0]
        token_idx = torch.argsort(score, stable=True)
        token_idx_merge, token_idx_keep = (
            token_idx[:, :merge_groups],
            token_idx[:, merge_groups:],
        )
        token_idx_keep = torch.gather(
            self.token_idx2.unsqueeze(0).expand(B, -1, -1),
            dim=2,
            index=token_idx_keep.unsqueeze(1).expand(-1, self.token_idx2.shape[0], -1),
        ).flatten(1)
        token_idx = torch.cat([token_idx_merge, token_idx_keep], dim=1)
        return token_idx

    def SelectRef(self, x, group_size, token_pos):
        x = x[:, :, token_pos[0] :: group_size[0], token_pos[1] :: group_size[1]]
        x = upsample2d(x, group_size)
        return x

    def AlignTokenIdx(
        self,
        token_idx,
    ):
        # Align the token index after merging to the token index before merging
        B = token_idx.shape[0]
        token_idx = torch.cat(
            [
                torch.gather(
                    self.token_idx2[
                        self.token_pos[0] * self.group_size[0] + self.token_pos[1]
                    ]
                    .unsqueeze(0)
                    .expand(B, -1),
                    dim=1,
                    index=token_idx[:, : self.merge_group_num],
                ),
                token_idx[:, self.merge_group_num :],
            ],
            dim=1,
        )
        return token_idx

    def MergeToken(self, x, token_idx):
        B, C, H, W = x.shape
        x_ = F.avg_pool2d(x, self.group_size, self.group_size)
        x_ = torch.gather(
            x_.flatten(2),
            dim=2,
            index=token_idx[:, : self.merge_group_num]
            .unsqueeze(1)
            .expand(-1, self.channel_num, -1),
        )
        x = torch.gather(
            x.flatten(2),
            dim=2,
            index=token_idx[:, self.merge_group_num :]
            .unsqueeze(1)
            .expand(-1, self.channel_num, -1),
        )
        x = torch.cat([x_, x], dim=2).transpose(1, 2)
        token_idx = self.AlignTokenIdx(token_idx)

        return x, token_idx

    def SelectToken(self, x, token_idx):
        token_idx = self.AlignTokenIdx(token_idx).sort(dim=1)[0]
        x = torch.gather(
            x.flatten(2),
            dim=2,
            index=token_idx.unsqueeze(1).expand(-1, self.channel_num, -1),
        ).transpose(1, 2)
        return x, token_idx

    def forward(self, x):
        B = x.shape[0]
        with torch.no_grad():
            x_ref = self.SelectRef(x, self.group_size, self.token_pos)
            group_score = self.dist_func(x_ref, x).unsqueeze(1)
            group_score = F.avg_pool2d(group_score, 2, 2).flatten(1)
            token_idx = self.SortIndex(group_score, self.merge_group_num)
        x, token_idx = self.merge_module(x, token_idx)

        return x, token_idx


class PatchEmbedMerge(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = False,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        score_gate=None,
        merge_method="keep1",
        distance="manhattan",
        merge_group_size=(1, 2),
        merge_group_num=0,
        token_pos=(0, 0),
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple(
                [s // p for s, p in zip(self.img_size, self.patch_size)]
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.patch_pixel_num = in_chans * patch_size**2
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.token_drop = NepamAblation(
            self.patch_size,
            merge_group_size,
            self.img_size,
            merge_group_num,
            score_gate,
            merge_method,
            distance,
            token_pos,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(
            x,
            "b c (h hp) (w wp)->b (c hp wp) h w",
            hp=self.patch_size[0],
            wp=self.patch_size[1],
        )
        x, index = self.token_drop(x)
        x = F.linear(x, weight=self.proj.weight.flatten(1), bias=self.proj.bias)
        x = self.norm(x)
        return x, index


class PruneViT(VisionTransformer):
    def __init__(
        self,
        img_size: int | F.Tuple[int] = 224,
        patch_size: int | F.Tuple[int] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: float | None = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm: bool | None = None,
        drop_rate: float = 0,
        pos_drop_rate: float = 0,
        patch_drop_rate: float = 0,
        proj_drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        weight_init: str = "",
        embed_layer: Callable = ...,
        norm_layer: Callable | None = nn.LayerNorm,
        act_layer: Callable | None = nn.GELU,
        block_fn: Callable = Block,
        mlp_layer: Callable = Mlp,
        score_gate=None,
        merge_method="keep1",
        distance="manhattan",
        merge_group_size=(1, 2),
        merge_group_num=0,
        token_pos=(0, 0),
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            global_pool,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            init_values,
            class_token,
            no_embed_class,
            pre_norm,
            fc_norm,
            drop_rate,
            pos_drop_rate,
            patch_drop_rate,
            proj_drop_rate,
            attn_drop_rate,
            drop_path_rate,
            weight_init,
            embed_layer,
            norm_layer,
            act_layer,
            block_fn,
            mlp_layer,
        )
        self.patch_embed = embed_layer(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            None,
            False,
            merge_method=merge_method,
            merge_group_num=merge_group_num,
            merge_group_size=merge_group_size,
            distance=distance,
            token_pos=token_pos,
            score_gate=score_gate,
        )

    def _pos_embed(self, x, index=None):
        if index is None:
            return super()._pos_embed(x)
        else:
            if self.no_embed_class:
                # deit-3, updated JAX (big vision)
                # position embedding does not overlap with class token, add then concat
                x = x + self.pos_embed.repeat(x.shape[0], 1, 1).gather(
                    dim=1, index=index.unsqueeze(-1).repeat(1, 1, self.embed_dim)
                )
                if self.cls_token is not None:
                    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            else:
                # original timm, JAX, and deit vit impl
                # pos_embed has entry for class token, concat then add
                if self.cls_token is not None:
                    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                pos_index = (
                    F.pad(1 + index, [1, 0, 0, 0], value=0)
                    .unsqueeze(-1)
                    .expand(-1, -1, self.embed_dim)
                )
                x = x + self.pos_embed.expand(x.shape[0], -1, -1).gather(
                    dim=1, index=pos_index
                )
            return self.pos_drop(x)

    def forward_features(self, x):
        x, index = self.patch_embed(x)
        x = self._pos_embed(x, index)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x


def deit_small_patch16_224_keep1_man(pretrained=True, **kwargs):
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        merge_method="keep1",
        merge_group_num=39,
        merge_group_size=(1, 2),
        distance="manhattan",
        token_pos=(0, 0),
        score_gate=None,
        embed_layer=PatchEmbedMerge,
    )
    model = build_model_with_cfg(
        PruneViT,
        "deit_small_patch16_224",
        pretrained,
        pretrained_strict=False,
        **model_args,
    )
    return model


if __name__ == "__main__":
    model = deit_small_patch16_224_keep1_man()
    inp = torch.randn(3, 3, 224, 224)
    out = model(inp)
