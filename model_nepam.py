# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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
        group_merged_num=0,
        score_gate=None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.group_size = group_size
        self.feature_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        token_idx = self.GenTokenIdx(self.feature_size)
        self.register_buffer("token_idx2", token_idx)
        self.group_merged_num = group_merged_num
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
                token_idx = self.SortAndSelect(group_score, self.group_merged_num)

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
        group_merged_num=0,
        score_gate=None,
        merge_method="keep1",
        distance="manhattan",
        token_pos=(0, 0),
    ) -> None:
        super().__init__(patch_size, group_size, img_size, group_merged_num, score_gate)
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
        dist = torch.dist(in1, in2, p=1)
        return dist

    def P2NormDist(self, in1, in2):
        dist = torch.dist(in1, in2, p=2)
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
        x = x[:, :, token_pos[1] :: group_size[0], token_pos[1] :: group_size[1]]
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
                    self.token_idx2[0].unsqueeze(0).expand(B, -1),
                    dim=1,
                    index=token_idx[:, : self.group_merged_num],
                ),
                token_idx[:, self.group_merged_num :],
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
            index=token_idx[:, : self.group_merged_num]
            .unsqueeze(1)
            .expand(-1, self.channel_num, -1),
        )
        x = torch.gather(
            x.flatten(2),
            dim=2,
            index=token_idx[:, self.group_merged_num :]
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
            token_idx = self.SortIndex(group_score, self.group_merged_num)
            x, token_idx = self.merge_module(x, token_idx)

        return x, token_idx


if __name__ == "__main__":
    model = NepamAblation(
        (16, 16), (2, 2), (224, 224), 10, merge_method="avg", distance="cosine"
    )
    model2 = NepamAblation(
        (16, 16), (2, 2), (224, 224), 10, merge_method="keep1", distance="cosine"
    )
    inp = torch.randn(2, 768, 14, 14)
    out = model(inp)
    out2 = model2(inp)
