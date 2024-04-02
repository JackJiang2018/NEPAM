#%%
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
        pos_kernel=(7, 7),
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

if __name__=="__main__":
    model=NEPAM((16,16),(2,2),(224,224),10)
    inp=torch.randn(2,768,14,14)
    out=model(inp)
