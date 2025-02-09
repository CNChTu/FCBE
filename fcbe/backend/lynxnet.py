import torch
from torch import nn
import torch.nn.functional as F


# From https://github.com/CNChTu/Diffusion-SVC/
# License: MIT

# tools

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return pad, pad - (kernel_size + 1) % 2


class SwiGLU(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * F.silu(gate)


class ConformerConvModule(nn.Module):
    def __init__(
            self,
            dim,
            expansion=2,
            kernel_size=31,
            dropout=0.,  # dropout rate
            norm=False,  # pre layer normalization
            conv_act='SiLU',  # SiLU, ReLU, PReLU
            gate='GLU',  # GLU, SwiGLU
            gate_upsample=1,  # expansion factor of gate
            res_inner=False,  # false: only layer residual, true: layer and inner residual
            bias=True,  # bias in 1x1 conv
    ):
        super().__init__()

        # pre-normalization
        if norm:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = None

        # dropout
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # get axis, padding
        inner_dim = dim * expansion
        padding = calc_same_padding(kernel_size)

        # activation
        if conv_act == 'SiLU':
            self.conv_act = nn.SiLU()
        elif conv_act == 'ReLU':
            self.conv_act = nn.ReLU()
        elif conv_act == 'PReLU':
            self.conv_act = nn.PReLU(inner_dim)
        else:
            raise ValueError(f'{conv_act} is not a valid activation')

        if gate == 'GLU':
            self.gate = nn.GLU(dim=1)
        elif gate == 'SwiGLU':
            self.gate = SwiGLU(dim=1)
        else:
            raise ValueError(f'{gate} is not a valid GLU type')

        # network
        self.gate_in = nn.Conv1d(
            dim,
            inner_dim * 2 * gate_upsample,
            1,
            bias=bias
        )

        if gate_upsample > 1:
            self.gate_down = nn.Conv1d(
                inner_dim * gate_upsample,
                inner_dim,
                1,
                bias=bias
            )
        else:
            self.gate_down = None

        if res_inner:
            self.res_proj_in = nn.Linear(dim, inner_dim, bias=False)
            self.res_proj_out = nn.Linear(inner_dim, dim, bias=False)
            if norm:
                self.res_norm = nn.LayerNorm(inner_dim)
            else:
                self.res_norm = None
        else:
            self.res_norm = None
            self.res_proj_in = None
            self.res_proj_out = None
        self.dw_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size,
            padding=padding[0],
            groups=inner_dim
        )
        self.proj_out = nn.Linear(inner_dim, dim, bias=bias)


    def forward(self, x):
        # get residual
        raw_x = x

        # pre-normalization
        if self.norm is not None:
            x = self.norm(x)

        x = x.transpose(1, 2)

        # gate
        x = self.gate_in(x)
        x = self.gate(x)
        if self.gate_down is not None:
            x = self.gate_down(x)

        if self.dropout is not None:
            x = self.dropout(x)
        # residual(inner)
        if self.res_proj_in is not None:
            x_res_inner = self.res_proj_in(raw_x)
            x = x + x_res_inner
            if self.res_norm is not None:
                x_res_inner = self.res_norm(x_res_inner)
        else:
            x_res_inner = None

        x = self.dw_conv(x)
        x = self.proj_out(x)

        x = x.transpose(1, 2)

        if self.dropout is not None:
            x = self.dropout(x)

        # residual(layer)
        if x_res_inner is not None:
            x = x + self.res_proj_out(x_res_inner)
        else:
            x = x + raw_x

        return x
