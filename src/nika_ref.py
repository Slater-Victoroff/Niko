import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from soap import SOAP
from load_data import load_video_frames
from encoding_utils import FourierEncoding
from configs import REFERENCES


class TuckerFactor(nn.Module):
    def __init__(self, target_dim, rank, is_complex=False, base_mag=1e-2, device='cuda'):
        """
        Have to split into chunks because there's some weird bug in PyTorch
        where if the dim is over like 520 or something everything just breaks.

        Perhaps someone can figure that out later, but the workaround seems easier atm.
        """
        super().__init__()
        self.max_chunk_size = 500
        self.target_dim = target_dim
        self.rank = rank
        self.is_complex = is_complex
        self.device = device

        def make_chunk(chunk_size):
            if self.is_complex:
                return nn.Parameter(torch.randn(chunk_size, rank, device=device) * base_mag), \
                          nn.Parameter(torch.zeros(chunk_size, rank, device=device))  # real, imag
            else:
                return nn.Parameter(torch.randn(chunk_size, rank, device=device) * base_mag)
        num_chunks = int((target_dim - 1) // self.max_chunk_size) + 1
        self.chunked = False
        if num_chunks > 1:
            self.chunked = True

        if self.chunked:
            if self.is_complex:
                self.real_chunks = nn.ParameterList()
                self.imag_chunks = nn.ParameterList()
            else:
                self.chunks = nn.ParameterList()

            for i in range(num_chunks):
                start = i * self.max_chunk_size
                end = min((i + 1) * self.max_chunk_size, target_dim)
                chunk_size = end - start
                if self.is_complex:
                    real_param, imag_param = make_chunk(chunk_size)
                    self.real_chunks.append(real_param)
                    self.imag_chunks.append(imag_param)
                else:
                    param = make_chunk(chunk_size)
                    self.chunks.append(param)
        else:
            if self.is_complex:
                self.U_real = nn.Parameter(torch.randn(target_dim, rank, device=device) * base_mag)
                self.U_imag = nn.Parameter(torch.zeros(target_dim, rank, device=device))
            else:
                self.U = nn.Parameter(torch.randn(target_dim, rank, device=device) * base_mag)

    def forward(self):
        if self.chunked:
            if self.is_complex:
                U_real = torch.cat(list(self.real_chunks), dim=0)
                U_imag = torch.cat(list(self.imag_chunks), dim=0)
                U = torch.complex(U_real, U_imag)
            else:
                U = torch.cat(list(self.chunks), dim=0)
        else:
            if self.is_complex:
                U = torch.complex(self.U_real, self.U_imag)
            else:
                U = self.U
        return U

    def get(self, target):
        U = self.forward()
        target = torch.as_tensor(target, device=U.device, dtype=torch.float32)

        t_norm = 2.0 * target - 1.0  # [-1, 1]
        t_norm = t_norm.view(1, -1, 1, 1)

        grid = torch.zeros((1, t_norm.shape[1], 1, 2), device=U.device, dtype=t_norm.dtype)
        grid[..., 1] = t_norm.squeeze(-1)  # y coord (H)
        # x coord (W=1) stays 0

        def _sample(inp):
            inp_ = inp.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # [1, R, T, 1]
            out = F.grid_sample(inp_, grid, mode="bilinear", align_corners=True, padding_mode="border")
            return out.squeeze(0).squeeze(-1).transpose(0, 1)  # [B, R]

        if torch.is_complex(U):
            return torch.complex(_sample(U.real), _sample(U.imag))
        return _sample(U)


class RealTucker(nn.Module):
    def __init__(self, target_shape, ranks, device='cuda'):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.rC, self.rH, self.rW, self.rT = ranks

        self.UH = TuckerFactor(self.H, self.rH, is_complex=False, device=device)
        self.UW = TuckerFactor(self.W, self.rW, is_complex=False, device=device)
        self.UC = TuckerFactor(self.C, self.rC, is_complex=False, device=device)
        self.UT = TuckerFactor(self.T, self.rT, is_complex=False, device=device)

        self.G = nn.Parameter(torch.randn(self.rT, self.rC, self.rH, self.rW, device=device) * 1e-2)

    def forward(self, t):
        UT = self.UT.get(t)
        UC = self.UC()
        UH = self.UH()
        UW = self.UW()
        return tucker_construct(UT, UC, UH, UW, self.G).contiguous()


class ComplexTucker(RealTucker):

    def __init__(self, target_shape, ranks, device='cuda'):
        super().__init__(target_shape, ranks, device=device)
        half_W = (self.W // 2) + 1
        self.UH = TuckerFactor(self.H, self.rH, is_complex=True, device=device)
        self.UW = TuckerFactor(half_W, self.rW, is_complex=True, device=device)
        self.UC = TuckerFactor(self.C, self.rC, is_complex=True, device=device)
        self.UT = TuckerFactor(self.T, self.rT, is_complex=True, device=device)

        self.G = None  # override parent
        self.G_real = nn.Parameter(torch.randn(self.rT, self.rC, self.rH, self.rW, device=device) * 1e-2)
        self.G_imag = nn.Parameter(torch.zeros(self.rT, self.rC, self.rH, self.rW, device=device))

        self.feature_grid = FeatureGrid([self.C * 2, self.H, half_W, self.T], grid_res=[self.C * 2, self.H, half_W, 1], device=device)

    def forward(self, t):
        UH = self.UH()
        UW = self.UW()
        UC = self.UC()
        UT = self.UT.get(t)
        G = torch.complex(self.G_real, self.G_imag)
        construct = tucker_construct(UT, UC, UH, UW, G)

        grid = self.feature_grid(t)
        complex_grid = torch.complex(*grid.chunk(2, dim=1))
        construct = construct * complex_grid
        real_tucker = torch.fft.irfft2(construct, norm='ortho').real
        return real_tucker.contiguous()


def grid_sample_base(H, W, device):
    y_lin = torch.arange(0, H, device=device)
    x_lin = torch.arange(0, W, device=device)
    y_norm = 2.0 * (y_lin / (H - 1)) - 1.0
    x_norm = 2.0 * (x_lin / (W - 1)) - 1.0
    y, x = torch.meshgrid(y_norm, x_norm, indexing='ij')  # [H, W]
    return torch.stack((x, y), dim=-1)  # [H, W, 2]


class FeatureGrid(nn.Module):
    def __init__(self, target_shape, grid_res, zero_init=False, device="cuda"):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.grid_c = grid_res[0]
        self.grid_h = grid_res[1]
        self.grid_w = grid_res[2]
        self.grid_t = grid_res[3]

        self.grid = nn.Parameter(torch.randn(self.grid_c, self.grid_h, self.grid_w, self.grid_t, device=device) * 1e-2)
        if self.grid_c != self.C:
            self.channel_proj = nn.Linear(self.grid_c, self.C, bias=True).to(device)
            nn.init.normal_(self.channel_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.channel_proj.bias)
        self.register_buffer(
            "_xy_base",
            grid_sample_base(self.H, self.W, device=device),
            persistent=False
        )
    
    def _5d_grid(self):
        return self.grid.permute(0, 3, 1, 2).unsqueeze(0)

    def forward(self, t):
        device = self.grid.device
        B = t.shape[0]

        sample_grid3 = torch.empty((B, self.H, self.W, 3), device=device, dtype=self._xy_base.dtype)
        sample_grid3[..., :2] = self._xy_base
        sample_grid3[..., 2] = (2.0 * t - 1.0).view(B, 1, 1)

        sample_grid3 = sample_grid3.unsqueeze(1)  # [B,1,H,W,3]

        grid_5d = self._5d_grid().expand(B, -1, -1, -1, -1)

        sampled = F.grid_sample(
            grid_5d,               # [B, C, T_g, H_g, W_g]
            sample_grid3,           # [B, 1, H_out, W_out, 3]
            mode='bilinear',
            align_corners=False,
            padding_mode='border',
        )  # → [B, C, 1, H_out, W_out]

        result = sampled.squeeze(2)
        if hasattr(self, 'channel_proj'):
            result = self.channel_proj(result.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return result.contiguous()


def tucker_construct(UT, UC, UH, UW, G):
    UT = UT.contiguous()
    UC = UC.contiguous()
    UH = UH.contiguous()
    UW = UW.contiguous()
    G = G.contiguous()

    def _col_norm(M, eps=1e-8):
        if torch.is_complex(M):
            norms_sq = (M.real**2 + M.imag**2).sum(dim=0, keepdim=True)
            norms = torch.sqrt(norms_sq + eps)
        else:
            norms = M.norm(dim=0, keepdim=True) + eps
        return M / norms

    UH = _col_norm(UH)
    UW = _col_norm(UW)
    UC = _col_norm(UC)
    UT = _col_norm(UT)

    X = torch.einsum('ijkl,ti,cj,hk,wl->tchw', G, UT, UC, UH, UW)
    return X


class BasicUpres(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, k, encoding_len=64, device='cuda'):
        super().__init__()
        self.k = k

        self.upres = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels * (k ** 2), kernel_size=1),
            nn.PixelShuffle(upscale_factor=k),
        ).to(device)

        #kaiming init
        for m in self.upres.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        base = self.upres(x)
        return base


class ConvOperator(nn.Module):
    def __init__(self, in_channels, out_channels, h_dim, encoding_len=128, device='cuda'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.operator_head = nn.Sequential(
            nn.Conv2d(in_channels, h_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1, groups=h_dim),
            nn.GELU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=1),
        ).to(device)

        self.operator_tail = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1, groups=h_dim),
            nn.GELU(),
            nn.Conv2d(h_dim, out_channels, kernel_size=1),
        ).to(device)

        self.encoding = FourierEncoding(
            target_dim=encoding_len,
            max_freq=64,
            freq_init="log",
            device=device
        )

        self.t_modulator = nn.Sequential(
            nn.Linear(encoding_len, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 2 * h_dim),
        ).to(device)

        nn.init.zeros_(self.operator_tail[-1].weight)
        nn.init.zeros_(self.operator_tail[-1].bias)
        nn.init.zeros_(self.t_modulator[-1].weight)
        nn.init.zeros_(self.t_modulator[-1].bias)

    def forward(self, x, t):
        initial = self.operator_head(x)
        time_emb = self.encoding(t)
        modulation = self.t_modulator(time_emb)
        gamma, beta = modulation.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.operator_head[-1].out_channels, 1, 1)
        beta = beta.view(-1, self.operator_head[-1].out_channels, 1, 1)
        modulated = initial * (1 + gamma) + beta
        conv_x = self.operator_tail(modulated)
        return conv_x


class NikaBlock(nn.Module):
    def __init__(self, target_shape, k, real_tucker_ranks, complex_tucker_ranks, grid_ranks, conv_hidden, out_channels, device):
        super().__init__()
        self.C, self.H, self.W, self.T = target_shape
        self.H = int(self.H // k); self.W = int(self.W // k)
        self.internal_shape = [self.C, self.H, self.W, self.T]
        self.dT = 1.0 / (self.T - 1)
        self.real_tucker = RealTucker(
            target_shape=self.internal_shape,
            ranks=real_tucker_ranks,
            device=device,
        )
        self.real_tucker = torch.compile(self.real_tucker)

        self.grid_features = FeatureGrid(
            target_shape=self.internal_shape,
            grid_res=grid_ranks,
            device=device,
        )
        self.grid_features = torch.compile(self.grid_features)

        self.complex_tucker = ComplexTucker(
            target_shape=self.internal_shape,
            ranks=complex_tucker_ranks,
            device=device,
        )

        self.n_heads = 3

        self.groupnorm = nn.GroupNorm(num_groups=self.n_heads, num_channels=self.n_heads * self.C).to(device)
        self.groupnorm = torch.compile(self.groupnorm)

        op_hdim = 64
        self.operator_steps = 2

        self.forward_operators = nn.ModuleList()
        self.backward_operators = nn.ModuleList()
        for _ in range(self.operator_steps):
            fwd = ConvOperator(
                in_channels = 2 * self.n_heads * self.C,
                out_channels = self.n_heads * self.C,
                h_dim = op_hdim,
                device = device,
            )
            bwd = ConvOperator(
                in_channels = 2 * self.n_heads * self.C,
                out_channels = self.n_heads * self.C,
                h_dim = op_hdim,
                device = device,
            )
            self.forward_operators.append(torch.compile(fwd))
            self.backward_operators.append(torch.compile(bwd))

        self.upres = BasicUpres(
            in_channels = self.n_heads * self.C,
            out_channels = out_channels,
            hidden = conv_hidden,
            k = k,    
            device = device,
        )
        self.upres = torch.compile(self.upres)
        self.register_buffer(
            "_zero_base",
            torch.zeros(1, self.C, self.H, self.W, device=device),
            persistent=False,
        )
    
    def _create_base_block(self, norm_t, zero_real_tucker=False, zero_complex_tucker=False, zero_feature_grid=False):
        if type(norm_t) is not torch.Tensor:
            norm_t = torch.tensor([norm_t], device=self.grid_features.grid.device, dtype=torch.float32)

        curr_real_tucker = self.real_tucker(norm_t) if not zero_real_tucker else self._zero_base
        curr_real_grid = self.grid_features(norm_t) if not zero_feature_grid else self._zero_base
        curr_complex_tucker = self.complex_tucker(norm_t) if not zero_complex_tucker else self._zero_base

        current_base = torch.cat([curr_real_grid, curr_real_tucker, curr_complex_tucker], dim=1)
        current_input = self.groupnorm(current_base)
        return current_input

    def forward(self, norm_t, zero_real_tucker=False, zero_complex_tucker=False, zero_feature_grid=False, return_operators=False):
        current_input = self._create_base_block(norm_t, zero_real_tucker, zero_complex_tucker, zero_feature_grid)

        operator_residual = torch.zeros_like(current_input)
        for i in range(self.operator_steps):
            step_len = (i + 1) * self.dT
            mask_prev = (norm_t >= step_len)
            norm_t_prev = (norm_t[mask_prev] - step_len) if mask_prev.any() else None
            mask_next = (norm_t <= (1 - step_len))
            norm_t_next = (norm_t[mask_next] + step_len) if mask_next.any() else None

            prev_base = self._create_base_block(norm_t_prev, zero_real_tucker, zero_complex_tucker, zero_feature_grid) if mask_prev.any() else None
            next_base = self._create_base_block(norm_t_next, zero_real_tucker, zero_complex_tucker, zero_feature_grid) if mask_next.any() else None

            prev_frames = torch.zeros_like(current_input); prev_frames[mask_prev] = prev_base
            forward_prediction = self.forward_operators[i](torch.cat([prev_frames, current_input], dim=1), norm_t_prev)
            operator_residual += forward_prediction

            next_frames = torch.zeros_like(current_input); next_frames[mask_next] = next_base
            backward_prediction = self.backward_operators[i](torch.cat([current_input, next_frames], dim=1), norm_t_next)
            operator_residual += backward_prediction

        aggregated = current_input + operator_residual
        refined = self.upres(aggregated)

        if return_operators:
            refined_forward = self.upres(forward_prediction)
            refined_backward = self.upres(backward_prediction)
            return refined, refined_forward, refined_backward
        return refined


if __name__ == "__main__":
    device = "cuda:1"
    name = "jockey"
    torch.manual_seed(42)
    vid = load_video_frames(f"static/benchmarks/uvg/{name}", device, max_frames=600, dtype=torch.uint8, normalize=False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    feature_test(vid, name, f"large", device=device)
