import os

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from soap import SOAP
import time
from dataloader import create_param_dataloaders


def vrmse(pred, target, eps=1e-7):
    mse = ((pred - target) ** 2).mean(dim=(-2, -1))    # (B, C)
    var = target.std(dim=(-2, -1)) ** 2                # (B, C)
    return torch.sqrt(mse / (var + eps)).mean()


class TuckerStateEncoder(nn.Module):
    def __init__(self, in_channels, latent_shape, ranks, h_dim, device, is_complex=False):
        super().__init__()
        self.in_channels = in_channels
        C, H, W = latent_shape
        print(f"Initializing TuckerStateEncoder with latent shape {latent_shape} and ranks {ranks}")
        rC, rH, rW = ranks
        self.is_complex = bool(is_complex)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, h_dim, 1),
            nn.GELU(),
            nn.Conv2d(h_dim, h_dim, 3, padding=1, groups=h_dim),
            nn.GELU(),
            nn.Conv2d(h_dim, h_dim, 1),
            nn.GELU(),
            nn.Conv2d(h_dim, h_dim, 3, padding=1, groups=h_dim),
            nn.GELU(),
        ).to(device)

        # channel head: if complex, emit real+imag -> 2x params
        channel_out = C * rC * (2 if self.is_complex else 1)
        self.channel_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, channel_out),
        ).to(device)


        # height head: final out channels doubled for complex
        height_out = rH * (2 if self.is_complex else 1)
        self.height_head = nn.Sequential(
            nn.Conv1d(h_dim, h_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(h_dim, height_out, 1),
        ).to(device)


        # width head: final out channels doubled for complex
        width_out = rW * (2 if self.is_complex else 1)
        self.width_head = nn.Sequential(
            nn.Conv1d(h_dim, h_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(h_dim, width_out, 1),
        ).to(device)

        self.C = C
        self.rC = rC

    def forward(self, x):
        # x: [B, C_in, H, W]
        f = self.trunk(x)                      # [B, D, H, W]

        f_c = f.mean(dim=(-1, -2))            # [B, D]
        ch_out = self.channel_head(f_c)
        if self.is_complex:
            ch_out = ch_out.view(-1, self.C, self.rC * 2)  # (B, C, rC*2)
            real = ch_out[..., : self.rC]
            imag = ch_out[..., self.rC :]
            U_C = torch.complex(real, imag)
        else:
            U_C = ch_out.view(-1, self.C, self.rC)

        f_h = f.mean(dim=-1)                  # [B, D, H]
        hh = self.height_head(f_h).permute(0, 2, 1)   # [B, H, rH*(2?)]
        if self.is_complex:
            hh = hh.view(hh.shape[0], hh.shape[1], -1)  # (B, H, rH*2)
            real = hh[..., : (hh.shape[2] // 2)]
            imag = hh[..., (hh.shape[2] // 2) :]
            U_H = torch.complex(real, imag)
        else:
            U_H = hh

        f_w = f.mean(dim=-2)                  # [B, D, W]
        wh = self.width_head(f_w).permute(0, 2, 1)   # [B, W, rW*(2?)]
        if self.is_complex:
            wh = wh.view(wh.shape[0], wh.shape[1], -1)  # (B, W, rW*2)
            real = wh[..., : (wh.shape[2] // 2)]
            imag = wh[..., (wh.shape[2] // 2) :]
            U_W = torch.complex(real, imag)
            real_p = F.avg_pool1d(U_W.real.permute(0, 2, 1), kernel_size=2, stride=2, padding=1).permute(0, 2, 1)
            imag_p = F.avg_pool1d(U_W.imag.permute(0, 2, 1), kernel_size=2, stride=2, padding=1).permute(0, 2, 1)
            U_W = torch.complex(real_p, imag_p)
        else:
            U_W = wh

        return U_C, U_H, U_W


class ParamConditioner(nn.Module):
    def __init__(self, in_channels, out_dim, h_dim, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, out_dim),
        ).to(device)

    def forward(self, params):
        return self.net(params)


class FastUpres(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, hidden=64, k=2):
        super(FastUpres, self).__init__()
        self.conv_upres = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels * k * k, kernel_size=1),
            nn.PixelShuffle(upscale_factor=k)
        )

    def forward(self, x):
        x = self.conv_upres(x)
        return x


def tucker_construct(UC, UH, UW, G):
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

    X = torch.einsum('bijk,bci,bhj,bwk->bchw', G, UC, UH, UW)
    return X


class ConvOperator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x):
        return self.net(x)

class NikoBlock(nn.Module):
    def __init__(self, target_shape, k, core_ranks, h_dim, context_len, pred_len, num_params, device, min_eps=1e-3):
        super().__init__()
        C, H, W = target_shape
        rC, rH, rW, rP = core_ranks 
        self.real_encoder = TuckerStateEncoder(
            C,
            [C, H // k, W // k],
            [rC, rH, rW],
            h_dim,
            device
)

        self.complex_encoder = TuckerStateEncoder(
            C,
            [C, H // k, W // k],
            [rC, rH, rW],
            h_dim,
            device,
            is_complex=True,
        )

        self.real_core = nn.Parameter(torch.randn(core_ranks, device=device) * 0.01)
        self.complex_core_real = nn.Parameter(torch.randn(core_ranks, device=device) * 0.01)
        self.complex_core_imag = nn.Parameter(torch.zeros(core_ranks, device=device))

        self.core_modulator = ParamConditioner(num_params, core_ranks[-1], h_dim, device)
        self.groupnorm = nn.GroupNorm(num_groups=3, num_channels=3 * C, affine=False).to(device)

        self.operators = nn.ModuleList()
        for _ in range(context_len - 1):
            self.operators.append(ConvOperator(in_channels=2 * 3 * C, out_channels=3 * C, hidden=h_dim).to(device))

        self.upres = FastUpres(in_channels=3 * C, out_channels=C, hidden=h_dim, k=k).to(device)

        self.k = k
    
    def _construct_block(self, frame, params):
        U_C, U_H, U_W = self.real_encoder(frame)
        U_P = self.core_modulator(params)
        mod_core = torch.einsum("bp, chwp -> bchw", U_P, self.real_core)

        real_latent = tucker_construct(U_C, U_H, U_W, mod_core)

        U_C_c, U_H_c, U_W_c = self.complex_encoder(frame)
        mod_c_core_real = torch.einsum("bp, chwp -> bchw", U_P, self.complex_core_real)
        mod_c_core_imag = torch.einsum("bp, chwp -> bchw", U_P, self.complex_core_imag)
        complex_core = torch.complex(mod_c_core_real, mod_c_core_imag)
        complex_latent = tucker_construct(U_C_c, U_H_c, U_W_c, complex_core)
        complex_base = torch.fft.rfft2(frame)
        complex_latent = complex_latent * complex_base
        complex_component = torch.fft.irfft2(complex_latent, norm="ortho")

        combined = torch.cat([frame, real_latent, complex_component], dim=1)  # (B, 3C, H, W)
        combined = self.groupnorm(combined)
        return combined

    def forward(self, context_frames, params):
        params = params.to(dtype=torch.float32, device=next(self.parameters()).device)
        base_frame = context_frames[:, -1, ...]  # (B, 2, H, W)

        downres_frames = context_frames[..., ::self.k, ::self.k]
        current_frames = downres_frames[:, -1, ...]  # (B, 2, H, W)

        core_basis = self._construct_block(current_frames, params)  # (B, 3C, H//k, W//k)
    
        operator_residual = torch.zeros_like(core_basis)
        for i, op in enumerate(self.operators):
            frame = downres_frames[:, i, ...]
            context = self._construct_block(frame, params)
            op_input = torch.cat([context, core_basis], dim=1)
            operator_residual = operator_residual + op(op_input)

        pred = F.tanh(self.upres(core_basis + operator_residual))
        return pred + base_frame


def train_niko(
    data_dir,
    pred_test,
    batch_size,
    device,
    epochs: int = 100,
    accum_steps: int = 4,
    log_dir: str = "logs",
    max_hours: float = 12.0,
    log_interval: int = 100,
    val_every: int = 1,
):
    context_frames, predict_frames = pred_test
    train_loader, val_loader, params = create_param_dataloaders(
        data_dir,
        batch_size=batch_size,
        context_frames=context_frames,
        predict_frames=predict_frames,
        device=device,
    )

    k = 2
    num_control_params = 2
    os.makedirs(log_dir, exist_ok=True)

    print(f"length of loader: {len(train_loader)}")
    model = NikoBlock(
        target_shape=(4, 128, 512),
        k=k,
        core_ranks=(4, 50, 100, 2),
        h_dim=64,
        context_len=context_frames,
        pred_len=predict_frames,
        num_params=num_control_params,
        device=device,
    )

    opt = SOAP(model.parameters())
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    # timing and training caps
    train_time = 0.0
    start_wall = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        batch_count = 0
        epoch_loss_sum = 0.0
        epoch_batches = 0

        interval_loss = 0.0
        interval_batches = 0
        interval_time = 0.0

        for batch in train_loader:
            batch_start = time.time()

            # dataset must return (ctx, tgt, params)
            if len(batch) != 3:
                raise ValueError("Dataset must return (ctx, tgt, params) tuples")
            base_frames, target_frames, batch_params = batch

            base_frames = base_frames.to(device)
            if predict_frames == 1:
                target_frames = target_frames.to(device).squeeze(1)
            else:
                target_frames = target_frames.to(device)

            pred = model(base_frames, batch_params)
            loss = vrmse(pred, target_frames) / accum_steps
            loss.backward()

            batch_count += 1
            epoch_loss_sum += (loss.item() * accum_steps)
            epoch_batches += 1

            # interval stats
            interval_loss += (loss.item() * accum_steps)
            interval_batches += 1

            if batch_count % accum_steps == 0:
                opt.step()
                opt.zero_grad()

            batch_end = time.time()
            elapsed = batch_end - batch_start
            train_time += elapsed
            interval_time += elapsed

            if batch_count % log_interval == 0:
                avg_interval_loss = interval_loss / interval_batches if interval_batches else float('inf')
                # fps: predicted frames per second = (batch_size * predict_frames) / (avg batch time)
                avg_batch_time = interval_time / interval_batches if interval_batches else 1e-9
                fps = (batch_size * predict_frames) / avg_batch_time
                print(
                    f"Epoch {epoch}, batch {batch_count}, avg loss (last {interval_batches}): {avg_interval_loss:.6f}, fps: {fps:.1f}"
                )
                interval_loss = 0.0
                interval_batches = 0
                interval_time = 0.0

            # enforce max training-only time
            if train_time >= max_hours * 3600:
                print("Reached training time cap; stopping.")
                break

        # step leftover gradients
        if batch_count % accum_steps != 0:
            opt.step()
            opt.zero_grad()

        avg_train_loss = epoch_loss_sum / epoch_batches if epoch_batches > 0 else float('inf')
        print(f"Epoch {epoch} finished. Avg train loss: {avg_train_loss:.6f}. Training-only time so far: {train_time:.1f}s")

        # save only model state_dict
        checkpoint_path = os.path.join(log_dir, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)

        # validation every `val_every` epochs (use full val set). Time it separately and do not add to train_time
        if epoch % val_every == 0:
            model.eval()
            val_start = time.time()
            total_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    # validation dataset must return (ctx, tgt, params)
                    if len(batch) != 3:
                        raise ValueError("Validation dataset must return (ctx, tgt, params) tuples")
                    base_frames, target_frames, batch_params = batch

                    base_frames = base_frames.to(device)
                    if predict_frames == 1:
                        target_frames = target_frames.to(device).squeeze(1)
                    else:
                        target_frames = target_frames.to(device)

                    pred = model(base_frames, batch_params)
                    loss = vrmse(pred, target_frames)
                    total_loss += loss.item()
                    val_batches += 1

            val_end = time.time()
            val_time = val_end - val_start
            avg_val_loss = total_loss / val_batches if val_batches > 0 else float('inf')
            delta = avg_train_loss - avg_val_loss
            print(f"Validation (epoch {epoch}) Avg loss: {avg_val_loss:.6f}, val_time: {val_time:.1f}s, train-val delta: {delta:.6f}")

        # stop if reached global time cap
        if train_time >= max_hours * 3600:
            break


if __name__ == "__main__":
    device = "cuda:0"
    DATA_DIR = "/app/data/datasets/rayleigh_benard/data/"
    pred_test = (4, 1)
    batch_size = 5

    train_niko(DATA_DIR, pred_test, batch_size, device)
