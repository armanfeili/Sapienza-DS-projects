import functools

import torch
import torch.nn as nn
import numpy as np

from modules import (
    SharedMLP,
    PVConv,
    PointNetSAModule,
    PointNetAModule,
    PointNetFPModule,
    Attention,
    Swish,
)
from model.transformer_branch import DiT
from modules.voxelization import Voxelization


# ─────────────────────────────────────────────────────────────────────────────
def _linear_gn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.GroupNorm(8, out_channels),
        Swish()
    )


def create_mlp_components(
    in_channels, out_channels, classifier=False, dim=2, width_multiplier=1
):
    r = width_multiplier

    if dim == 1:
        block = _linear_gn_relu
    else:
        block = SharedMLP

    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (
        len(out_channels) == 1 and out_channels[0] is None
    ):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc

    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_gn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))

    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet_components(
    blocks,
    in_channels,
    embed_dim,
    with_se=False,
    normalize=True,
    eps=0,
    width_multiplier=1,
    voxel_resolution_multiplier=1,
):
    r, vr = width_multiplier, voxel_resolution_multiplier

    layers, concat_channels = [], 0
    c = 0
    for k, (out_channels, num_blocks, voxel_resolution) in enumerate(blocks):
        out_channels = int(r * out_channels)
        for p in range(num_blocks):
            attention = (k % 2 == 0 and k > 0 and p == 0)
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(
                    PVConv,
                    kernel_size=3,
                    resolution=int(vr * voxel_resolution),
                    attention=attention,
                    with_se=with_se,
                    normalize=normalize,
                    eps=eps,
                )

            if c == 0:
                layers.append(block(in_channels, out_channels))
            else:
                layers.append(block(in_channels + embed_dim, out_channels))

            in_channels = out_channels
            concat_channels += out_channels
            c += 1

    return layers, in_channels, concat_channels


def create_pointnet2_sa_components(
    sa_blocks,
    extra_feature_channels,
    embed_dim=64,
    use_att=False,
    dropout=0.1,
    with_se=False,
    normalize=True,
    eps=0,
    width_multiplier=1,
    voxel_resolution_multiplier=1,
):
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + 3

    sa_layers, sa_in_channels = [], []
    c = 0
    for conv_configs, sa_configs in sa_blocks:
        k = 0
        sa_in_channels.append(in_channels)
        sa_subblocks = []

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c + 1) % 2 == 0 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(
                        PVConv,
                        kernel_size=3,
                        resolution=int(vr * voxel_resolution),
                        attention=attention,
                        dropout=dropout,
                        with_se=with_se,
                        with_se_relu=True,
                        normalize=normalize,
                        eps=eps,
                    )

                if c == 0:
                    sa_subblocks.append(block(in_channels, out_channels))
                elif k == 0:
                    sa_subblocks.append(block(in_channels + embed_dim, out_channels))

                in_channels = out_channels
                k += 1
            extra_feature_channels = in_channels

        num_centers, radius, num_neighbors, out_channels = sa_configs
        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels

        if num_centers is None:
            block = PointNetAModule
        else:
            block = functools.partial(
                PointNetSAModule, num_centers=num_centers, radius=radius, num_neighbors=num_neighbors
            )

        sa_subblocks.append(
            block(
                in_channels=extra_feature_channels + (embed_dim if k == 0 else 0),
                out_channels=out_channels,
                include_coordinates=True,
            )
        )
        c += 1
        in_channels = extra_feature_channels = sa_subblocks[-1].out_channels

        if len(sa_subblocks) == 1:
            sa_layers.append(sa_subblocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_subblocks))

    return sa_layers, sa_in_channels, in_channels, (1 if num_centers is None else num_centers)


def create_pointnet2_fp_modules(
    fp_blocks,
    in_channels,
    sa_in_channels,
    embed_dim=64,
    use_att=False,
    dropout=0.1,
    with_se=False,
    normalize=True,
    eps=0,
    width_multiplier=1,
    voxel_resolution_multiplier=1,
):
    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    c = 0
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_subblocks = []

        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_subblocks.append(
            PointNetFPModule(
                in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim,
                out_channels=out_channels,
            )
        )
        in_channels = out_channels[-1]

        if conv_configs is not None:
            outc, num_blocks, voxel_resolution = conv_configs
            outc = int(r * outc)
            for p in range(num_blocks):
                attention = (c + 1) % 2 == 0 and c < len(fp_blocks) - 1 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(
                        PVConv,
                        kernel_size=3,
                        resolution=int(vr * voxel_resolution),
                        attention=attention,
                        dropout=dropout,
                        with_se=with_se,
                        with_se_relu=True,
                        normalize=normalize,
                        eps=eps,
                    )

                fp_subblocks.append(block(in_channels, outc))
                in_channels = outc
        if len(fp_subblocks) == 1:
            fp_layers.append(fp_subblocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_subblocks))

        c += 1

    return fp_layers, in_channels


# ─────────────────────────────────────────────────────────────────────────────
class Tiger_Transformer(nn.Module):

    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(
        self,
        num_classes,
        embed_dim,
        use_att,
        dropout=0.1,
        extra_feature_channels=3,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
    ):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.in_channels = extra_feature_channels + 3

        # ─── Build the “Set Abstraction” (SA) layers ─────────────────────────
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks,
            extra_feature_channels=extra_feature_channels,
            with_se=True,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        # ─── The Transformer branch for global features ─────────────────────
        self.transformer = DiT(
            depth=8, tok_num=256, hidden_size=128, latent_size=512, output_channel=256
        )

        # ─── Feature fusion / adjustment modules ────────────────────────────
        self.feature_adjust = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.GroupNorm(8, num_channels=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
        )

        # ─── Build the “Feature Propagation” (FP) layers ────────────────────
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        # ─── The final classifier ───────────────────────────────────────────
        layers, _ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[128, dropout, num_classes],  # dropout used as “width” here
            classifier=True,
            dim=2,
            width_multiplier=width_multiplier,
        )
        self.classifier = nn.Sequential(*layers)

        # ─── Time embedding MLP (for diffusion‐timestep → vector) ──────────
        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        # ─── Time‐mask & remapping for feature fusion during FP stage ──────
        self.t_mask_op = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GroupNorm(8, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, 256),
            nn.Sigmoid(),
        )

        self.conv_remap = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.GroupNorm(8, num_channels=256),
            nn.Sigmoid(),
        )
        self.tf_remap = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.GroupNorm(8, num_channels=256),
            nn.Sigmoid(),
        )

    def get_timestep_embedding(self, timesteps: torch.Tensor, device):
        """
        Standard sinusoidal time embedding (used by many diffusion models).
        timesteps: (B,), dtype=torch.int64
        returns:   (B, embed_dim)
        """
        assert len(timesteps.shape) == 1
        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        emb = timesteps[:, None] * emb[None, :]               # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, embed_dim)
        if self.embed_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        return emb  # (B, embed_dim)

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        """
        inputs: (B, in_channels + extra_features, N)
        t:      (B,)
        """
        # ─── Build a time‐embedding per point (B, embed_dim, N) ─────────────
        temb = self.embedf(self.get_timestep_embedding(t, inputs.device))[:, :, None].expand(
            -1, -1, inputs.shape[-1]
        )  # (B, embed_dim, N)

        coords, features = inputs[:, :3, :].contiguous(), inputs
        coords_list, in_features_list = [], []

        # ─── Apply Set‐Abstraction (SA) layers ─────────────────────────────
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i == 0:
                features, coords, temb = sa_blocks((features, coords, temb))
            else:
                features, coords, temb = sa_blocks(
                    (torch.cat([features, temb], dim=1), coords, temb)
                )

        # Override the “extra” features from the first level with nothing
        in_features_list[0] = inputs[:, 3:, :].contiguous()

        # ─── Apply Feature Propagation (FP) layers, injecting Transformer once ─
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            if fp_idx == 1:
                # Run the global Transformer branch (DiT) on the “last‐but‐one” features
                # and coordinates
                features_tf = self.transformer(
                    in_features_list[-1 - fp_idx], t, coords_list[-1 - fp_idx]
                )

            features, coords, temb = fp_blocks(
                (
                    coords_list[-1 - fp_idx],
                    coords,
                    torch.cat([features, temb], dim=1),
                    in_features_list[-1 - fp_idx],
                    temb,
                )
            )
            if fp_idx == 1:
                # Fuse Transformer features vs. convolutional features via a time‐mask
                t_mask = self.t_mask_op(torch.mean(temb, dim=-1))
                features = (
                    t_mask[:, :, None] * self.conv_remap(features)
                    + (1 - t_mask[:, :, None]) * self.tf_remap(features_tf)
                )
                features = self.feature_adjust(features)

        return self.classifier(features)

    # ─────────────────────────────────────────────────────────────────────────
    def get_pspe_feats(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute PSPE (Phase‐Shifted Positional Encoding) for a noisy point cloud.

        Inputs:
          x_t: (B, 3, N)   – the noisy point cloud at timestep t
          t:   (B,)        – the integer diffusion timestep for each sample
        Returns:
          pspe: (B, embed_dim, N)
        """
        B, _, N = x_t.shape

        # Use sinusoidal time embedding repeated per point
        temb = self.get_timestep_embedding(t, x_t.device)      # (B, embed_dim)
        pspe = temb.unsqueeze(-1).repeat(1, 1, N)               # (B, embed_dim, N)

        return pspe

    # ─────────────────────────────────────────────────────────────────────────
    def get_bape_feats(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute BAPE (Binary‐Applied Positional Encoding) for a noisy point cloud.

        Inputs:
          x_t: (B, 3, N)   – the noisy point cloud at timestep t
          t:   (B,)        – the integer diffusion timestep for each sample
        Returns:
          bape: (B, 1, N)
        """
        B, _, N = x_t.shape

        # Create a binary mask based on the sign of the z‐coordinate
        coords = x_t                              # (B, 3, N)
        bape = (coords[:, 2:3, :] > 0.0).float()  # (B, 1, N)

        return bape
