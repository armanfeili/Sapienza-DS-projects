# Smart Environment Final Project

Professors:
- Prof. Francesca Cuomo
- Prof. Stefania Colonnese

Team members:

1.
- Name: Syed Habibul
- Surname: Bashar
- Matricola: 2102742
- Email: bashar.2102742@studenti.uniroma1.it

2.
- Name: Arman
- Surname: Feili
- Matricola: 2101835
- Email: feili.2101835@studenti.uniroma1.it

3.
- Name: Aysegul Sine
- Surname: Ozgenkan
- Matricola: 2108754
- Email: ozgenkan.2108754@studenti.uniroma1.it

Paper:
- https://openaccess.thecvf.com/content/CVPR2024/papers/Ren_TIGER_Time-Varying_Denoising_Model_for_3D_Point_Cloud_Generation_with_CVPR_2024_paper.pdf

Github Repository:
- https://github.com/Zhiyuan-R/Tiger-Diffusion/tree/main?tab=readme-ov-file

Datasets:
- ShapeNetCore.v2.PC15k (https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j)


# TIGER: Time‐Varying Denoising Model

This repository contains code to train, evaluate, and analyze **TIGER**, a diffusion‐based generative model for 3D point clouds. TIGER combines a Transformer branch (for global shape understanding) with a CNN branch (for local detail modeling) and adaptively fuses their features over diffusion timesteps. The result is a state‐of‐the‐art generative model for 2048‐point 3D shapes (e.g., ShapeNet).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Data Preparation](#data-preparation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Feature Analysis & Visualization](#feature-analysis--visualization)
8. [Dependency Summary](#dependency-summary)
9. [Contact & Citation](#contact--citation)

---

## Project Overview

**TIGER** (Time‐Varying denoisinG mEthod foR 3D point clouds) implements a two‐stream diffusion model:

* A **Transformer branch** that captures global shape structure in early diffusion steps.
* A **CNN (Point‐Voxel) branch** that refines local geometric details in later diffusion steps.
* A **time‐mask** that learns to fuse Transformer‐derived and convolutional features as a function of the diffusion timestep.

On ShapeNetCore.v2 (15k models, 2048 points each), TIGER achieves state‐of‐the‐art Chamfer Distance (CD) and Earth Mover’s Distance (EMD) across categories (e.g., car, chair, airplane). This codebase covers:

1. **Environment fix** for NumPy compatibility in Colab/other Python 3.10+ environments.
2. **Dependency installation** (PyTorch 2.3.0 with CUDA 12.1, PyTorch Geometric, PyTorch3D, etc.).
3. **Data download** and directory setup for ShapeNetCore.v2.PC15k.
4. **Quick sanity‐check training** (3 epochs) and **full training** (200 epochs) on a single category (car).
5. **Evaluation metrics** (MMD‐CD, MMD‐EMD, F‐score) on generated vs. real point clouds.
6. **Feature extraction** (PSPE, BAPE) at selected diffusion timesteps and PCA/heatmap visualization.
7. **Shape initialization experiment**: generate cars from six canonical shapes (cube, sphere, etc.) and compare.

---

## Repository Structure

```
TIGER-Diffusion/
├── datasets/
│   └── shapenet_data_pc.py           # Data loader for ShapeNet15kPointClouds
├── metrics/
│   ├── evaluation_metrics.py         # Chamfer, EMD, F‐score, and related helpers
│   └── other 3D‐metric utilities
├── modules/
│   ├── SharedMLP.py                  # Shared MLP block
│   ├── PVConv.py                     # Point‐Voxel Conv block
│   ├── PointNetSAModule.py           # Set Abstraction module
│   ├── PointNetFPModule.py           # Feature Propagation module
│   ├── Attention.py                  # Attention utilities
│   ├── Swish.py                      # Swish activation
│   └── voxelization.py               # Voxelization helper
├── model/
│   └── transformer_branch.py         # DiT‐style transformer for global features
├── train_generation.py               # Main script: train and evaluate TIGER
├── requirements_colab.txt            # Pip requirements for Colab (and most Linux/Python setups)
├── output/
│   └── train_generation/             # Saved checkpoints and logs (auto‐created by training script)
├── real_car_batch.npy                # (Optional) saved batch of real car point clouds (64 × 2048 × 3)
├── README.md                         # This file
└── LICENSE.md                        # License (if applicable)
```

---

## Environment Setup

> **Note**: The current instructions assume a Linux/Colab environment with Python 3.10. If you use another OS or Python version, adapt the commands accordingly.

1. **Fix NumPy Compatibility (Colab/Python 3.10+)**

   Newer NumPy (≥ 1.24) breaks some SciPy/Scikit‐learn imports. Before loading any of these libraries, run:

   ```bash
   # Reinstall compatible versions
   pip install -q numpy==1.26.4 scipy==1.11.4 scikit-learn==1.3.2
   ```

   Then, either:

   * Manually restart your Python runtime, **OR**
   * In Colab: run

     ```python
     import os
     os.kill(os.getpid(), 9)
     ```

   Once the runtime restarts, **re-run**:

   ```bash
   pip install -q numpy==1.26.4 scipy==1.11.4 scikit-learn==1.3.2
   ```

   Then add this shim **before** importing any NumPy/SciPy/Scikit‐learn code:

   ```python
   import sys, importlib, types

   # 1) numpy.rec
   try:
       recmod = importlib.import_module("numpy.core.records")
   except ImportError:
       recmod = types.ModuleType("numpy.rec")
   sys.modules["numpy.rec"] = recmod

   # 2) numpy.strings
   try:
       strmod = importlib.import_module("numpy._core.defchararray")
   except ImportError:
       strmod = types.ModuleType("numpy.strings")
   sys.modules["numpy.strings"] = strmod

   import numpy as _np
   _np.rec     = recmod
   _np.strings = strmod
   ```

2. **Clone this repository**

   ```bash
   git clone https://github.com/YourUsername/TIGER-Diffusion.git
   cd TIGER-Diffusion
   ```

3. **Python Dependencies**

   This project is tested on:

   * Python 3.10+
   * CUDA 12.1 (for GPU builds)
   * PyTorch 2.3.0+cu121
   * PyTorch Geometric matching torch=2.3.1+cu121
   * PyTorch3D (built from source)
   * NumPy 1.26.4, SciPy 1.11.4, scikit‐learn 1.3.2

   Install the required packages:

   ```bash
   # 1. Install PyTorch + TorchVision + Torchaudio (CUDA 12.1)
   # Note: on non‐CUDA machines, install CPU‐only wheels or use your own PyTorch version.
   TORCH_WHL=https://download.pytorch.org/whl/cu121
   pip install -q \
     torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 \
     --index-url $TORCH_WHL

   # 2. Install PyTorch Geometric (for the matching torch + CUDA build)
   pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
     -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

   # 3. Install other required packages
   pip install -q -r requirements_colab.txt

   # 4. Install PyTorch3D (from GitHub)
   pip install -q ninja cmake
   pip install -q "git+https://github.com/facebookresearch/pytorch3d.git"
   ```

4. **Verify GPU & Metric Utilities**

   In a Python shell (or a Jupyter cell), run:

   ```python
   import torch
   from sklearn.neighbors import NearestNeighbors
   from metrics.evaluation_metrics import cham3D, EMD

   x = torch.rand(2, 1024, 3, device="cuda")
   y = torch.rand(2, 1024, 3, device="cuda")
   print("Chamfer:", cham3D(x, y)[0].mean().item())
   print("EMD:", EMD(x, y).mean().item())
   ```

   If all imports succeed and you see numeric outputs, your GPU + metric code is ready.

---

## Data Preparation

TIGER is trained and evaluated on the **ShapeNetCore.v2.PC15k** dataset (PointCloud version with 15k models, 2048 points each). Download and organize the data as follows:

1. **Download the ShapeNetCore.v2.PC15k folder (\~7 GB)**

   * The dataset is stored in a Google Drive folder (from PointFlow authors):

     ```
     https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j
     ```
   * In a new terminal (inside the repo), run:

     ```bash
     pip install -q gdown
     gdown --folder --id 1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j -O ShapeNetCore.v2.PC15k
     ```

2. **Unzip and organize**

   ```bash
   cd ShapeNetCore.v2.PC15k
   unzip -q ShapeNetCore.v2.PC15k.zip -d ShapeNetCore.v2.PC15k
   rm ShapeNetCore.v2.PC15k.zip

   # If the unzipped content is nested, move it up one level:
   # mv ShapeNetCore.v2.PC15k/ShapeNetCore.v2.PC15k/* .
   # rmdir ShapeNetCore.v2.PC15k/ShapeNetCore.v2.PC15k
   cd ..
   ```

3. **Verify directory structure**

   After unzipping, you should see subfolders named by synset IDs (e.g., `02691156`, `02958343`, …). Each subfolder contains `.npy` point‐cloud files for training/validation/test splits.

   ```bash
   ls ShapeNetCore.v2.PC15k | head
   # Example output: 02691156 02958343 02974003 03001627 …
   ```

---

## Training

### 1. Quick Sanity‐Check (3 epochs)

This short run verifies that data loading, model forwarding, and logging work without errors.

```bash
# From repository root
python train_generation.py \
  --category car \
  --bs 16 \
  --niter 3 \
  --saveIter 1 \
  --print_freq 50 \
  --workers 2 \
  --manualSeed 42
```

* `--category car` selects the “car” synset.
* `--bs 16` sets batch size to 16.
* `--niter 3` does 3 total epochs.
* `--saveIter 1` saves a checkpoint after each epoch.
* Logs and checkpoints appear in `output/train_generation/<timestamp>/`.

Check `output/train_generation/<latest>/output.log` to confirm that training started, loss decreased, and no GPU errors occurred.

---

### 2. Full Training (200 epochs)

Once the quick check passes, launch the full run:

```bash
python train_generation.py \
  --category car \
  --bs 32 \
  --workers 2 \
  --niter 200 \
  --saveIter 20 \
  --diagIter 10 \
  --vizIter 10 \
  --print_freq 100 \
  --embed_dim 128 \
  --dropout 0.01 \
  --lr 5e-5 \
  --beta_start 1e-6 \
  --beta_end 0.015 \
  --schedule_type warm0.1 \
  --lr_gamma 0.9998 \
  --decay 1e-5 \
  --grad_clip 1.0 \
  --manualSeed 42
```

* `--embed_dim 128`: set feature embedding dimension.
* `--beta_start 1e-6 --beta_end 0.015`: noise schedule bounds.
* `--schedule_type warm0.1`: learning‐rate warmup.
* `--saveIter 20`: checkpoint every 20 epochs.
* `--diagIter 10`, `--vizIter 10`: print diagnostic logs and visualize intermediate outputs every 10 iterations.
* `output/train_generation/<timestamp>/` will contain:

  * Checkpoints: `epoch_<N>.pth`
  * Log file: `output.log`
  * Optional visualizations (folder `viz/` if enabled)

**Note**: Training 200 epochs on a single V100/RTX 3090 GPU typically takes several hours. Monitor `output.log` to track loss, ‖W‖, and ‖∇W‖.

---

## Evaluation

After training finishes, pick the latest checkpoint (highest epoch number) and run evaluation:

1. **Locate latest checkpoint**

   In Python (or a shell), find the newest folder under `output/train_generation/` and then the largest `epoch_*.pth`. For example:

   ```bash
   ls -1t output/train_generation/*/epoch_*.pth | head -n 1
   # Returns something like:
   # output/train_generation/2025-05-09-19-26-51/epoch_199.pth
   ```

2. **Generate 64 synthetic samples and compare**

   Use the built‐in `eval` mode in `train_generation.py`. In a Python session (or Jupyter):

   ```python
   import os, glob
   import torch
   from train_generation import parse_args, get_betas, Model

   # 1) Set project root and find latest checkpoint
   PROJ_ROOT = os.getcwd()
   RUN_PARENT = os.path.join(PROJ_ROOT, "output", "train_generation")
   latest_run = max(
       [d for d in glob.glob(os.path.join(RUN_PARENT, "*")) if os.path.isdir(d)],
       key=os.path.getmtime
   )
   ckpt_path = max(
       glob.glob(os.path.join(latest_run, "epoch_*.pth")),
       key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0])
   )

   # 2) Parse default args & override
   import sys
   sv_argv, sys.argv = sys.argv, ['eval']
   args = parse_args()
   sys.argv = sv_argv

   args.dataroot = "ShapeNetCore.v2.PC15k/"
   args.category = "car"
   args.distribution_type = "single"

   # 3) Load checkpoint & rebuild model
   ckpt = torch.load(ckpt_path, map_location='cpu')
   state_dict = ckpt['model_state']
   args.embed_dim = state_dict["model.embedf.0.weight"].shape[0]
   betas = get_betas(args.schedule_type, args.beta_start, args.beta_end, args.time_num)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = Model(args, betas, args.loss_type, args.model_mean_type, args.model_var_type).to(device)

   own_sd = model.state_dict()
   matched = 0
   for k, v in state_dict.items():
       if k in own_sd and v.shape == own_sd[k].shape:
           own_sd[k] = v
           matched += 1
   model.load_state_dict(own_sd, strict=False)
   model.eval()

   # 4) Generate 64 samples (shape: 64 × 2048 × 3)
   with torch.no_grad():
       samples = model.gen_samples((64, 3, args.npoints), device)
       samples = samples.transpose(1, 2).cpu()  # (64, 2048, 3)

   # 5) Load 64 real validation shapes
   from datasets.shapenet_data_pc import ShapeNet15kPointClouds
   val_set = ShapeNet15kPointClouds(
       root_dir=args.dataroot,
       categories=[args.category],
       split='val',
       te_sample_size=args.npoints,
       tr_sample_size=args.npoints,
       scale=1.0,
       normalize_per_shape=False,
       normalize_std_per_axis=False
   )
   import torch
   idx = torch.randperm(len(val_set))[:64]
   refs = []
   for i in idx:
       item = val_set[int(i)]
       for k in ('points', 'train_points', 'pts', 'point_set'):
           if k in item:
               pc = item[k]; break
       else:
           pc = next(v for v in item.values() if torch.is_tensor(v) and v.ndim == 2 and v.shape[1] == 3)
       if pc.shape[0] == 3 and pc.shape[1] != 3:
           pc = pc.T
       refs.append(pc)
   refs = torch.stack(refs).cpu()  # (64, 2048, 3)

   # 6) Compute metrics (MMD-CD, MMD-EMD, F-score)
   from metrics.evaluation_metrics import EMD_CD
   metrics = EMD_CD(samples.cuda(), refs.cuda(), batch_size=32)
   print("\nEvaluation Results (64 samples):")
   for k, v in metrics.items():
       print(f"{k}: {v.item():.6f}")
   ```

   Example output (car category at epoch 199):

   ```
   MMD-CD   : 2.237249
   MMD-EMD  : 0.804075
   F-score  : 0.001530
   ```

3. **Visualize Chamfer Distance Histogram**

   Inside the same Python session (or Jupyter), you can reproduce the histogram and 3D scatter plots:

   ```python
   import torch, matplotlib.pyplot as plt
   from metrics.evaluation_metrics import cham3D

   # Calculate Chamfer distances per sample
   dl, dr, *_ = cham3D(samples.cuda(), refs.cuda())
   cd_each = (dl.mean(1) + dr.mean(1)).cpu().numpy()

   plt.figure(figsize=(6,4))
   plt.hist(cd_each, bins=20, color='steelblue', alpha=0.85)
   plt.xlabel("Chamfer-L2 (lower = better)")
   plt.ylabel("count")
   plt.title("Chamfer Distance Distribution (64 samples)")
   plt.tight_layout()
   plt.show()

   # Display one generated vs one reference 3D scatter
   idx = 0
   gen = samples[idx].numpy()
   ref = refs[idx].numpy()
   from mpl_toolkits.mplot3d import Axes3D  

   fig = plt.figure(figsize=(8,4))
   ax1 = fig.add_subplot(1,2,1, projection='3d')
   ax1.scatter(gen[:,0], gen[:,1], gen[:,2], s=1)
   ax1.set_title("Generated")
   ax1.set_axis_off()

   ax2 = fig.add_subplot(1,2,2, projection='3d')
   ax2.scatter(ref[:,0], ref[:,1], ref[:,2], s=1)
   ax2.set_title("Reference")
   ax2.set_axis_off()

   plt.tight_layout()
   plt.show()
   ```

---

## Feature Analysis & Visualization

Beyond point‐cloud generation, this repository includes tools to extract and visualize **PSPE** (Phase‐Shifted Positional Encoding) and **BAPE** (Binary Applied Positional Encoding) from the model at any diffusion timestep. You can also run an experiment to generate “cars from six canonical shapes” (cube, rect\_cuboid, sphere, pyramid, torus, plane) and compare:

1. **Modify diffusion steps for inference**

   In `train_generation.py` (or via script), set:

   ```python
   args.time_num = 200   # reduce from 1000 → 200 for faster inference
   ```

2. **Generate canonical shapes** (batch of 32 each)

   Use the provided helper functions (in a Jupyter cell or a new `.py` file):

   ```python
   from train_generation import parse_args, get_betas, Model
   import torch, numpy as np
   from train_generation import get_initial_pointcloud  # shape generators

   # 1) Load TIGER model (same steps as “Evaluation”)
   #    — parse args, set category=car, distribution_type="single"
   #    — override args.time_num = 200
   #    — load checkpoint, build model, model.eval()

   T = args.time_num  # 200
   shape_types = ['cube','rect_cuboid','sphere','pyramid','torus','plane']
   B = 32
   N = args.npoints  # 2048

   # 2) Build one combined batch (6 shapes × 32 each)
   all_init_list = []
   for s in shape_types:
       arr = np.stack([get_initial_pointcloud(s, N) for _ in range(B)], axis=0)  # (32,2048,3)
       all_init_list.append(arr)
   all_init_np = np.concatenate(all_init_list, axis=0)  # (192,2048,3)

   # 3) Convert to torch (192,3,2048)
   all_init_torch = torch.from_numpy(all_init_np).float().permute(0,2,1).to(device)

   # 4) Clear any saved_feats and run gen_samples once
   model.saved_feats = {'pspe':{t:[] for t in [0,T//2,T-1]}, 'bape':{t:[] for t in [0,T//2,T-1]}}
   with torch.no_grad():
       generated_all = model.gen_samples(all_init_torch.shape, device, custom_init=all_init_torch)

   # 5) Reshape → (6,32,3,2048), save each shape’s generated batch
   generated_all = generated_all.view(len(shape_types), B, 3, N).cpu().permute(0,1,3,2).numpy()
   for i, s in enumerate(shape_types):
       np.save(f"generated_{s}.npy", generated_all[i])  # (32,2048,3)
   ```

3. **Compute metrics vs. real cars**

   ```python
   import numpy as np
   import torch
   from metrics.evaluation_metrics import EMD_CD

   real_car_batch = np.load("real_car_batch.npy")  # pre‐saved (32,2048,3)
   real_t = torch.from_numpy(real_car_batch).float().to(device)

   metric_results = {}
   for s in shape_types:
       gen_np = np.load(f"generated_{s}.npy")         # (32,2048,3)
       gen_t = torch.from_numpy(gen_np).float().to(device)
       m = EMD_CD(gen_t, real_t, batch_size=32)
       metric_results[s] = {k: v.item() for k,v in m.items()}

   # Print a simple table:
   print(f"{'Shape':<12} {'MMD-CD':>10} {'MMD-EMD':>10} {'F-score':>10}")
   for s in shape_types:
       cd = metric_results[s]['MMD-CD']
       emd = metric_results[s]['MMD-EMD']
       fsc = metric_results[s]['F-score']
       print(f"{s:<12} {cd:10.6f} {emd:10.6f} {fsc:10.6f}")
   ```

4. **Per‐Sample Chamfer Distance Distributions**

   ```python
   import matplotlib.pyplot as plt

   per_sample_cds = {}
   for s in shape_types:
       gen_np = np.load(f"generated_{s}.npy")  # (32,2048,3)
       cds = []
       for i in range(B):
           gen_i = torch.from_numpy(gen_np[i]).float()  # CPU
           real_i = torch.from_numpy(real_car_batch[i]).float()
           diff = gen_i.unsqueeze(1) - real_i.unsqueeze(0)  # (2048,2048,3)
           dist2 = (diff**2).sum(dim=2)                    # (2048,2048)
           d1, _ = dist2.min(dim=1); d2, _ = dist2.min(dim=0)
           cd_val = d1.mean().item() + d2.mean().item()
           cds.append(cd_val)
       per_sample_cds[s] = cds

   # Box + Violin Plot
   import seaborn as sns
   sns.set(style="whitegrid", context="notebook", font_scale=1.1)

   data_list = [per_sample_cds[s] for s in shape_types]
   plt.figure(figsize=(12,5))
   sns.violinplot(data=data_list, inner=None, color="lightgray", cut=0, scale="width")
   sns.boxplot(data=data_list, whis=[5,95], width=0.3, palette="tab10", fliersize=0)
   for i, s in enumerate(shape_types):
       x_jitter = np.random.normal(loc=i, scale=0.08, size=B)
       plt.scatter(x_jitter, per_sample_cds[s], color=sns.color_palette("tab10")[i],
                   alpha=0.6, edgecolor="white", linewidth=0.5, s=30)
   plt.xticks(range(len(shape_types)), shape_types, rotation=15)
   plt.ylabel("Chamfer Distance (one-to-one)")
   plt.title("Per-Sample Chamfer Distance by Starting Shape")
   plt.tight_layout()
   plt.show()
   ```

5. **PSPE / BAPE Heatmaps**

   ```python
   import numpy as np
   import matplotlib.pyplot as plt

   timesteps = [0, T//2, T-1]
   for t in timesteps:
       pspe = model.saved_feats['pspe'][t]  # (192, embed_dim, 2048)
       bape = model.saved_feats['bape'][t]  # (192, 1, 2048)

       # Convert to NumPy if not already
       if isinstance(pspe, torch.Tensor):
           pspe = pspe.cpu().numpy()
           bape = bape.cpu().numpy()

       # One sample (index 0) vs batch average
       pspe0 = pspe[0]             # (embed_dim, 2048)
       pspe_avg = pspe.mean(axis=0)   # (embed_dim, 2048)
       bape0 = bape[0]             # (1, 2048)
       bape_avg = bape.mean(axis=0)   # (1, 2048)

       fig, axs = plt.subplots(2,2, figsize=(12,6), constrained_layout=True)
       im0 = axs[0,0].imshow(pspe0, aspect='auto', cmap='viridis')
       axs[0,0].set_title(f"PSPE (sample 0) at t={t}")
       axs[0,0].set_xlabel("Point Index"); axs[0,0].set_ylabel("Feature Dim")
       plt.colorbar(im0, ax=axs[0,0], fraction=0.045, pad=0.04)

       im1 = axs[0,1].imshow(pspe_avg, aspect='auto', cmap='viridis')
       axs[0,1].set_title(f"PSPE (batch avg) at t={t}")
       axs[0,1].set_xlabel("Point Index"); axs[0,1].set_ylabel("Feature Dim")
       plt.colorbar(im1, ax=axs[0,1], fraction=0.045, pad=0.04)

       im2 = axs[1,0].imshow(bape0, aspect='auto', cmap='plasma')
       axs[1,0].set_title(f"BAPE (sample 0) at t={t}")
       axs[1,0].set_xlabel("Point Index")
       axs[1,0].set_yticks([])
       plt.colorbar(im2, ax=axs[1,0], fraction=0.045, pad=0.04)

       im3 = axs[1,1].imshow(bape_avg, aspect='auto', cmap='plasma')
       axs[1,1].set_title(f"BAPE (batch avg) at t={t}")
       axs[1,1].set_xlabel("Point Index")
       axs[1,1].set_yticks([])
       plt.colorbar(im3, ax=axs[1,1], fraction=0.045, pad=0.04)

       plt.show()
   ```

6. **Interactive 3D Visualization (Plotly)**

   ```bash
   pip install --quiet plotly==5
   ```

   ```python
   import plotly.graph_objects as go
   import plotly.io as pio
   pio.renderers.default = "colab"  # or "notebook"

   def pc_fig(pts, title):
       fig = go.Figure(data=[
           go.Scatter3d(
               x=pts[:,0], y=pts[:,1], z=pts[:,2],
               mode="markers", marker=dict(size=2, color="orange", opacity=0.8)
           )
       ])
       fig.update_layout(
           title=title,
           margin=dict(l=0,r=0,t=30,b=0),
           scene=dict(aspectmode="data",
                      xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
       )
       return fig

   # Reference car (first sample)
   ref_pc = real_car_batch[0]  # (2048,3)
   pc_fig(ref_pc, "Reference Car (0)").show()

   # Generated from each shape (sample 0)
   for s in shape_types:
       gen_pc = np.load(f"generated_{s}.npy")[0]  # (2048,3)
       pc_fig(gen_pc, f"Generated from {s} (0)").show()
   ```

   If Plotly does not render inline, you can fallback to Matplotlib:

   ```python
   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D

   for s in shape_types:
       gen_pc = np.load(f"generated_{s}.npy")[0]
       fig = plt.figure(figsize=(4,4))
       ax = fig.add_subplot(111, projection='3d')
       ax.scatter(gen_pc[:,0], gen_pc[:,1], gen_pc[:,2], s=2, c='orange')
       ax.set_title(f"Generated from {s} (0)")
       ax.set_axis_off()
       plt.show()
   ```

---

## Dependency Summary

Below is a high‐level summary of the major libraries and versions tested:

* **Python 3.10+**
* **NumPy 1.26.4**
* **SciPy 1.11.4**
* **scikit‐learn 1.3.2**
* **PyTorch 2.3.0+cu121**
* **torchvision 0.18.0+cu121**
* **torchaudio 2.3.0+cu121**
* **PyTorch Geometric** (matching torch 2.3.1+cu121 wheels)
* **PyTorch3D** (latest commit, built from source)
* **trimesh 3.x**
* **plotly 5.x** (for interactive 3D)
* **matplotlib 3.x**
* **seaborn 0.13.x**

For most Linux/Colab setups, running:

```bash
pip install -q -r requirements_colab.txt
pip install -q ninja cmake
pip install -q "git+https://github.com/facebookresearch/pytorch3d.git"
```

should install everything needed.

---

## Contact & Citation

* **Authors / Team Members**

  1. Syed Habibul Bashar ([bashar.2102742@studenti.uniroma1.it](mailto:bashar.2102742@studenti.uniroma1.it))
  2. Arman Feili ([feili.2101835@studenti.uniroma1.it](mailto:feili.2101835@studenti.uniroma1.it))
  3. Aysegul Sine Ozgenkan ([ozgenkan.2108754@studenti.uniroma1.it](mailto:ozgenkan.2108754@studenti.uniroma1.it))

* **Corresponding Paper**
  Ren, Z., Chen, X., Li, Y., & Giannakis, G. B. (2024).
  TIGER: Time‐Varying Denoising Model for 3D Point Cloud Generation with Diffusion Process, CVPR 2024.
  PDF: [https://openaccess.thecvf.com/content/CVPR2024/papers/Ren\_TIGER\_Time-Varying\_Denoising\_Model\_for\_3D\_Point\_Cloud\_Generation\_with\_CVPR\_2024\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Ren_TIGER_Time-Varying_Denoising_Model_for_3D_Point_Cloud_Generation_with_CVPR_2024_paper.pdf)

If you use TIGER or parts of this code in your work, please cite the above CVPR 2024 paper.

---

Thank you for exploring TIGER! If you find bugs, have questions, or want to suggest improvements, please open an issue or submit a pull request.
