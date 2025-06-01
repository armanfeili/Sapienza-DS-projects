import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from model.pvcnn_generation import PVCNN2Base
from model.tiger import Tiger_Transformer
import torch.distributed as dist
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from tqdm import trange  # add this import at the top of the file


'''
some utils
'''
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate(vertices, faces):
    '''
    vertices: [numpoints, 3]
    '''
    M = rotation_matrix([0, 1, 0], np.pi / 2).transpose()
    N = rotation_matrix([1, 0, 0], -np.pi / 4).transpose()
    K = rotation_matrix([0, 0, 1], np.pi).transpose()

    v, f = vertices[:,[1,2,0]].dot(M).dot(N).dot(K), faces[:,[1,2,0]]
    return v, f

def norm(v, f):
    v = (v - v.min())/(v.max() - v.min()) - 0.5

    return v, f

def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
    return pNorm, gradNorm


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)

'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs

class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool):

        model_output = denoise_fn(data, t)


        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, denoise_fn, shape, device,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq,
                                 noise_fn=torch.randn,clip_denoised=True, keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps =  self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0,total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised,
                                  return_pred_xstart=False)
            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    '''losses'''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=data_start, x_t=data_t, t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data_t, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(data_start.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, t, noise=None):
        """
        Training loss calculation
        """
        B, D, N = data_start.shape
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(data_start.shape, dtype=data_start.dtype, device=data_start.device)
        assert noise.shape == data_start.shape and noise.dtype == data_start.dtype

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type == 'mse':
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            eps_recon = denoise_fn(data_t, t)
            assert data_t.shape == data_start.shape
            assert eps_recon.shape == torch.Size([B, D, N])
            assert eps_recon.shape == data_start.shape
            losses = ((noise - eps_recon)**2).mean(dim=list(range(1, len(data_start.shape))))
        elif self.loss_type == 'kl':
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, clip_denoised=False,
                return_pred_xstart=False)
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses

    '''debug'''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T-1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_= torch.zeros([B, T], device=x_start.device), torch.zeros([B, T], device=x_start.device)
            for t in reversed(range(T)):

                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn, data_start=x_start, data_t=self.q_sample(x_start=x_start, t=t_b), t=t_b,
                    clip_denoised=clip_denoised, return_pred_xstart=True)
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start.shape
                new_mse_b = ((pred_xstart-x_start)**2).mean(dim=list(range(1, len(x_start.shape))))
                assert new_vals_b.shape == new_mse_b.shape ==  torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None]==torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start)
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size([B, T]) and \
                   total_bpd_b.shape == prior_bpd_b.shape ==  torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()


class Tiger_Transformer_custom(Tiger_Transformer):
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
        dropout,
        extra_feature_channels=3,
        width_multiplier=1,
        voxel_resolution_multiplier=1
    ):
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
    # ────────────────────────────────────────────────────────────────────
    def get_pspe_feats(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute and return the raw PSPE features for input x_t at diffusion timestep t.

        Args:
          - x_t: Tensor of shape (B, 3, N), the noisy point cloud at timestep t.
          - t:   Tensor of shape (B,), containing the integer timestep index for each sample.

        Returns:
          - pspe: Tensor of shape (B, embed_dim, N)
        """
        # Simply forward to the parent class’s implementation of PSPE.
        return super().get_pspe_feats(x_t, t)
    # ────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────
    def get_bape_feats(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute and return the raw BAPE features for input x_t at timestep t.

        Args:
          - x_t: Tensor of shape (B, 3, N), the noisy point cloud at timestep t.
          - t:   Tensor of shape (B,), containing the integer timestep index for each sample.

        Returns:
          - bape: Tensor of shape (B, 1, N) (or (B, embed_dim, N) if BAPE uses that dimension)
        """
        # Simply forward to the parent class’s implementation of BAPE.
        return super().get_bape_feats(x_t, t)
    # ────────────────────────────────────────────────────────────────────


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)
        self.model     = Tiger_Transformer_custom(
                            num_classes=args.nc,
                            embed_dim=args.embed_dim,
                            use_att=args.attention,
                            dropout=args.dropout,
                            extra_feature_channels=0
                         )

        # ───────────────────────────────────────────────────────────────
        # NEW: store which timesteps should have PSPE/BAPE features saved
        # Let T = total number of diffusion timesteps (args.time_num)
        T = args.time_num
        # We want to capture features at t = 0, t = T//2, and t = T-1
        self.timesteps_to_save = [0, T // 2, T - 1]

        # Prepare a dict to hold “saved_feats” for both PSPE and BAPE.
        # Each key is a timestep; we will append feature‐tensors there during sampling.
        self.saved_feats = {
            'pspe': {t: [] for t in self.timesteps_to_save},
            'bape': {t: [] for t in self.timesteps_to_save},
        }

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t):
        B, D,N= data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, shape, device, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False,
                    custom_init=None):
        """
        Generate final x0 point cloud, starting either from pure noise or the provided custom_init.
        Args:
          - shape: tuple (B, 3, N) indicating batch size, channels, num_points
          - device: torch.device
          - custom_init: if not None, a tensor of shape (B,3,N) to use as x_T
          - noise_fn: function, default=torch.randn
          - clip_denoised: whether to clamp predicted x0 between [-0.5,0.5]
          - keep_running: unused for standard sampling (False)

        Returns:
          - x_t at t=0, i.e. the generated point cloud of shape (B,3,N)
        """
        # Number of diffusion timesteps
        T = self.diffusion.num_timesteps

        B, C, N = shape

        # 1) If custom_init is provided, use that as x_T; otherwise sample Gaussian noise.
        if custom_init is not None:
            assert custom_init.shape == torch.Size([B, C, N]), \
                f"custom_init must be shape {(B, C, N)}, got {tuple(custom_init.shape)}"
            x_t = custom_init.clone().to(device)
        else:
            x_t = noise_fn(size=shape, dtype=torch.float, device=device)

        # Pre‐allocate a single timestep tensor of shape (B,) on the correct device
        t = torch.empty((B,), dtype=torch.int64, device=device)

        # 2) Run reverse diffusion from t = T-1 down to t = 0, showing a tqdm bar
        for t_int in trange(T - 1, -1, -1, desc="Reverse diffusion", unit="step"):
            # Fill the t‐tensor with the current timestep index
            t.fill_(t_int)

            # If this timestep is one we want to save features at, do so now:
            if t_int in self.timesteps_to_save:
                pspe_feat = self.model.get_pspe_feats(x_t, t)   # (B, embed_dim, N)
                bape_feat = self.model.get_bape_feats(x_t, t)   # (B, 1, N) or (B, embed_dim, N)
                self.saved_feats['pspe'][t_int].append(pspe_feat.detach().cpu())
                self.saved_feats['bape'][t_int].append(bape_feat.detach().cpu())

            # Single reverse‐denoising step at timestep t_int
            x_t = self.diffusion.p_sample(
                denoise_fn=self._denoise,
                data=x_t,
                t=t,
                noise_fn=noise_fn,
                clip_denoised=clip_denoised,
                return_pred_xstart=False
            )

        # After looping from T-1 to 0, x_t is now the final x0
        return x_t

    def gen_sample_traj(self, shape, device, freq, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas


def get_dataset(dataroot, npoints,category):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return tr_dataset, te_dataset


def get_dataloader(opt, train_dataset, test_dataset=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if test_dataset is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            test_sampler = None
    else:
        train_sampler = None
        test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True)

    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=test_sampler,
                                                   shuffle=False, num_workers=int(opt.workers), drop_last=False)
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler


def train(gpu, opt, output_dir, noises_init):
    # ---------------------------------------------------------------- init
    set_seed(opt)
    logger       = setup_logging(output_dir)
    should_diag  = (opt.distribution_type != 'multi') or (gpu == 0)
    if should_diag:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')

    # ---------- distributed boiler-plate ---------------------------------
    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        opt.ngpus_per_node = torch.cuda.device_count()
        base_rank          = opt.rank * opt.ngpus_per_node
        opt.rank           = base_rank + gpu

        dist.init_process_group(backend      = opt.dist_backend,
                                init_method  = opt.dist_url,
                                world_size   = opt.world_size,
                                rank         = opt.rank)

        torch.cuda.set_device(gpu)
        device = torch.device(f'cuda:{gpu}')
        # adjust per-process batch-size / workers
        opt.bs      = int(opt.bs // opt.ngpus_per_node)
        opt.workers = 0
    else:                                   # single-GPU / CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ---------------------------------------------------------------------

    # ----------------------- data / model / optimizer  -------------------
    train_set, _ = get_dataset(opt.dataroot, opt.npoints, opt.category)
    train_loader, _, train_sampler, _ = get_dataloader(opt, train_set)

    betas  = get_betas(opt.schedule_type,
                       opt.beta_start, opt.beta_end, opt.time_num)
    model  = Model(opt, betas,
                   opt.loss_type, opt.model_mean_type, opt.model_var_type)
    model  = model.to(device)

    if opt.distribution_type == 'multi':
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[gpu],
                                                    output_device=gpu)
    elif opt.distribution_type == 'single' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(),
                           lr = opt.lr,
                           betas = (opt.beta1, 0.999),
                           weight_decay = opt.decay)

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)

    if opt.model:                            # resume
        ckpt        = torch.load(opt.model, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
    else:
        start_epoch = 0
    # single place to reference the right device later on
    device = next(model.parameters()).device
    # ---------------------------------------------------------------------

    #######################################################################
    #  TRAINING LOOP                                                      #
    #######################################################################
    for epoch in range(start_epoch, int(opt.niter)):

        if opt.distribution_type == 'multi':
            train_sampler.set_epoch(epoch)

        model.train()
        for i, batch in enumerate(train_loader):
            x           = batch['train_points'].transpose(1, 2).to(device)
            noise_batch = noises_init[batch['idx']].transpose(1, 2).to(device)

            loss = model.get_loss_iter(x, noise_batch).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

            if i % int(opt.print_freq) == 0 and should_diag:
                net_p, net_g = getGradNorm(model)
                logger.info(
                    f'[{epoch:3d}/{opt.niter:3d}]'
                    f'[{i:3d}/{len(train_loader):3d}]  '
                    f'loss={loss.item():8.4f}  '
                    f'|‖W‖={net_p:8.2f}  '
                    f'|‖∇W‖={net_g:8.2f}'
                )

        lr_scheduler.step()  # one decay step per epoch

        # ---------------- diagnostics / visualisation -------------------
        if should_diag and (epoch + 1) % int(opt.diagIter) == 0:
            ...
        if should_diag and (epoch + 1) % int(opt.vizIter) == 0:
            ...

        # ---------------------- checkpointing --------------------------
        if (epoch + 1) % int(opt.saveIter) == 0 and should_diag:
            ckpt_path = f'{output_dir}/epoch_{epoch}.pth'
            torch.save(
                {'epoch': epoch,
                 'model_state': model.state_dict(),
                 'optimizer_state': optimizer.state_dict()},
                ckpt_path
            )
            logger.info(f'Checkpoint saved →  {ckpt_path}')
            if opt.distribution_type == 'multi':
                dist.barrier()

    # ---------------------- clean-up (multi-GPU only) -------------------
    if opt.distribution_type == 'multi' and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def main():
    opt = parse_args()
    if opt.category == 'airplane':
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)
    output_dir = get_output_dir(dir_id, exp_id)
    copy_source(__file__, output_dir)

    ''' workaround '''
    train_dataset, _ = get_dataset(opt.dataroot, opt.npoints, opt.category)
    noises_init = torch.randn(len(train_dataset), opt.npoints, opt.nc)

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir, noises_init))
    else:
        train(opt.gpu, opt, output_dir, noises_init)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',       type=str,   default='ShapeNetCore.v2.PC15k/')
    parser.add_argument('--category',       type=str,   default='chair')

    parser.add_argument('--bs',             type=int,   default=64,    help='input batch size')
    parser.add_argument('--workers',        type=int,   default=16,    help='number of dataloader workers')
    parser.add_argument('--niter',          type=int,   default=10000, help='number of epochs')

    parser.add_argument('--nc',             type=int,   default=3,     help='number of channels (3)')
    parser.add_argument('--npoints',        type=int,   default=2048,  help='points per shape')
    # diffusion schedule
    parser.add_argument('--beta_start',     type=float, default=1e-4)
    parser.add_argument('--beta_end',       type=float, default=2e-2)
    parser.add_argument('--schedule_type',  type=str,   default='linear')
    parser.add_argument('--time_num',       type=int,   default=1000)

    # model hyperparameters
    parser.add_argument('--attention',      action='store_true', help='enable attention branch')
    parser.add_argument('--dropout',        type=float, default=0.1)
    parser.add_argument('--embed_dim',      type=int,   default=64)
    parser.add_argument('--loss_type',      type=str,   default='mse', choices=['mse','kl'])
    parser.add_argument('--model_mean_type',type=str,   default='eps')
    parser.add_argument('--model_var_type', type=str,   default='fixedsmall')

    # optimizer / scheduler
    parser.add_argument('--lr',             type=float, default=2e-4)
    parser.add_argument('--beta1',          type=float, default=0.5)
    parser.add_argument('--decay',          type=float, default=0.0)
    parser.add_argument('--grad_clip',      type=float, default=None)
    parser.add_argument('--lr_gamma',       type=float, default=0.998)

    parser.add_argument('--model',          type=str,   default='', help='checkpoint to resume')

    # distributed (mostly unused in Colab)
    parser.add_argument('--world_size',     type=int,   default=1)
    parser.add_argument('--dist_url',       type=str,   default='tcp://127.0.0.1:9991')
    parser.add_argument('--dist_backend',   type=str,   default='nccl')
    parser.add_argument('--distribution_type', type=str, default='single', choices=['single','multi'])
    parser.add_argument('--rank',           type=int,   default=0)
    parser.add_argument('--gpu',            type=int,   default=None, help='GPU id or None')

    # logging / checkpoint
    parser.add_argument('--saveIter',       type=int,   default=100,  help='epochs per checkpoint')
    parser.add_argument('--diagIter',       type=int,   default=50,   help='epochs per diag')
    parser.add_argument('--vizIter',        type=int,   default=50,   help='epochs per viz')
    parser.add_argument('--print_freq',     type=int,   default=50,   help='iters per log')
    parser.add_argument('--manualSeed',     type=int,   default=42)

    return parser.parse_args()


if __name__ == '__main__':
    main()
