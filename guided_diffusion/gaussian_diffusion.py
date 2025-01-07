import math
import os
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm


from .posterior_mean_variance import get_mean_processor, get_var_processor

from .EM_onestep import EM_Initial,EM_onestep
from util.pytorch_colors import rgb_to_ycbcr, ycbcr_to_rgb
# from util.sobel import Sobel
from skimage.io import imsave
import cv2
__SAMPLER__ = {}

def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            raise NameError(f"Name {name} is already registered!") 
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_sampler(name: str):
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def create_sampler(sampler,
                   steps,
                   noise_schedule,
                   model_mean_type,
                   model_var_type,
                   dynamic_threshold,
                   clip_denoised,
                   rescale_timesteps,
                   timestep_respacing=""):

    
    sampler = get_sampler(name=sampler)
    
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
         
    return sampler(use_timesteps=space_timesteps(steps, timestep_respacing),
                   betas=betas,
                   model_mean_type=model_mean_type,
                   model_var_type=model_var_type,
                   dynamic_threshold=dynamic_threshold,
                   clip_denoised=clip_denoised, 
                   rescale_timesteps=rescale_timesteps)


class GaussianDiffusion:
    def __init__(self,
                 betas,
                 model_mean_type,
                 model_var_type,
                 dynamic_threshold,
                 clip_denoised,
                 rescale_timesteps
                 ):

        # use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (self.betas <=1).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0])
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.mean_processor = get_mean_processor(model_mean_type,
                                                 betas=betas,
                                                 dynamic_threshold=dynamic_threshold,
                                                 clip_denoised=clip_denoised)    
    
        self.var_processor = get_var_processor(model_var_type,
                                               betas=betas)



    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        
        mean = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start) * x_start
        variance = extract_and_expand(1.0 - self.alphas_cumprod, t, x_start)
        log_variance = extract_and_expand(self.log_one_minus_alphas_cumprod, t, x_start)

        return mean, variance, log_variance

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return coef1 * x_start + coef2 * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        posterior_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = extract_and_expand(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = extract_and_expand(self.posterior_log_variance_clipped, t, x_t)

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_sample_loop(self,
                      model,
                      x_start, 
                      record, 
                      I, 
                      V, 
                      save_root,
                      img_index,
                      wavalets,
                      sobel
                      ):
        """
        The function used for sampling from noise.
        """ 
        img = x_start
        device = x_start.device

        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=device)
            
            img = img 

            out = self.p_sample(x=img, t=time, model=model, infrared=I, visible=V, wavalets=wavalets, sobel=sobel)


            img = out['sample'].detach_()
           
            if record:
                if idx % 1 == 0:
                    file_path = os.path.join(save_root, 'progress', str(img_index))
                    os.makedirs(file_path) if not os.path.exists(file_path) else file_path

                    temp_img= img.detach().cpu().squeeze().numpy()
                    temp_img=np.transpose(temp_img, (1,2,0))
                    temp_img=cv2.cvtColor(temp_img,cv2.COLOR_RGB2YCrCb)[:,:,0]
                    temp_img=(temp_img-np.min(temp_img))/(np.max(temp_img)-np.min(temp_img))
                    temp_img=((temp_img)*255).astype('uint8')
                    imsave(os.path.join(file_path, "{}.png".format(f"x_{str(idx).zfill(4)}")),temp_img)
 
        return img
        
    def p_sample(self, model, x, t):
        raise NotImplementedError

    def p_mean_variance(self, model, x, t):

        model_output = model(x, self._scale_timesteps(t))
        
        # In the case of "learned" variance, model will give twice channels.
        if model_output.shape[1] == 2 * x.shape[1]:
            model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        else:
            # The name of variable is wrong. 
            # This will just provide shape information, and 
            # will not be used for calculating something important in variance.
            model_var_values = model_output

        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_variance, model_log_variance = self.var_processor.get_variance(model_var_values, t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {'mean': model_mean,
                'variance': model_variance,
                'log_variance': model_log_variance,
                'pred_xstart': pred_xstart}

    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


@register_sampler(name='ddpm')
class DDPM(SpacedDiffusion):
    def p_sample(self, model, x, t):
        out = self.p_mean_variance(model, x, t)
        # out = return {'mean': model_mean,
        #               'variance': model_variance,
        #               'log_variance': model_log_variance,
        #               'pred_xstart': pred_xstart}
        sample = out['mean']

        noise = torch.randn_like(x)
        if t != 0:  # no noise when t == 0
            sample += torch.exp(0.5 * out['log_variance']) * noise

        return {'sample': sample, 'pred_xstart': out['pred_xstart']}
    

@register_sampler(name='ddim')
class DDIM(SpacedDiffusion):
    # def p_sample(self, model, x, t, bfHP, infrared, visible, lamb,rho,eta=0.0):
    #
    #     out = self.p_mean_variance(model, x, t)
    #
    #
    #
    #
    #     # x_0_hat_ycbcr = rgb_to_ycbcr(out['pred_xstart'])/255 # (-1,1)
    #     # x_0_hat_y = torch.unsqueeze((x_0_hat_ycbcr[:,0,:,:]),1)
    #     # assert x_0_hat_y.shape[1]==1
    #     #
    #     # x_0_hat_y_BF, bfHP = EM_onestep(f_pre = x_0_hat_y,
    #     #                                     I = infrared,
    #     #                                     V = visible,
    #     #                                     HyperP = bfHP,lamb=lamb,rho=rho)
    #     #
    #     # x_0_hat_ycbcr[:,0,:,:] = x_0_hat_y_BF
    #     # out['pred_xstart'] = ycbcr_to_rgb(x_0_hat_ycbcr*255)
    #
    #
    #     eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])
    #
    #
    #     alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
    #     alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
    #     sigma = (
    #         eta
    #         * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
    #         * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
    #     )
    #
    #     noise = torch.randn_like(x)
    #
    #
    #     mean_pred = (
    #         out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
    #         + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
    #     )
    #
    #     sample = mean_pred
    #     if t != 0:
    #         sample += sigma * noise
    #
    #     return {"sample": sample, "pred_xstart": out["pred_xstart"]}, bfHP

    def p_sample(self, model, x, t, infrared, visible, wavalets, sobel, eta=0.0):
        # import pdb;pdb.set_trace()
        # sigma = t /20
        #


        out = self.p_mean_variance(model, x, t)
        lambda_t = 1
        if t < 10:
            lambda_t = (t/10)**2
            # lambda_t = 0
        inf_yl, inf_yh = wavalets.forward(infrared)
        vis_yl, vis_yh = wavalets.forward(visible)
        x_start_yl, x_start_yh = wavalets.forward((out['pred_xstart']))
        new_yh = vis_yh
        new_yl = (inf_yl + vis_yl) / 2
        for i in range(len(new_yh)):
            new_yh[i] = torch.where(torch.abs(inf_yh[i]) > torch.abs(vis_yh[i]), inf_yh[i], vis_yh[i])
            x_start_yh[i] = x_start_yh[i] - new_yh[i]
        x_start_yl = x_start_yl - new_yl
        new_x_start = wavalets.inverse(x_start_yl, x_start_yh)
        # new_x_start = (new_x_start - torch.min(new_x_start)) / (torch.max(new_x_start) - torch.min(new_x_start)) * 2 - 1

        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)

        new_target_start = wavalets.inverse(new_yl, new_yh)

        new_sobel = self.sobel_func(out['pred_xstart'], infrared, visible, sobel, sobel_eta=0.05)
        new_x_start = out['pred_xstart'] - lambda_t * new_x_start - new_sobel
        # new_x_start = out['pred_xstart'] - torch.sqrt(alpha_bar) * (lambda_t * new_x_start + new_sobel)
        # new_x_start = (new_x_start +　self.p_sample_hr(out,  x, new_x_start, t)) / 2
        # new_x_start = (new_x_start + self.p_sample_hr(out, x, new_x_start, t))/ 2
        new_hr_v, new_hr_start_v = self.p_sample_hr(out, x, visible, t)
        new_hr_i, new_hr_start_i = self.p_sample_hr(out, x, infrared, t)

        # new_x_start =  new_x_start - lambda_t * new_x_start   -  new_sobel*sobel_eta

        new_x_start = 0.9 * new_x_start + 0.05 * new_hr_start_i + 0.05 * new_hr_start_v
        # out['pred_xstart'] = 0.99 * new_x_start +  0.1 *alpha_bar * new_hr_start
        # out['pred_xstart'] = (new_x_start + new_hr_start)/2
        # out['pred_xstart']  = (1 - torch.sqrt(alpha_bar)) * new_x_start + torch.sqrt(alpha_bar) * new_hr_start
        out['pred_xstart'] = new_x_start
        # 根据计算公式
        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(x)

        mean_pred = (
                out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) *eps
        )

        # sample = (1-torch.sqrt(alpha_bar)) * mean_pred +  torch.sqrt(alpha_bar)* new_hr
        # sample = (mean_pred + new_hr)/2
        sample = mean_pred
        if t != 0:
            sample += sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def sobel_func(self, x, infrared, visible, sobel, sobel_eta=0.05):
        new_sobel = sobel.sobel(x) - torch.max(sobel.sobel(infrared), sobel.sobel(visible))
        new_sobel = torch.cat([new_sobel, new_sobel, new_sobel], axis=1)

        return new_sobel * sobel_eta


    def p_sample_hr(self, out, x, target, t, eta=0.0):

        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
        # new_x_start = out['pred_xstart']

        x0_t = out['pred_xstart']

        # scale = round(2)

        # A = torch.nn.AdaptiveAvgPool2d((x.shape[-2] // scale, x.shape[-1] // scale))
        # y = A(target)
        # Ap = lambda z: MeanUpsample(z, scale)


        A = torch.nn.AdaptiveAvgPool2d((int(x.shape[-2] * 0.8), int(x.shape[-1] // 0.8)))
        y = A(target)
        # import pdb;pdb.set_trace()
        # new_x_start = x0_t - Ap(A(x0_t) - y)
        new_x_start = x0_t - torch.nn.functional.interpolate((A(x0_t) - y), size=(x.shape[-2:]), scale_factor=None,
                                                         mode='nearest',
                                                         align_corners=None)

        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        mean_pred = (
                new_x_start * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * (1 - alpha_bar_prev ** 2).sqrt() * eps
            # + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) *eps
        )
        return mean_pred, new_x_start

    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2


# =================
# Helper functions
# =================
def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    # out = torch.nn.functional.interpolate(x, size=tar_shape, scale_factor=None, mode='nearest', align_corners=None)
    return out
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

# ================
# Helper function
# ================

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.float):
        array = torch.tensor([array])
   
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def compute_alpha(beta, t):
    # import pdb;pdb.set_trace()
    beta = torch.cat([torch.zeros(1).to(t.device), torch.from_numpy(beta).to(t.device)], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

