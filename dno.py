import math
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DNOOptions:
    num_opt_steps: int = field(
        default=500,
        metadata={
            "help": "Number of optimization steps (300 for editing, 500 for refinement, can go further for better results)"
        },
    )
    lr: float = field(default=5e-2, metadata={"help": "Learning rate"})
    perturb_scale: float = field(
        default=0, metadata={"help": "scale of the noise perturbation"}
    )
    diff_penalty_scale: float = field(
        default=0,
        metadata={
            "help": "penalty for the difference between the final z and the initial z"
        },
    )
    lr_warm_up_steps: int = field(
        default=50, metadata={"help": "Number of warm-up steps for the learning rate"}
    )
    lr_decay_steps: int = field(
        default=None,
        metadata={"help": "Number of decay steps (if None, then set to num_opt_steps)"},
    )
    decorrelate_scale: float = field(
        default=1000, metadata={"help": "penalty for the decorrelation of the noise"}
    )
    decorrelate_dim: int = field(
        default=3,
        metadata={
            "help": "dimension to decorrelate (we usually decorrelate time dimension)"
        },
    )
    
    # DPoser 相关参数
    dposer_penalty_scale: float = field(
        default=0.0, 
        metadata={"help": "DPoser regularization weight (0 to disable)"}
    )
    dposer_timestep_strategy: str = field(
        default='truncated',
        metadata={"help": "Timestep strategy: 'random', 'fixed', 'truncated'"}
    )
    dposer_t_max: float = field(default=0.15, metadata={"help": "Max timestep for truncated strategy"})
    dposer_t_min: float = field(default=0.05, metadata={"help": "Min timestep for truncated strategy"})
    dposer_t_fixed: float = field(default=0.1, metadata={"help": "Fixed timestep value"})
    dposer_use_snr_weighting: bool = field(default=True, metadata={"help": "Use SNR-based weighting"})
    dposer_num_timesteps: int = field(default=1000, metadata={"help": "Number of diffusion timesteps"})

    def __post_init__(self):
        # if lr_decay_steps is not set, then set it to num_opt_steps
        if self.lr_decay_steps is None:
            self.lr_decay_steps = self.num_opt_steps


class DNO:
    """
    Args:
        start_z: (N, 263, 1, 120)
    """

    def __init__(
        self,
        model,
        criterion,
        start_z,
        conf: DNOOptions,
        diffusion=None,  # 传入diffusion对象用于更准确的DPoser
    ):
        self.model = model
        self.criterion = criterion
        self.diffusion = diffusion
        # for diff penalty
        self.start_z = start_z.detach()
        self.conf = conf

        self.current_z = self.start_z.clone().requires_grad_(True)
        # excluding the first dimension (batch size)
        self.dims = list(range(1, len(self.start_z.shape)))

        self.optimizer = torch.optim.Adam([self.current_z], lr=conf.lr)

        self.lr_scheduler = []
        if conf.lr_warm_up_steps > 0:
            self.lr_scheduler.append(
                lambda step: warmup_scheduler(step, conf.lr_warm_up_steps)
            )
        self.lr_scheduler.append(
            lambda step: cosine_decay_scheduler(
                step, conf.lr_decay_steps, conf.num_opt_steps, decay_first=False
            )
        )

        self.step_count = 0
        self.hist = []
        
        # DPoser设置
        self.use_dposer = conf.dposer_penalty_scale > 0
        if self.use_dposer:
            print(f"DPoser regularization enabled with weight {conf.dposer_penalty_scale}")
            self.dposer_loss_fn = torch.nn.MSELoss(reduction='none')
            self._setup_dposer_schedule()

    def _setup_dposer_schedule(self):
        """设置更准确的DPoser噪声调度（类似原版）"""
        if self.diffusion is not None:
            # 使用diffusion对象的调度参数
            self.sqrt_alphas_cumprod = torch.from_numpy(self.diffusion.sqrt_alphas_cumprod).float()
            self.sqrt_one_minus_alphas_cumprod = torch.from_numpy(self.diffusion.sqrt_one_minus_alphas_cumprod).float()
            self.alphas_cumprod = torch.from_numpy(self.diffusion.alphas_cumprod).float()
        else:
            # 使用cosine调度作为fallback
            print("Warning: No diffusion object provided, using cosine schedule for DPoser")
            betas = self._cosine_beta_schedule(self.conf.dposer_num_timesteps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
            self.alphas_cumprod = alphas_cumprod

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine beta schedule (类似原版DPoser使用的调度)"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _get_noise_schedule_params(self, t_val):
        """获取噪声调度参数（更接近原版）"""
        device = self.current_z.device
        
        # 将连续时间转换为离散时间步
        if hasattr(self, 'sqrt_alphas_cumprod'):
            timestep_idx = int(t_val * (len(self.sqrt_alphas_cumprod) - 1))
            timestep_idx = max(0, min(timestep_idx, len(self.sqrt_alphas_cumprod) - 1))
            
            sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timestep_idx].to(device)
            sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timestep_idx].to(device)
            alpha_cumprod = self.alphas_cumprod[timestep_idx].to(device)
        else:
            # Fallback to simple schedule
            sqrt_alpha_cumprod = torch.sqrt(torch.tensor(1.0 - t_val, device=device))
            sqrt_one_minus_alpha_cumprod = torch.sqrt(torch.tensor(t_val, device=device))
            alpha_cumprod = torch.tensor(1.0 - t_val, device=device)
        
        return sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod, alpha_cumprod

    def compute_dposer_loss(self, x_0, step):
        """
        改进的DPoser正则化损失计算（更接近原版）
        Args:
            x_0: 当前解码得到的运动 [batch_size, 263, 1, nframes] 
            step: 当前优化步骤
        Returns:
            dposer_loss: DPoser正则化损失 [batch_size,]
        """
        if not self.use_dposer:
            return torch.zeros(x_0.shape[0], device=x_0.device)
        
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # 选择时间步（与原版相同的策略）
        if self.conf.dposer_timestep_strategy == 'random':
            t_val = torch.rand(1).item() * (self.conf.dposer_t_max - self.conf.dposer_t_min) + self.conf.dposer_t_min
        elif self.conf.dposer_timestep_strategy == 'fixed':
            t_val = self.conf.dposer_t_fixed
        elif self.conf.dposer_timestep_strategy == 'truncated':
            # 与原版完全相同的truncated策略
            progress = (self.conf.num_opt_steps - step - 1) / self.conf.num_opt_steps
            t_val = self.conf.dposer_t_min + progress * (self.conf.dposer_t_max - self.conf.dposer_t_min)
        else:
            raise ValueError(f"Unknown timestep strategy: {self.conf.dposer_timestep_strategy}")
        
        # 获取精确的噪声调度参数
        sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod, alpha_cumprod = self._get_noise_schedule_params(t_val)
        
        # 添加噪声（使用准确的扩散过程）
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        
        # 去噪估计 (stop gradient)
        with torch.no_grad():
            try:
                # 这里仍然使用完整solver，这是你的方法的优势
                x_0_hat = self.model(x_t)
            except Exception as e:
                print(f"Warning: DPoser computation failed: {e}")
                return torch.zeros(batch_size, device=device)
        
        # 计算基础损失
        base_loss = self.dposer_loss_fn(x_0, x_0_hat)  # [batch_size, 263, 1, nframes]
        
        # 应用SNR权重（类似原版）
        if self.conf.dposer_use_snr_weighting:
            # 计算SNR: α_t / σ_t
            sigma = sqrt_one_minus_alpha_cumprod
            alpha = sqrt_alpha_cumprod
            SNR = alpha / sigma if sigma > 1e-8 else torch.tensor(1000.0, device=device)
            
            # 原版权重公式：0.5 * sqrt(1 + SNR)
            weight = 0.5 * torch.sqrt(1 + SNR)
            weighted_loss = weight * base_loss
        else:
            weighted_loss = 0.5 * base_loss  # 原版默认权重
        
        # 对除batch维度外的所有维度求平均
        dposer_loss = torch.mean(weighted_loss, dim=[1, 2, 3])
        
        return dposer_loss

    def __call__(self, num_steps: int = None):
        if num_steps is None:
            num_steps = self.conf.num_opt_steps

        batch_size = self.start_z.shape[0]
        with tqdm(range(num_steps)) as prog:
            for i in prog:
                info = {"step": [self.step_count] * batch_size}

                # learning rate scheduler
                lr_frac = 1
                if len(self.lr_scheduler) > 0:
                    for scheduler in self.lr_scheduler:
                        lr_frac *= scheduler(self.step_count)
                    self.set_lr(self.conf.lr * lr_frac)
                info["lr"] = [self.conf.lr * lr_frac] * batch_size

                # criterion
                x = self.model(self.current_z)
                # [batch_size,]
                loss = self.criterion(x)
                assert loss.shape == (batch_size,)
                info["loss"] = loss.detach().cpu()
                total_loss = loss.sum()

                # DPoser regularization
                if self.conf.dposer_penalty_scale > 0:
                    loss_dposer = self.compute_dposer_loss(x, self.step_count)
                    assert loss_dposer.shape == (batch_size,)
                    total_loss += self.conf.dposer_penalty_scale * loss_dposer.sum()
                    info["loss_dposer"] = loss_dposer.detach().cpu()
                else:
                    info["loss_dposer"] = torch.zeros(batch_size)

                # diff penalty
                if self.conf.diff_penalty_scale > 0:
                    # [batch_size,]
                    loss_diff = (self.current_z - self.start_z).norm(p=2, dim=self.dims)
                    assert loss_diff.shape == (batch_size,)
                    total_loss += self.conf.diff_penalty_scale * loss_diff.sum()
                    info["loss_diff"] = loss_diff.detach().cpu()
                else:
                    info["loss_diff"] = torch.zeros(batch_size)

                # decorrelate
                if self.conf.decorrelate_scale > 0:
                    loss_decorrelate = noise_regularize_1d(
                        self.current_z,
                        dim=self.conf.decorrelate_dim,
                    )
                    assert loss_decorrelate.shape == (batch_size,)
                    total_loss += self.conf.decorrelate_scale * loss_decorrelate.sum()
                    info["loss_decorrelate"] = loss_decorrelate.detach().cpu()
                else:
                    info["loss_decorrelate"] = torch.zeros(batch_size)

                # backward
                self.optimizer.zero_grad()
                total_loss.backward()

                # log grad norm (before)
                info["grad_norm"] = (
                    self.current_z.grad.norm(p=2, dim=self.dims).detach().cpu()
                )

                # grad mode
                self.current_z.grad.data /= self.current_z.grad.norm(
                    p=2, dim=self.dims, keepdim=True
                )

                # optimize z
                self.optimizer.step()

                # noise perturbation
                # match the noise fraction to the learning rate fraction
                noise_frac = lr_frac
                info["perturb_scale"] = [
                    self.conf.perturb_scale * noise_frac
                ] * batch_size

                noise = torch.randn_like(self.current_z)
                self.current_z.data += noise * self.conf.perturb_scale * noise_frac

                # log the norm(z - start_z)
                info["diff_norm"] = (
                    (self.current_z - self.start_z)
                    .norm(p=2, dim=self.dims)
                    .detach()
                    .cpu()
                )

                # log current z
                info["z"] = self.current_z.detach().cpu()
                info["x"] = x.detach().cpu()

                self.step_count += 1
                self.hist.append(info)
                
                # Update progress bar
                postfix_dict = {"loss": info["loss"].mean().item()}
                if self.conf.dposer_penalty_scale > 0:
                    dposer_val = info["loss_dposer"].mean().item()
                    postfix_dict["dposer"] = dposer_val
                prog.set_postfix(postfix_dict)

            # output is a list (over batch) of dict (over keys) of lists (over steps)
            hist = []
            for i in range(batch_size):
                hist.append({})
                for k in self.hist[0].keys():
                    hist[-1][k] = [info[k][i] for info in self.hist]
            return {
                # last step's z
                "z": self.current_z.detach(),
                # previous steps' x
                "x": x.detach(),
                "hist": hist,
            }

    def set_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr


def warmup_scheduler(step, warmup_steps):
    if step < warmup_steps:
        return step / warmup_steps
    return 1


def cosine_decay_scheduler(step, decay_steps, total_steps, decay_first=True):
    # decay the last "decay_steps" steps from 1 to 0 using cosine decay
    # if decay_first is True, then the first "decay_steps" steps will be decayed from 1 to 0
    # if decay_first is False, then the last "decay_steps" steps will be decayed from 1 to 0
    if step >= total_steps:
        return 0
    if decay_first:
        if step >= decay_steps:
            return 0
        return (math.cos((step) / decay_steps * math.pi) + 1) / 2
    else:
        if step < total_steps - decay_steps:
            return 1
        return (
            math.cos((step - (total_steps - decay_steps)) / decay_steps * math.pi) + 1
        ) / 2


def noise_regularize_1d(noise, stop_at=2, dim=3):
    """
    Args:
        noise (torch.Tensor): (N, C, 1, size)
        stop_at (int): stop decorrelating when size is less than or equal to stop_at
        dim (int): the dimension to decorrelate
    """
    all_dims = set(range(len(noise.shape)))
    loss = 0
    size = noise.shape[dim]

    # pad noise in the size dimention so that it is the power of 2
    if size != 2 ** int(math.log2(size)):
        new_size = 2 ** int(math.log2(size) + 1)
        pad = new_size - size
        pad_shape = list(noise.shape)
        pad_shape[dim] = pad
        pad_noise = torch.randn(*pad_shape).to(noise.device)

        noise = torch.cat([noise, pad_noise], dim=dim)
        size = noise.shape[dim]

    while True:
        # this loss penalizes spatially correlated noise
        # the noise is rolled in the size direction and the dot product is taken
        # (bs, )
        loss = loss + (noise * torch.roll(noise, shifts=1, dims=dim)).mean(
            # average over all dimensions except 0 (batch)
            dim=list(all_dims - {0})
        ).pow(2)

        # stop when size is 8
        if size <= stop_at:
            break

        # (N, C, 1, size) -> (N, C, 1, size // 2, 2)
        noise_shape = list(noise.shape)
        noise_shape[dim] = size // 2
        noise_shape.insert(dim + 1, 2)
        noise = noise.reshape(noise_shape)
        # average pool over (2,) window
        noise = noise.mean([dim + 1])
        size //= 2

    return loss