import streamlit as st
import torch
import functools
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

st.text("Version : 1.0 ")

def marginal_prob_std(t, sigma, device):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

    Returns:
        The standard deviation.
    """
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / np.log(sigma))


def diffusion_coeff(t, sigma, device):
    """Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

    Returns:
        The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
        marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


## From Yang Song notebook
## The number of sampling steps.
def Euler_Maruyama_sampler(
    score_model,
    marginal_prob_std,
    diffusion_coeff,
    device,
    batch_size=64,
    num_steps=500,
    eps=1e-5,
):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
    the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
    Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns:
    Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = (
        torch.randn(batch_size, 1, 28, 28, device=device)
        * marginal_prob_std(t)[:, None, None, None]
    )
    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = (
                x
                + (g**2)[:, None, None, None]
                * score_model(x, batch_time_step)
                * step_size
            )
            x = mean_x + torch.sqrt(step_size) * g[
                :, None, None, None
            ] * torch.randn_like(x)
    # Do not include any noise in the last sampling step.
    return mean_x


st.header("Generate images")

td = "GPU"

#td = st.radio(
#    "Training device",
#    ('CPU', 'GPU' ))

sigma_emnist = 21500
sigma_kmnist = 20200
sigma_hasy =  17000
#if td == 'CPU':
#    device = "cpu"
#    marginal_prob_std_fncpu = functools.partial(marginal_prob_std, sigma=sigma_emnist, device=device)
#    diffusion_coeff_fncpu = functools.partial(diffusion_coeff, sigma=sigma_emnist, device=device)
#    score_modelcpu = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fncpu))
#    score_modelcpu = score_modelcpu.to(device)
#    score_modelcpu.load_state_dict(torch.load("ckpt.pth", map_location=device))
#    samplercpu = Euler_Maruyama_sampler

#else:
device = "cuda"
emnist_marginal_prob_std_fngpu = functools.partial(marginal_prob_std, sigma=sigma_emnist, device=device)
emnist_diffusion_coeff_fngpu = functools.partial(diffusion_coeff, sigma=sigma_emnist, device=device)
emnist_score_modelgpu = torch.nn.DataParallel(ScoreNet(marginal_prob_std=emnist_marginal_prob_std_fngpu))
emnist_score_modelgpu = emnist_score_modelgpu.to(device)
emnist_score_modelgpu.load_state_dict(torch.load("ckpt.pth", map_location=device))
emnist_samplergpu = functools.partial(Euler_Maruyama_sampler,
    score_model=emnist_score_modelgpu,
    marginal_prob_std=emnist_marginal_prob_std_fngpu,
    diffusion_coeff=emnist_diffusion_coeff_fngpu,
    device=device
)


kmnist_marginal_prob_std_fngpu = functools.partial(marginal_prob_std, sigma=sigma_kmnist, device=device)
kmnist_diffusion_coeff_fngpu = functools.partial(diffusion_coeff, sigma=sigma_kmnist, device=device)
kmnist_score_modelgpu = torch.nn.DataParallel(ScoreNet(marginal_prob_std=kmnist_marginal_prob_std_fngpu))
kmnist_score_modelgpu = kmnist_score_modelgpu.to(device)
kmnist_score_modelgpu.load_state_dict(torch.load("kmnist_ckpt_20200_110_142.pth", map_location=device))
kmnist_samplergpu = functools.partial(Euler_Maruyama_sampler,
    score_model=kmnist_score_modelgpu,
    marginal_prob_std=kmnist_marginal_prob_std_fngpu,
    diffusion_coeff=kmnist_diffusion_coeff_fngpu,
    device=device
)    


hasy_marginal_prob_std_fngpu = functools.partial(marginal_prob_std, sigma=sigma_hasy, device=device)
hasy_diffusion_coeff_fngpu = functools.partial(diffusion_coeff, sigma=sigma_hasy, device=device)
hasy_score_modelgpu = torch.nn.DataParallel(ScoreNet(marginal_prob_std=hasy_marginal_prob_std_fngpu))
hasy_score_modelgpu = hasy_score_modelgpu.to(device)
hasy_score_modelgpu.load_state_dict(torch.load("hasy_ckpt_17000_4_17.pth", map_location=device))
hasy_samplergpu = functools.partial(Euler_Maruyama_sampler,
    score_model=hasy_score_modelgpu,
    marginal_prob_std=hasy_marginal_prob_std_fngpu,
    diffusion_coeff=hasy_diffusion_coeff_fngpu,
    device=device
)


option = st.selectbox(
    'From dataset',
    ('EMNIST', 'kMNIST', 'HASYv2'))
if option=='EMNIST':
    sampler=emnist_samplergpu
elif option=='kMNIST':
    sampler=kmnist_samplergpu
elif option=='HASYv2':
    sampler=hasy_samplergpu
    
    


if st.button("Generate"):
    # Generate samples using the specified sampler.
    # if device == "cuda":
    samples = sampler()
    # else:
    #     samples = samplercpu(
    #         score_modelcpu,
    #         marginal_prob_std_fncpu,
    #         diffusion_coeff_fncpu,
    #         device,
    #         64,
    #     )
    # Sample visualization.
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(64)))
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    tensor=sample_grid.permute(2, 1, 0)
    if device == "cuda":
        tensor=tensor.cpu()
    tensor=tensor.numpy()
    st.image(tensor, caption="Generated images", use_column_width=True)
