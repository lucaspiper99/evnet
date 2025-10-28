import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .utils import gaussian_kernel, gabor_kernel
from typing import Any, List, Tuple, Union

EPSILON = 1e-5

class Identity(nn.Module):
    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        return x

class Affine(nn.Module):
    def __init__(self, m=[1], b=[0]):
        super().__init__()
        m = torch.tensor(m, dtype=torch.float32).view(1,len(m),1,1)
        b = torch.tensor(b, dtype=torch.float32).view(1,len(b),1,1)
        self.m = nn.Parameter(m, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, x):
        return x * self.m + self.b

class MultiKernelConv2D(nn.Module):
    def __init__(self, in_channels_idx, out_channels_idx, kernel_sizes, paddings, bias, padding_modes, groups=None):
        super().__init__()
        if not groups: groups = [1]*len(in_channels_idx)
        assert len(in_channels_idx)==len(out_channels_idx)==len(kernel_sizes)==len(paddings)==len(bias)==len(padding_modes)==len(groups)
        self.in_channels_idx = in_channels_idx
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=len(in_ch), out_channels=out_ch, kernel_size=k, padding=p, groups=g, bias=b, padding_mode=m)
            for in_ch, out_ch, k, p, g, b, m in zip(in_channels_idx, out_channels_idx, kernel_sizes, paddings, groups, bias, padding_modes)
        ])

    def forward(self, x):
        outputs = [conv(x[:, self.in_channels_idx[c]]) for c, conv in enumerate(self.convs)]
        return torch.cat(outputs, dim=1)


class GFB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (kernel_size // 2, kernel_size // 2)

        # Param instatiations
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

    def initialize(self, sf, theta, sigx, sigy, phase, color, complex_color2=False):
        # random_channel = torch.randint(0, self.in_channels, (self.out_channels,))
        if complex_color2:
            for i in range(self.out_channels//2):
                self.weight[i, int(color[i])] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                                theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
            for i in range(self.out_channels//2, self.out_channels):
                self.weight[i, 0] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                                theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
                self.weight[i, 1] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                                theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
                self.weight[i] /= 2
        else:
            for i in range(self.out_channels):
                self.weight[i, int(color[i])] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                                theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
        self.weight = nn.Parameter(self.weight, requires_grad=False)


class VOneBlock(nn.Module):
    def __init__(self, sf, theta, sigx, sigy, phase, color, in_channels=3,
                 k_exc=25, noise_mode=None, noise_scale=1, noise_level=0, fano_factor=1,
                 simple_channels=128, complex_channels=128, ksize=25, stride=4, input_size=224
                 ):
        super().__init__()

        self.in_channels = in_channels

        self.simple_channels = simple_channels
        self.complex_channels = complex_channels
        self.out_channels = simple_channels + complex_channels
        self.stride = stride
        self.input_size = input_size

        self.sf = sf
        self.theta = theta
        self.sigx = sigx
        self.sigy = sigy
        self.phase = phase
        self.color = color
        self.k_exc = k_exc

        self.set_noise_mode(noise_mode, noise_scale, noise_level, fano_factor)
        self.fixed_noise = None
        
        self.simple_conv_q0 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q1 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase, color=self.color)
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2, color=self.color)

        self.simple = nn.ReLU(inplace=False)
        self.complex = Identity()
        self.gabors = Identity()
        #self.noise = nn.ReLU(inplace=True)
        self.noise = Identity()
        self.output = Identity()

    def forward(self, x):
        # Gabor activations [Batch, out_channels, H/stride, W/stride]
        x = self.gabors_f(x)
        # Noise [Batch, out_channels, H/stride, W/stride]
        x = self.noise_f(x)
        # V1 Block output: (Batch, out_channels, H/stride, W/stride)
        x = self.output(x)
        return x

    def gabors_f(self, x):
        s_q0 = self.simple_conv_q0(x)
        s_q1 = self.simple_conv_q1(x)
        c = self.complex(torch.sqrt(s_q0[:, self.simple_channels:, :, :] ** 2 +
                                    s_q1[:, self.simple_channels:, :, :] ** 2) / np.sqrt(2))
        s = self.simple(s_q0[:, 0:self.simple_channels, :, :])
        return self.gabors(self.k_exc * torch.cat((s, c), 1))


    def noise_f(self, x):
        if self.noise_mode == 'neuronal':
            x *= self.noise_scale
            x += self.noise_level
            if self.fixed_noise is not None:
                x += self.fixed_noise * torch.sqrt(self.fano_factor * F.relu(x.clone()) + EPSILON)
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * \
                     torch.sqrt(self.fano_factor * F.relu(x.clone()) + EPSILON)
            #x -= self.noise_level
            #x /= self.noise_scale
        if self.noise_mode == 'gaussian':
            if self.fixed_noise is not None:
                x += self.fixed_noise * self.noise_scale
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * self.noise_scale
        return self.noise(x)

    def set_noise_mode(self, noise_mode=None, noise_scale=1, noise_level=1, fano_factor=1):
        if not noise_mode: print(f'[VOneBlock] Using noise_mode={noise_mode}.')
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level
        self.fano_factor = fano_factor

    def fix_noise(self, batch_size=128, seed=42):
        noise_mean = torch.zeros(batch_size, self.out_channels, int(self.input_size/self.stride),
                                 int(self.input_size/self.stride))
        if seed:
            torch.manual_seed(seed)
        if self.noise_mode:
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean, scale=1).rsample().to(torch.cuda.current_device())

    def unfix_noise(self):
        self.fixed_noise = None


class DoG(nn.Module):
    def __init__(self, in_channels, p_channels, m_channels, kernel_size, r_c,  r_s, opponency, with_light_adapt):
        super().__init__()
        num_conv = bool(p_channels) + bool(m_channels)
        assert r_c.shape[0] == r_s.shape[0] == num_conv
        assert opponency.shape[0] == p_channels+m_channels
        self.in_channels = in_channels
        self.p_channels = p_channels
        self.m_channels = m_channels
        self.kernel_size = kernel_size
        self.r_c = r_c
        self.r_s = r_s
        self.opponency = opponency
        
        multi_conv_in = [torch.arange(in_channels)]*bool(p_channels)+\
        [torch.arange(in_channels, in_channels*2)]*bool(m_channels)
        if (bool(m_channels) and not with_light_adapt) or p_channels == 0:
            multi_conv_in[-1] = torch.arange(in_channels)

        self.multi_conv = MultiKernelConv2D(
            multi_conv_in,
            [p_channels]*bool(p_channels)+[m_channels]*bool(m_channels), kernel_size,
            ['same']*num_conv, [False]*num_conv, ['reflect']*num_conv
            )
        for i in range(p_channels+m_channels):
            conv_i = i // p_channels if p_channels > 0 else 0
            filter_i = i % p_channels if p_channels > 0 else 0
            conv = self.multi_conv.convs[conv_i]
            center = gaussian_kernel(sigma=r_c[conv_i]/np.sqrt(2), size=kernel_size[conv_i], norm=False)
            surround = gaussian_kernel(sigma=r_s[conv_i]/np.sqrt(2), size=kernel_size[conv_i], norm=False)
            kernels = tuple((c * center + s * surround) for c, s in zip(*opponency[i]))
            kernels = torch.stack(kernels, dim=0) if in_channels > 1 else torch.sum(torch.stack(kernels, dim=0), dim=0)
            conv.weight.data[filter_i] = kernels
            conv.weight.data[filter_i] /= torch.abs(conv.weight.data[filter_i].sum())
            conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.multi_conv(x)


class LightAdaptation(nn.Module):
    def __init__(self, in_channels, p_channels, m_channels, light_adapt_threshold):
        super().__init__()
        num_conv = bool(p_channels) + bool(m_channels)
        self.in_channels = in_channels
        self.epsilon = 0.05 if light_adapt_threshold else EPSILON
        
        def mean_without_zeros(x):
            mask = (x > self.epsilon)
            sum_nonzero = (x * mask).sum(dim=(1, 2, 3), keepdim=True)
            count_nonzero = mask.sum(dim=(1, 2, 3), keepdim=True)
            mean = sum_nonzero / count_nonzero.clamp(min=1)
            return mean.expand(-1, num_conv, -1, -1)

        if light_adapt_threshold:
            self.mean = lambda x: mean_without_zeros(x)
            return
        self.mean = lambda x: torch.mean(x, axis=(1, 2, 3), keepdim=True, dtype=torch.float).expand(-1, num_conv, -1, -1)
        return

    def forward(self, x):
        num = x.unsqueeze(1) - self.mean(x).unsqueeze(2)
        denom = self.mean(x).unsqueeze(2) + self.epsilon
        return  (num / denom).reshape(x.shape[0], -1, *x.shape[2:])


class ContrastNormalization(nn.Module):
    def __init__(self, in_channels, p_channels, m_channels, kernel_sizes, radii, c50, n):
        super().__init__()
        num_conv = bool(p_channels) + bool(m_channels)
        self.in_channels = in_channels
        self.kernel_size = kernel_sizes
        self.radius = radii
        select_idxs = torch.tensor([0]*p_channels+[1]*m_channels) if p_channels > 0 else torch.tensor([0])
        self.c50 = torch.index_select(torch.from_numpy(c50), 0, select_idxs)
        self.c50 = nn.Parameter(self.c50.view(1, p_channels+m_channels, 1, 1), requires_grad=False)
        self.n = torch.index_select(torch.from_numpy(n), 0, select_idxs)
        self.n = nn.Parameter(self.n.view(1, p_channels+m_channels, 1, 1), requires_grad=False)

        idxs = [p_channels]*bool(p_channels)+[m_channels]*bool(m_channels)
        self.multi_conv = MultiKernelConv2D(
            torch.arange(p_channels+m_channels).split(idxs), idxs,
            kernel_sizes, ['same']*num_conv, [False]*num_conv, ['reflect']*num_conv, idxs
            )
        
        for conv_i, (num_channels, conv) in enumerate(zip(idxs, self.multi_conv.convs)):
            filter = gaussian_kernel(sigma=radii[conv_i]/np.sqrt(2), size=kernel_sizes[conv_i])
            conv.weight.data = filter.expand(num_channels, -1, -1).unsqueeze(1) / torch.sum(filter)
            conv.weight.requires_grad = False

    def forward(self, x) -> torch.Tensor:
        return x / (self.multi_conv(x ** 2) ** .5 + self.c50)**self.n


class SubcorticalBlock(nn.Module):
    def __init__(
        self, in_channels, p_channels, m_channels, 
        rc_dog, rs_dog, opponency_dog, kernel_dog,
        kernel_cn, radius_cn, c50_cn, exp_cn,
        k_exc, fano_factor=.4, noise_mode=None, light_adapt_threshold=False,
        with_light_adapt=True, with_dog=True, with_contrast_norm=True, with_relu=True
        ):
        super().__init__()
        self.in_channels = in_channels
        self.p_channels = p_channels
        self.m_channels = m_channels
        self.noise_mode = noise_mode
        self.fano_factor = fano_factor
        self.fixed_noise = None
        self.k_exc = torch.tensor(k_exc, dtype=torch.float32).view(1, p_channels+m_channels, 1, 1)
        self.k_exc = nn.Parameter(self.k_exc, requires_grad=False)

        self.light_adapt = Identity()
        self.dog = Identity()
        self.contrast_norm = Identity()
        self.nonlinearity = Identity()

        if with_light_adapt:
            self.light_adapt = LightAdaptation(in_channels, p_channels, m_channels, light_adapt_threshold)
        if with_dog:
            self.dog = DoG(in_channels, p_channels, m_channels, kernel_dog, rc_dog, rs_dog, opponency_dog, with_light_adapt)
        if with_contrast_norm:
            self.contrast_norm = ContrastNormalization(in_channels, p_channels, m_channels, kernel_cn, radius_cn, c50_cn, exp_cn)
        if with_relu:
            self.nonlinearity = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.light_adapt(x)
        x = self.dog(x)
        x = self.contrast_norm(x)
        x = self.nonlinearity(x)
        x *= self.k_exc
        x = self.noise_f(x)
        return x

    def noise_f(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_mode:
            r = self.fixed_noise if self.fixed_noise is not None else\
                torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample()
            if self.noise_mode == 'neuronal':
                x += r * torch.sqrt(self.fano_factor * ((torch.abs(x.clone()) + EPSILON)))
        return x

    def fix_noise(self, image_size=224, batch_size=128, seed=42):
        noise_mean = torch.zeros(
            batch_size, self.p_channels+self.m_channels, image_size, image_size
            )
        if seed:
            torch.manual_seed(seed)
        if self.noise_mode:
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean, scale=1).rsample().to(torch.cuda.current_device())

    def unfix_noise(self):
        self.fixed_noise = None


class EVBlock(nn.Module):
    def __init__(self, subcorticalblock, voneblock, backend_in_features, lgn_to_v2=False):
        super().__init__()
        self.subcorticalblock = subcorticalblock
        self.voneblock = voneblock
        self.lgn_to_v2 = lgn_to_v2
        stride = voneblock.simple_conv_q0.stride[0]
        assert stride%2==0
        self.pool = nn.MaxPool2d(kernel_size=(stride+1), stride=stride, padding=(stride+1)//2)
        bottleneck_in_features = voneblock.simple_channels + voneblock.complex_channels
        bottleneck_out_features = backend_in_features if not self.lgn_to_v2 else backend_in_features - subcorticalblock.p_channels - subcorticalblock.m_channels
        self.bottleneck = nn.Conv2d(bottleneck_in_features, bottleneck_out_features, 1, bias=False, groups=1)
    
    def forward(self, x):
        if self.lgn_to_v2:
            y_lgn = self.subcorticalblock(x)
            y_v1 = self.voneblock(y_lgn)
            return torch.concat([self.bottleneck(y_v1), self.pool(y_lgn)], dim=1)
        else:
            y_lgn = self.subcorticalblock(x)
            y_v1 = self.voneblock(y_lgn)
            return self.bottleneck(y_v1)


class StopForwardException(Exception):
    """Internal sentinel used to stop model forward after capturing activation."""
    pass


class EnsembleUpToLayer(nn.Module):
    """
    Ensemble wrapper that runs `n_models` forward passes up to `layer_name`,
    averages the activations at that layer, then runs the remainder of the
    network once using the averaged activation.

    Usage:
        ensemble = EnsembleUpToLayer(base_model, n_models=8, layer_name='voneblock')
        ensemble.eval()
        with torch.no_grad():
            out = ensemble(input_tensor)
    """

    def __init__(self, base_model: nn.Module, n_models: int, layer_name: str):
        super().__init__()
        assert isinstance(base_model, nn.Module)
        assert isinstance(n_models, int) and n_models >= 1
        assert isinstance(layer_name, str) and len(layer_name) > 0

        self.model = base_model
        self.n_models = n_models
        self.layer_name = layer_name
        self._target_module = self._get_module(layer_name)
        if self._target_module is None:
            raise ValueError(f"Could not find target module named '{layer_name}' in the model.")


    def _get_module(self, layer_name: str) -> Union[nn.Module, None]:
        """
        Get a module given a string. We try:
          1) Dotted attribute traversal on the root (best for 'model.conv1' etc).
          2) Fallback to searching model.named_modules() for a module with name equal to or
             ending with layer_name (matching the last path component).
        Returns the module or None.
        """
        try:
            mod = self.model
            for part in layer_name.split('.'):
                if hasattr(mod, part):
                    mod = getattr(mod, part)
                else:
                    if isinstance(mod, (nn.Sequential, nn.ModuleList)) and part.isdigit():
                        mod = mod[int(part)]
                    else:
                        raise AttributeError(f"Module has no attribute/child '{part}' during lookup of '{dotted}'")
            return mod
        except Exception:
            pass
        return None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
         - run model up to target module N times, capturing the target output each time (early-abort).
         - average the captured activations.
         - run model once more with a hook that replaces the target module's output with the averaged activation.
        """

        assert not self.model.training

        captured: List[Any] = []

        def capture_hook(module, _input, output):
            captured.append(output.detach().cpu() if output.device is not None else output.detach())
            raise StopForwardException()

        device = x.device

        for i in range(self.n_models):
            handle = self._target_module.register_forward_hook(capture_hook)
            try:
                with torch.no_grad():
                    _ = self.model(x)
            except StopForwardException:
                pass
            finally:
                handle.remove()

            if len(captured) != i + 1: # an activation failed to be captured
                raise RuntimeError(f"Failed to capture activation on iteration {i}. "
                                   f"Captured count={len(captured)} expected={i+1}.")

        captured = [o.to(device) for o in captured]

        avg_activation = torch.stack([o.detach() for o in captured], dim=0).mean(dim=0)

        avg_activation = avg_activation.detach().to(device)

        def replace_hook(module, _input, _output):
            return avg_activation

        handle2 = self._target_module.register_forward_hook(replace_hook)
        try:
            with torch.no_grad():
                out = self.model(x)
        finally:
            handle2.remove()

        return out

