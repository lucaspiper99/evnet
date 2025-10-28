from collections import OrderedDict
from torch import nn
import numpy as np
from .modules import SubcorticalBlock, VOneBlock, Identity, EVBlock
from .params import get_tuned_params, generate_gabor_param
from .backends import get_resnet, get_efficientnet, get_cornet_z


def EVNet(
    in_channels=3, num_classes=1000, model_arch='resnet-50', image_size=224, visual_degrees=7,
    # SubcorticalBlock
    with_subcorticalblock=True,  p_channels=3, m_channels=0, colors_p_cells=['r/g', 'g/r', 'b/y'], 
    with_light_adapt=True, with_dog=True, with_contrast_norm=True, with_relu=False,
    subcort_noise_mode='neuronal', subcort_fano_factor=0.4,  
    # VOneBlock
    with_voneblock=True, sf_corr=0.75, sf_max=8, sf_min=0, rand_param=False, gabor_seed=0, simple_channels=256, complex_channels=256,
    vone_noise_mode='neuronal', vone_noise_scale=1, vone_noise_level=0, vone_ksize=31, vone_stride=4,
    gabor_color_prob=None, vone_k_exc=None, vone_fano_factor=None, light_adapt_threshold=False, 
    lgn_to_v2=False
    ):
    """Generates n.SEquential with SubcorticalBloc, VOneBlock and back-end architecture.

    :param int in_channels: Number of input channels, defaults to 3
    :param int num_classes: Number of classes, defaults to 1000
    :param str model_arch: Model architecture ('resnet-50', 'efficientnet-b0', 'cornet-z'), defaults to 'resnet-50'
    :param int image_size: Image size, defaults to 224
    :param int visual_degrees: Visual degrees, defaults to 7
    :param bool with_subcorticalblock: Whether to include a SubcorticalBlock, defaults to True
    :param int p_channels:  Number of P channels, defaults to 3
    :param int m_channels:  Number of M channels, defaults to 0
    :param list colors_p_cells: P-channel color opposition, defaults to ['r/g', 'g/r', 'b/y']
    :param bool with_light_adapt: Whether to include in light adaptation module the model, defaults to True
    :param bool with_dog: Whether to include in DoG convolution module the model, defaults to True
    :param bool with_contrast_norm: Whether to include in contrast normalization module the model, defaults to True
    :param bool with_relu: Whether to include in ReLU the model, defaults to False
    :param str subcort_noise_mode: Whether to include in subcortical noise the model, defaults to 'neuronal'
    :param float subcort_fano_factor: SubcorticalBlock noise fano factor, defaults to 0.4
    :param bool with_voneblock: Whether to include a VOneBlock, defaults to True
    :param float sf_corr: SF correlation factor, defaults to 0.75
    :param int sf_max: Max spatial frequency, defaults to 8
    :param int sf_min: Min spatial frequency, defaults to 0
    :param bool rand_param: Random Gabor parameters flag, defaults to False
    :param int gabor_seed: Gabor RNG seed, defaults to 0
    :param int simple_channels: Number of simple channels, defaults to 256
    :param int complex_channels: Number of complex channels , defaults to 256
    :param str vone_noise_mode: VOneBlock noise mode ('neuronal', 'None'), defaults to 'neuronal'
    :param int vone_noise_scale: VOneBlock noise scale, defaults to 1
    :param int vone_noise_level: VOneBlock noise level, defaults to 0
    :param int vone_ksize: VOneBlock kernel size, defaults to 31
    :param int vone_stride: VOneBlock stride, defaults to 4
    :param list gabor_color_prob: GFB color input, defaults to None
    :param float vone_k_exc: VOneBlock scaling factor k_exc, defaults to None
    :param float vone_fano_factor: VOneBlock noise fano factor, defaults to None
    :param bool light_adapt_threshold: Whether to use light adaptation thresholding, defaults to False
    :param bool lgn_to_v2: Whether to include LGN-V" skip connections, defaults to False
    :return nn.Sequential: model 
    """    

    subcorticalblock_params, gabor_params, arch_params = None, None, None
    if with_subcorticalblock:
        subcort_params = get_tuned_params(
            p_channels, m_channels, colors_p_cells, ['w/b'], visual_degrees, image_size
            )
        subcorticalblock = SubcorticalBlock(
            **subcort_params, in_channels=in_channels, m_channels=m_channels, p_channels=p_channels,
            fano_factor=subcort_fano_factor, noise_mode=subcort_noise_mode,
            with_dog=with_dog, with_light_adapt=with_light_adapt, with_contrast_norm=with_contrast_norm, with_relu=with_relu,
            light_adapt_threshold=light_adapt_threshold
            )

    if with_voneblock:
        out_channels = simple_channels + complex_channels
        vone_in_channels = p_channels + m_channels if with_subcorticalblock else in_channels

        sf, theta, phase, nx, ny, color = generate_gabor_param(
                simple_channels, complex_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min,
                color_prob=gabor_color_prob, in_channels=vone_in_channels
                )
        gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                        'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                        'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy(), 'color': color.copy()}

        arch_params = {'k_exc': vone_k_exc, 'arch': model_arch, 'ksize': vone_ksize, 'stride': vone_stride}

        # Conversions
        ppd = image_size / visual_degrees
        sf = sf / ppd
        sigx = nx / sf
        sigy = ny / sf
        theta = theta / 180 * np.pi
        phase = phase / 180 * np.pi

        voneblock = VOneBlock(
            sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase, color=color, in_channels=vone_in_channels, k_exc=vone_k_exc,
            noise_mode=vone_noise_mode, noise_scale=vone_noise_scale, noise_level=vone_noise_level, fano_factor=vone_fano_factor,
            simple_channels=simple_channels, complex_channels=complex_channels,
            ksize=vone_ksize, stride=vone_stride, input_size=image_size,
            )

    if model_arch:
        if model_arch == 'resnet-50':
            backend = get_resnet(
                in_channels=(p_channels+m_channels if with_subcorticalblock else in_channels),
                num_classes=num_classes,
                layers=50,
                backend=with_voneblock
                )
        if model_arch == 'efficientnet-b0':
            backend = get_efficientnet(
                b=0,
                in_channels=(p_channels+m_channels if with_subcorticalblock else in_channels),
                num_classes=num_classes,
                backend=with_voneblock,
                )
        if model_arch == 'cornet-z':
            backend = get_cornet_z(
                in_channels=(p_channels+m_channels if with_subcorticalblock else in_channels),
                num_classes=num_classes,
                backend=with_voneblock
                )

    model_dict = OrderedDict([])
    if lgn_to_v2:
        assert with_subcorticalblock and with_voneblock and model_arch
        evblock = EVBlock(subcorticalblock, voneblock, backend.in_channels, lgn_to_v2=True)
        model_dict.update({'evblock': evblock})
    else:
        if with_subcorticalblock:
            model_dict.update({'subcorticalblock': subcorticalblock})
        if with_voneblock:
            model_dict.update({'voneblock': voneblock})
        if with_voneblock and len(model_arch) > 0:
            model_dict.update({'voneblock_bottleneck': nn.Conv2d(out_channels, backend.in_channels, 1, bias=False, groups=1)})
    
    if len(model_arch) > 0:
        model_dict.update({'model': backend})
    else:
        model_dict.update({'output': Identity()})

    model = nn.Sequential(model_dict)

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.subcort_params = subcort_params
    model.gabor_params = gabor_params
    model.arch_params = arch_params
    model.is_stochastic = False if (subcort_noise_mode is None and vone_noise_mode is None) else True

    return model
