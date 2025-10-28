import yaml
from .diffeomorphism import Diffeo 
from .color_jitter import RandomSmoothColor 
from .rand_filter import RandomFilter
from .prime import PRIMEAugModule, GeneralizedPRIMEModule

def get_prime(preprocessing):
    with open('prime/prime_config.yml', "r") as f:
        config = yaml.safe_load(f)

    augmentations = []

    diffeo = Diffeo(
        sT=config['diffeo']['sT'], rT=config['diffeo']['rT'],
        scut=config['diffeo']['scut'], rcut=config['diffeo']['rcut'],
        cutmin=config['diffeo']['cutmin'], cutmax=config['diffeo']['cutmax'],
        alpha=config['diffeo']['alpha'], stochastic=True
    )
    augmentations.append(diffeo)


    color = RandomSmoothColor(
        cut=config['color_jit']['cut'], T=config['color_jit']['T'],
        freq_bandwidth=config['color_jit']['max_freqs'], stochastic=True
    )
    augmentations.append(color)


    filt = RandomFilter(
        kernel_size=config['rand_filter']['kernel_size'],
        sigma=config['rand_filter']['sigma'], stochastic=True
    )
    augmentations.append(filt)


    return GeneralizedPRIMEModule(
        preprocess=preprocessing,
        mixture_width=config['augmix']['mixture_width'],
        mixture_depth=config['augmix']['mixture_depth'],
        no_jsd=config['augmix']['no_jsd'], max_depth=3,
        aug_module=PRIMEAugModule(augmentations),
    )
