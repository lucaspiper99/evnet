import torch
import argparse
import tqdm
import numpy as np
from collections import OrderedDict
from torch import nn
from params import get_kernel_size
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from tqdm import tqdm

from evnet.tuning import *
from evnet.modules import Affine, LightAdaptation, DoG, ContrastNormalization 


parser = argparse.ArgumentParser(
    description="Finetune SubcorticalBlock hyperparameters on response properties via Bayesian Optimization."
)
parser.add_argument(
    "--cell_type", default="p", choices=['p', 'm'], type=str, help="Which LGN cell type to tune.",
)
parser.add_argument(
    "--n_calls", default=640, type=int, help="Number of calls to the optimization function.",
)
parser.add_argument(
    "--n_initial_points", default=64, type=int, help="Number of initial points before optimization.",
)
parser.add_argument(
    "--initial_point_generator", default="sobol", type=str, help="Initial point generation strategy.",
)
parser.add_argument(
    "--seed", default=42, type=int, help="RNG seed.",
)
parser.add_argument(
    "--out_dir", type=str, help="Output directory for CSV file.",
)
parser.add_argument(
    "--devices", default="0", type=str, help="Device to use when training the model",
)


ARGS = parser.parse_args()
EPSILON = 1e-5
ERROR_COST = 10

def get_model_dict(radius_cn, c50_cn, exp_cn, kernel_cn, rc_dog, rs_dog, ratio_dog, kernel_dog):
    """Returns model ordered dictionary to create nn.Sequential.

    :param float radius_cn: Contrast normalization radius
    :param float c50_cn: Contrast normalization semissaturation constant
    :param float exp_cn: Contrast normalization exponent
    :param int kernel_cn: Contrast normalization kernel size
    :param float rc_dog: DoG center radius
    :param float rs_dog: DoG surround radius
    :param float ratio_dog: DoG Peak contrast sensitivity ratio
    :param int kernel_dog: DoG kernel size
    :return OrderedDict: Model dictionary
    """    
    model_dict = OrderedDict()

    # Light Adaptation
    light_adapt = LightAdaptation(
        in_channels=3,
        p_channels=p_channels,
        m_channels=m_channels,
        light_adapt_threshold=False
        )
    model_dict['light_adapt'] = light_adapt
    

    # DoG
    dog = DoG(
        in_channels=3,
        p_channels=p_channels,
        m_channels=m_channels,
        kernel_size=np.array([kernel_dog]),
        r_c=np.array([rc_dog]),
        r_s=np.array([rs_dog]),
        opponency=np.array([[[1, 0, 0], [0, ratio_dog, 0]]]),
        with_light_adapt=True
        )
    model_dict['dog'] = dog
    
    # Contrast Normalization
    contrast_norm = ContrastNormalization(
        in_channels=3,
        p_channels=p_channels,
        m_channels=m_channels,
        kernel_sizes=np.array([kernel_cn]),
        radii=np.array([radius_cn]),
        c50=np.array([c50_cn]),
        n=np.array([exp_cn]),
        )
    model_dict['contrast_norm'] = contrast_norm

    model_dict['scaling'] = Affine(m=[1], b=[0])

    model_dict['relu'] = nn.ReLU()

    return model_dict


class tqdm_skopt(object):
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)
        self._min_value = float('inf')

    def __call__(self, res):
        self._bar.update()
        if res.fun < self._min_value:
            self._min_value = res.fun
        self._bar.set_postfix_str(f"Current min: {self._min_value:.2f}")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_bar']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._bar = tqdm(**self._bar_kwargs)


# General parameters
image_size = 224
visual_degrees = 7
ppd = image_size / visual_degrees
in_channels = 3
F1_freq = 1
sampling_rate = 8
grating_radius = 1
gaussian_energy = 0.75
contrast = .75 if ARGS.cell_type=='p' else .3
x, y = image_size//2, image_size//2
nyquist_f = 1/(visual_degrees/image_size)/2  / np.sqrt(2)
p_channels, m_channels = (0, 1) if ARGS.cell_type=='P' else (1, 0)

# Drifting grating parameters 
sf_levels = np.logspace(np.log10(.07), np.log10(12), num=12)
r_levels = np.logspace(np.log2(.02), np.log2(6), base=2, num=12)
contrast_levels = np.logspace(np.log2(.01), np.log2(1), 12, base=2)

# Reference response property values
refs = get_reference_distributions()
# https://doi.org/10.1523/JNEUROSCI.22-01-00338.2002
rc = refs[f'center_radius_{ARGS.cell_type}_cell'].mean()
rs = refs[f'surround_radius_{ARGS.cell_type}_cell'].mean()
# https://doi.org/10.1523/JNEUROSCI.22-01-00338.2002
ref_sup_idx = .711 if ARGS.cell_type=='p' else .464
# https://doi.org/10.1152/jn.00058.2022
ref_peak_target = 85 if ARGS.cell_type=='p' else 130
# https://doi.org/10.1016/0042-6989(94)E0066-T
ref_dog_sens = 4.4/325.2 if ARGS.cell_type=='p' else 1.1/148.0

# Search space
param_space = [
    Real(.5*rs*ppd, 1.5*rs*ppd, name="radius_cn"),
    Real(.01, 1.0, name="c50_cn"),
    Real(.01, 1.0, name="exp_cn"),
    Real(.8*rc*ppd, 1.2*rc*ppd, name="rc_dog"),
    Real(.8*rs*ppd, 1.2*rs*ppd, name="rs_dog"),
    Real(-ref_dog_sens*5, -ref_dog_sens*.25, name="ratio_dog")
]

data = {
    # Errors
    'l2_error': [], 'linf_error': [],
    # Parameters
    'rc_dog': [], 'rs_dog': [], 'kernel_dog': [], 'ratio_dog': [],
    'radius_cn': [], 'kernel_cn': [], 'c50_cn': [], 'exp_cn': [],
    'k_exc': [],
    # SF Tuning
    'sf_f1_resp': [], 'sf_r2': [],
    # Size Tuning
    'size_f1_resp': [], 'size_r2': [], 'size_sup_idx': [], 'size_optim_sf': [],
    # Size Tuning
    'contrast_f1_resp': [], 'contrast_sat_idx': [],
}

sf_popt_param_names = 'rc', 'rs', 'Kc', 'Ks'
for i, p in enumerate(sf_popt_param_names):
    data[f'sf_popt_{p}'] = []
size_popt_param_names = 're', 'ri', 'Ke', 'Ki'
for i, p in enumerate(size_popt_param_names):
    data[f'size_popt_{p}'] = []
contrast_popt_param_names = 'r_ramp', 'C0'
for i, p in enumerate(contrast_popt_param_names):
    data[f'contrast_popt_{p}'] = []


# Define the objective function
@use_named_args(param_space)
def objective(radius_cn, c50_cn, exp_cn, rc_dog, rs_dog, ratio_dog):
    """Computes the mean L2 error between SubcorticalBlock and reference LGN response properties. 

    :param float radius_cn: Contrast normalization radius
    :param float c50_cn: Contrast normalization semissaturation constant
    :param float exp_cn: Contrast normalization exponent
    :param float rc_dog: DoG center radius
    :param float rs_dog: DoG surround radius
    :param float ratio_dog: DoG Peak contrast sensitivity ratio
    :return float: L2 response property error
    """    

    kernel_dog = get_kernel_size(rs_dog, gaussian_energy)
    kernel_cn = get_kernel_size(radius_cn, gaussian_energy)

    model_dict = get_model_dict(radius_cn, c50_cn, exp_cn, kernel_cn, rc_dog, rs_dog, ratio_dog, kernel_dog)
    model = nn.Sequential(model_dict)

    # Scaling Factor to get sp/50-ms window
    m = get_scaling(model, [ref_peak_target/20], sampling_rate, image_size, visual_degrees)
    model_dict['scaling'] = Affine(m=m)
    model = nn.Sequential(model_dict)
    
    # SF Tuning
    popt_sf, f1_resp_sf, r2_sf = fit_dog(model, sf_levels, grating_radius, contrast, sampling_rate, visual_degrees, image_size)
    if len(popt_sf)==0:
        return ERROR_COST

    # Size Tuning
    optim_sf = sf_levels[np.argmax(f1_resp_sf)]
    popt_size, f1_resp_size, r2_size = fit_dogi(model, r_levels, optim_sf, contrast, sampling_rate, visual_degrees, image_size)
    if len(popt_size)==0:
        return ERROR_COST
    si_size = get_suppression_index(f1_resp_size)

    # Contrast Tuning
    popt_contrast, f1_resp_contrast, r2_contrast = fit_log_contrast(model, contrast_levels, optim_sf, visual_degrees/2*np.sqrt(2), sampling_rate, visual_degrees, image_size)
    si_contrast = get_saturation_index(contrast_levels, f1_resp_contrast)

    bounds = 2**-10, np.inf

    error_rc = np.log2(np.clip(popt_sf[0] / refs[f'center_radius_{ARGS.cell_type}_cell'].mean(), *bounds))
    error_rs = np.log2(np.clip(popt_sf[1] / refs[f'surround_radius_{ARGS.cell_type}_cell'].mean(), *bounds))

    error_re = np.log2(np.clip(popt_size[0] / refs[f'excitatory_radius_{ARGS.cell_type}_cell'].mean(), *bounds))
    error_ri = np.log2(np.clip(popt_size[1] / refs[f'inhibitory_radius_{ARGS.cell_type}_cell'].mean(), *bounds))

    error_sat_idx = np.log2(np.clip(si_contrast / refs[f'saturation_idx_{ARGS.cell_type}_cell'].mean(), *bounds))
    error_sup_idx = np.log2(np.clip((1-np.clip(si_size,0,1)) / (1-refs[f'suppression_idx_{ARGS.cell_type}_cell'].mean()), *bounds))

    l2_error = error_rc**2 + error_rs**2 + error_re**2 + error_ri**2 + error_sat_idx**2 + error_sup_idx**2
    linf_error = max(*[np.abs(a) for a in [
        error_rc, error_rs, error_re, error_ri, error_sat_idx, error_sup_idx
        ]])

    data['l2_error'].append(l2_error)
    data['linf_error'].append(linf_error)
    
    data['rc_dog'].append(rc_dog)
    data['rs_dog'].append(rs_dog)
    data['kernel_dog'].append(kernel_dog)
    data['ratio_dog'].append(ratio_dog)
    
    data['radius_cn'].append(radius_cn)
    data['kernel_cn'].append(kernel_cn)
    data['c50_cn'].append(c50_cn)
    data['exp_cn'].append(exp_cn)

    data['k_exc'].append(m[0])

    for i, p in enumerate(sf_popt_param_names):
        data[f'sf_popt_{p}'].append(popt_sf[i])
    data['sf_f1_resp'].append(f1_resp_sf)
    data['sf_r2'].append(r2_sf)

    data['size_optim_sf'].append(optim_sf)
    for i, p in enumerate(size_popt_param_names):
        data[f'size_popt_{p}'].append(popt_size[i])
    data['size_f1_resp'].append(f1_resp_size)
    data['size_r2'].append(r2_size)
    data['size_sup_idx'].append(si_size)  # Suppression Index

    for i, p in enumerate(contrast_popt_param_names):
        data[f'contrast_popt_{p}'].append(popt_contrast[i])
    data['contrast_f1_resp'].append(f1_resp_contrast)
    data['contrast_sat_idx'].append(si_contrast) # Saturation Index

    return l2_error


def main():

    torch.cuda.set_device(f'cuda:{ARGS.devices}' if torch.cuda.is_available() else "cpu")

    result = gp_minimize(
        objective, param_space, n_calls=ARGS.n_calls, n_initial_points=ARGS.n_initial_points,
        initial_point_generator='sobol', random_state=ARGS.seed,
        callback=[tqdm_skopt(total=ARGS.n_calls, desc="Bayesian Optimization")]
        )

    best_params = result.x
    best_score = -result.fun  # Convert back to maximization
    print(f"Best params: {best_params},\nBest fitness: {best_score}")

    pd.DataFrame(data).to_csv(os.path.join(ARGS.output_dir, f'{ARGS.cell_type}_cell_hyperparams_.csv'))

if __name__=='__main__':
    main()

