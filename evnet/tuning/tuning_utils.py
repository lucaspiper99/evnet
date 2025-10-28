import torch
import glob
import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf
from functools import partial
from evnet.utils import generate_grating
import evnet.params


def difference_of_gaussians(f, rc, rs, Kc, Ks, R0, C):
    return R0 + C*Kc*np.pi*(rc**2)*np.exp(-(np.pi*rc*f)**2) -\
        C*Ks*np.pi*(rs**2)*np.exp(-(np.pi*rs*f)**2)


def difference_of_gaussian_integral(s, re, ri, Ke, Ki, R0, C):
    integral_e = np.sqrt(np.pi) * re * erf(s / re)
    integral_i = np.sqrt(np.pi) * ri * erf(s / ri)
    return R0 + C * (Ke * integral_e - Ki * integral_i)


def log_contrast_func(c, r_ramp, C0, r_offset):
    return r_offset + r_ramp * np.log(1 + c / C0)


def get_r2(func, popt, x, y):
    y_pred = func(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def get_saturation_index(contrasts, responses):
    int_r = np.trapz(responses, contrasts)
    return float(2*(contrasts.max()-contrasts.min())*(int_r/(responses.max() - responses.min()))-1)

def get_suppression_index(responses):
    return responses[-1] / np.max(responses)

def get_low_freq_suppression_index(responses):
    return responses[0] / np.max(responses)

def get_f1_response(
    model, sf, radius, orientation, contrast, sampling_rate=8, image_size=224, visual_degrees=7
    ):
    sf = np.asarray(sf) if hasattr(sf, '__iter__') else np.array([sf])
    radius = np.asarray(radius) if hasattr(radius, '__iter__') else np.array([radius])
    orientation = np.asarray(orientation) if hasattr(orientation, '__iter__') else np.array([orientation])
    contrast = np.asarray(contrast) if hasattr(contrast, '__iter__') else np.array([contrast])

    f1_response = []
    iterable_prop = max([sf, radius, orientation, contrast], key=lambda x: x.size)
    get_curr = lambda x, idx : x[idx] if x.size > 1 else x[0]
    for idx, value in enumerate(iterable_prop):
        curr_sf = get_curr(sf, idx)
        curr_radius = get_curr(radius, idx)
        curr_orientation = get_curr(orientation, idx)
        curr_contrast = get_curr(contrast, idx)

        drifting_grating = torch.zeros((sampling_rate, image_size, image_size), dtype=torch.float)
        phases = np.linspace(0, 2*np.pi, sampling_rate, endpoint=False)
        for phase_idx, phase in enumerate(phases):
            drifting_grating[phase_idx] = generate_grating(
                **params.get_grating_params(
                    sf=curr_sf, phase=phase, radius=curr_radius, contrast=curr_contrast, angle=curr_orientation,
                    visual_degrees=visual_degrees, image_size=image_size
                    )
                )
    
        model.cuda().eval()
        with torch.no_grad():
            stimuli = drifting_grating.unsqueeze(1).expand(-1, 3, -1, -1)
            response = model(stimuli.cuda()).cpu()
        signal = torch.abs(response[:, 0, image_size//2, image_size//2])
        fft = torch.fft.fft(signal, axis=0)
        f1_response.append(torch.abs(fft[1]).item())
    return np.array(f1_response)


def fit_dog(
    model, sfs, radius=1, contrast=.75,
    sampling_rate=8, visual_degrees=7, image_size=224
    ):
    fit_func = partial(difference_of_gaussians, R0=0, C=1)
    f1_response = get_f1_response(model, sfs, radius, 0, contrast, sampling_rate, image_size, visual_degrees)
    try:
        popt, _ = curve_fit(
            fit_func,
            sfs, f1_response, maxfev=2000, bounds=[[0]*4, [np.inf]*4], p0=[1.]*4
            )
        r2 = get_r2(fit_func, popt, sfs, f1_response)
    except RuntimeError:
        return (), (), 0
    return popt, f1_response, r2        


def fit_dogi(
    model, radii, sf=1.85, contrast=.75,
    sampling_rate=8, visual_degrees=7, image_size=224
    ):
    fit_func = partial(difference_of_gaussian_integral, R0=0, C=1)
    f1_response = get_f1_response(model, sf, radii, 0, contrast, sampling_rate, image_size, visual_degrees)
    try:
        popt, _ = curve_fit(
            fit_func, radii, f1_response, maxfev=2000, bounds=[[0]*4, [np.inf]*4], p0=[1.]*4
            )
        r2 = get_r2(fit_func, popt, radii, f1_response)
    except RuntimeError:
        return (), (), 0
    return popt, f1_response, r2

def fit_log_contrast(
    model, contrasts, sf=1.85, radius=3.5*np.sqrt(2),
    sampling_rate=8, visual_degrees=7, image_size=224
    ):
    f1_response = get_f1_response(model, sf, radius, 0, contrasts, sampling_rate, image_size, visual_degrees)
    fit_func = partial(log_contrast_func, r_offset=0)
    try:
        popt, _ = curve_fit(
            fit_func, contrasts, f1_response, maxfev=2000, bounds=[[0, 0], [np.inf, np.inf]], p0=[1, 1]
            )
        r2 = get_r2(fit_func, popt, contrasts, f1_response)
    except RuntimeError:
        return (), (), 0
    return popt, f1_response, r2


def get_scaling(model, peak_target, sampling_rate, image_size, visual_degrees):
    sf, contrast, orientation = 1, .8, 0  # https://doi.org/10.1152/jn.00058.2022
    dpp = visual_degrees/image_size
    for r_idx, radius in enumerate(np.arange(dpp/2, 1, dpp)):
        drifting_grating = torch.zeros((sampling_rate, image_size, image_size), dtype=torch.float)
        phases = np.linspace(0, 2*np.pi, sampling_rate, endpoint=False)
        for phase_idx, phase in enumerate(phases):
            drifting_grating[phase_idx] = generate_grating(
                **params.get_grating_params(
                    sf=sf, phase=phase, radius=radius, contrast=contrast, angle=0,
                    visual_degrees=visual_degrees, image_size=image_size
                    )
                )

        model.cuda().eval()
        with torch.no_grad():
            stimuli = drifting_grating.unsqueeze(1).expand(-1, 3, -1, -1)
            response = model(stimuli.cuda()).cpu()
        signal = response[:, :, image_size//2, image_size//2]
        if r_idx == 0:
            peak = torch.ones((signal.size(1),)) * -torch.inf
        peak = torch.maximum(peak, signal.max(dim=0).values)

    return np.array(peak_target) / peak.numpy()


def get_reference_distributions():
    csv_data = {}
    csv_files = glob.glob("tuning/*.csv")
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        base_name = os.path.splitext(os.path.basename(csv_file))[0]  # Get file name without extension
        csv_data[base_name] = df.iloc[:, 1].to_numpy()
    return csv_data
