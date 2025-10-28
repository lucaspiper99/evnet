import math
import torch
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import product
from pathlib import Path
from typing import Literal

from .utils import sample_dist, set_seed

image_size = 224
visual_degrees = 7
ppd = (image_size/visual_degrees)

def get_kernel_size(radius, energy=0.9):
    """Get Gaussian kernel size based on percentage of energy retained. 

    :param float radius: Gaussian radius
    :param float energy: Gaussian energy to be retained, defaults to 0.9
    :return int: Kernel size
    """    
    if radius==np.inf:
        return 1
    assert 0 < energy < 1
    sigma = radius/np.sqrt(2)
    r_min = np.sqrt(-2*np.log(1-energy)) * sigma
    return int(2*np.ceil(r_min)+1)


def get_tuned_params(
    features_p, features_m, colors_p=['r/g', 'g/r', 'b/y'], colors_m=['w/b'],
    visual_degrees=visual_degrees, image_size=image_size
    ):
    """Gets SubcorticalBlock tuned hyperparameters.

    :param int features_p: Number of P cells
    :param int features_m: Number of M cells
    :param list colors_p: Color oposition scheme used in P cells, defaults to ['r/g', 'g/r', 'b/y']
    :param list colors_m: Color oposition scheme used in M cells, defaults to ['w/b']
    :param float visual_degrees: _description_, defaults to 7
    :param int image_size: Image size, defaults to 224
    :return dict: Tuned hyperparameters
    """
    global P_cell_params, M_cell_params
    if features_p > 0: assert features_p % len(colors_p) == 0
    if features_m > 0: assert features_m % len(colors_m) == 0

    res_factor = (image_size/visual_degrees) / (224/7)

    path_p_cell = Path(__file__).parent / "tuning" / "p_cell_hyperparams.csv"
    tuning_set = pd.read_csv(path_p_cell, index_col=0)
    P_cell_params = tuning_set.iloc[tuning_set["l2_error"].idxmin(), :].to_dict()
    
    path_m_cell = Path(__file__).parent / "tuning" / "m_cell_hyperparams.csv"
    tuning_set = pd.read_csv(path_m_cell, index_col=0)
    M_cell_params = tuning_set.iloc[tuning_set["l2_error"].idxmin(), :].to_dict()

    energy = .75

    color_mapping = {
        'r/g': np.array([[1,0,0],[0,1,0]], dtype=np.float32),  # R+/G- (center/surround)
        'g/r': np.array([[0,1,0],[1,0,0]], dtype=np.float32),  # G+/R-
        'b/y': np.array([[0,0,1],[.5,.5,0]], dtype=np.float32),  # B+/Y-
        'w/b': np.array([[1/3]*3,[1/3]*3], dtype=np.float32)  # ON/OFF
    }

    num_conv_p = bool(features_p)
    num_conv_m = bool(features_m)

    # DoG Radii
    rc = np.array([P_cell_params['rc_dog']]*num_conv_p+[M_cell_params['rc_dog']]*num_conv_m, dtype=np.float32)
    rs = np.array([P_cell_params['rs_dog']]*num_conv_p+[M_cell_params['rs_dog']]*num_conv_m, dtype=np.float32)
    rc *= res_factor
    rs *= res_factor

    # DoG Opponency
    opponency = np.zeros((features_p+features_m, 2, 3), dtype=np.float32)
    if features_p > 0:
        opponency[:features_p] = np.concatenate([color_mapping[c][None, ...] for c in colors_p])
        opponency[:features_p, 1] *= P_cell_params['ratio_dog']
    opponency[features_p:] = np.concatenate([color_mapping[c][None, ...] for c in colors_m])
    opponency[features_p:, 1] *= M_cell_params['ratio_dog']

    kernel_dog = np.array([get_kernel_size(r, energy=energy) for r in rs.tolist()], dtype=np.int16)
    
    # Contrast Normalization
    c50_cn = np.array([P_cell_params['c50_cn']]*num_conv_p+[M_cell_params['c50_cn']]*num_conv_m, dtype=np.float32)
    exp_cn = np.array([P_cell_params['exp_cn']]*num_conv_p+[M_cell_params['exp_cn']]*num_conv_m, dtype=np.float32)
    radius_cn = np.array([P_cell_params['radius_cn']]*num_conv_p+[M_cell_params['radius_cn']]*num_conv_m, dtype=np.float32)
    radius_cn *= res_factor
    kernel_cn = np.array([get_kernel_size(r, energy=energy) for r in radius_cn.tolist()], dtype=np.int16)

    k_exc = np.array([P_cell_params['k_exc']]*features_p+[M_cell_params['k_exc']]*features_m)

    ks_max = image_size * 2 - 1
    params = {
        'rc_dog': rc,
        'rs_dog': rs,
        'opponency_dog': opponency,
        'kernel_dog': np.minimum(kernel_dog, ks_max),
        'kernel_cn': np.minimum(kernel_cn, ks_max),
        'radius_cn': radius_cn,
        'c50_cn': c50_cn,
        'exp_cn': exp_cn,
        'k_exc': k_exc
    }
    
    return params


def get_grating_params(
    sf, angle=0, phase=0, contrast=1, radius=.5,
    image_size=image_size, visual_degrees=visual_degrees
    ) -> dict:
    """Gets drifting grating parameters.

    :param float sf: Spatial frequency
    :param int angle: Angle, defaults to 0
    :param int phase: Phase, defaults to 0
    :param int contrast: Contrast (0-1), defaults to 1
    :param float radius: Radius, defaults to .5
    :param int image_size: Image size, defaults to 224
    :param float visual_degrees: Visual degees, defaults to 7
    :return dict: Grating parameters
    """    
    ppd = image_size / visual_degrees  # pixels per FOV degree
    params = {
        'size': image_size,
        'radius': radius * ppd,
        'sf': sf/ppd,
        'theta': angle,
        'phase': phase,
        'contrast': contrast
    }
    return params


def generate_gabor_param(
    n_sc, n_cc, seed=0, rand_flag=False, sf_corr=0.75,
    sf_max=11.5, sf_min=0, diff_n=False, dnstd=0.22,
    # Additional parameters
    in_channels=3, color_prob=None,
    ):
    """Generate Gabor parameters.

    :param int n_sc: Number of simple cells
    :param int n_cc: Number of complex cells
    :param int seed: RNG Gabor seed, defaults to 0
    :param bool rand_flag: Random parameters flag, defaults to False
    :param float sf_corr: Spatial frequency correlation factor, defaults to 0.75
    :param float sf_max: Maximum spatial frequency, defaults to 11.5
    :param int sf_min: Minimum spatial frequency, defaults to 0
    :param bool diff_n: diff_n, defaults to False
    :param float dnstd: dnstd, defaults to 0.22
    :param int in_channels: Number of input channels to the GFB, defaults to 3
    :param list color_prob: GFB color input, defaults to None
    :return tuple: GFB parameters
    """    
    features = n_sc + n_cc

    # Generates random sample
    np.random.seed(seed)

    phase_bins = np.array([0, 360])
    phase_dist = np.array([1])

    if rand_flag:
        #print('Uniform gabor parameters')
        ori_bins = np.array([0, 180])
        ori_dist = np.array([1])

        # nx_bins = np.array([0.1, 10**0.2])
        nx_bins = np.array([0.1, 10**0])
        nx_dist = np.array([1])

        # ny_bins = np.array([0.1, 10**0.2]
        ny_bins = np.array([0.1, 10**0])
        ny_dist = np.array([1])

        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8, 11.2])
        sf_s_dist = np.array([1,  1,  1, 1, 1, 1, 1, 1, 1])
        sf_c_dist = np.array([1,  1,  1, 1, 1, 1, 1, 1, 1])

    else:
        #print('Neuronal distributions gabor parameters')
        # DeValois 1982a
        ori_bins = np.array([-22.5, 22.5, 67.5, 112.5, 157.5])
        ori_dist = np.array([66, 49, 77, 54])
        # ori_dist = np.array([110, 83, 100, 92])
        ori_dist = ori_dist / ori_dist.sum()

        # Ringach 2002b
        # nx_bins = np.logspace(-1, 0.2, 6, base=10)
        # ny_bins = np.logspace(-1, 0.2, 6, base=10)
        nx_bins = np.logspace(-1, 0., 5, base=10)
        ny_bins = np.logspace(-1, 0., 5, base=10)
        n_joint_dist = np.array([[2.,  0.,  1.,  0.],
                                 [8.,  9.,  4.,  1.],
                                 [1.,  2., 19., 17.],
                                 [0.,  0.,  1.,  7.]])
        # n_joint_dist = np.array([[2.,  0.,  1.,  0.,  0.],
        #                          [8.,  9.,  4.,  1.,  0.],
        #                          [1.,  2., 19., 17.,  3.],
        #                          [0.,  0.,  1.,  7.,  4.],
        #                          [0.,  0.,  0.,  0.,  0.]])
        n_joint_dist = n_joint_dist / n_joint_dist.sum()
        nx_dist = n_joint_dist.sum(axis=1)
        nx_dist = nx_dist / nx_dist.sum()
        ny_dist_marg = n_joint_dist / n_joint_dist.sum(axis=1, keepdims=True)

        # DeValois 1982b
        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8, 11.2])
        # foveal only
        sf_s_dist = np.array([4, 4, 8, 25, 33, 26, 28, 12, 8])
        sf_c_dist = np.array([0, 0, 9, 9, 7, 10, 23, 12, 14])
        # foveal + parafoveal
        # sf_s_dist = np.array([8, 14, 20, 43, 40, 44, 31, 16, 8])
        # sf_c_dist = np.array([2, 1, 11, 14, 22, 23, 32, 15, 16])


    phase = sample_dist(phase_dist, phase_bins, features)
    ori = sample_dist(ori_dist, ori_bins, features)

    # ori[ori < 0] = ori[ori < 0] + 180
    
    sfmax_ind = np.where(sf_bins <= sf_max)[0][-1]
    sfmin_ind = np.where(sf_bins >= sf_min)[0][0]

    sf_bins = sf_bins[sfmin_ind:sfmax_ind+1]
    sf_s_dist = sf_s_dist[sfmin_ind:sfmax_ind]
    sf_c_dist = sf_c_dist[sfmin_ind:sfmax_ind]

    sf_s_dist = sf_s_dist / sf_s_dist.sum()
    sf_c_dist = sf_c_dist / sf_c_dist.sum()

    cov_mat = np.array([[1, sf_corr], [sf_corr, 1]])

    if rand_flag:   # Uniform
        samps = np.random.multivariate_normal([0, 0], cov_mat, features)
        samps_cdf = stats.norm.cdf(samps)

        nx = np.interp(samps_cdf[:,0], np.hstack(([0], nx_dist.cumsum())), np.log10(nx_bins))
        nx = 10**nx

        if diff_n: 
            ny = sample_dist(ny_dist, ny_bins, features, scale='log10')
        else:
            ny = 10**(np.random.normal(np.log10(nx), dnstd))
            ny[ny<0.1] = 0.1
            ny[ny>1] = 1
            # ny = nx

        sf = np.interp(samps_cdf[:,1], np.hstack(([0], sf_s_dist.cumsum())), np.log2(sf_bins))
        sf = 2**sf

        # if n_sc > 0:
        #     sf_s = sample_dist(sf_s_dist, sf_bins, n_sc, scale='log2')
        # else:
        #     sf_s = np.array([])
        # if n_cc > 0:
        #     sf_c = sample_dist(sf_c_dist, sf_bins, n_cc, scale='log2')
        # else:
        #     sf_c = np.array([])
        # sf = np.concatenate((sf_s, sf_c))

        # nx = sample_dist(nx_dist, nx_bins, features, scale='log10')
    else:   # Biological

        if n_sc > 0:
            samps = np.random.multivariate_normal([0, 0], cov_mat, n_sc)
            samps_cdf = stats.norm.cdf(samps)

            nx_s = np.interp(samps_cdf[:,0], np.hstack(([0], nx_dist.cumsum())), np.log10(nx_bins))
            nx_s = 10**nx_s

            ny_samp = np.random.rand(n_sc)
            ny_s = np.zeros(n_sc)
            for samp_ind, nx_samp in enumerate(nx_s):
                bin_id = np.argwhere(nx_bins < nx_samp)[-1]
                ny_s[samp_ind] = np.interp(ny_samp[samp_ind], np.hstack(([0], ny_dist_marg[bin_id, :].cumsum())),
                                                 np.log10(ny_bins))
            ny_s = 10**ny_s

            sf_s = np.interp(samps_cdf[:,1], np.hstack(([0], sf_s_dist.cumsum())), np.log2(sf_bins))
            sf_s = 2**sf_s
        else:
            nx_s = np.array([])
            ny_s = np.array([])
            sf_s = np.array([])

        if n_cc > 0:
            samps = np.random.multivariate_normal([0, 0], cov_mat, n_cc)
            samps_cdf = stats.norm.cdf(samps)

            nx_c = np.interp(samps_cdf[:,0], np.hstack(([0], nx_dist.cumsum())), np.log10(nx_bins))
            nx_c = 10**nx_c

            ny_samp = np.random.rand(n_cc)
            ny_c = np.zeros(n_cc)
            for samp_ind, nx_samp in enumerate(nx_c):
                bin_id = np.argwhere(nx_bins < nx_samp)[-1]
                ny_c[samp_ind] = np.interp(ny_samp[samp_ind], np.hstack(([0], ny_dist_marg[bin_id, :].cumsum())),
                                                 np.log10(ny_bins))
            ny_c = 10**ny_c

            sf_c = np.interp(samps_cdf[:,1], np.hstack(([0], sf_c_dist.cumsum())), np.log2(sf_bins))
            sf_c = 2**sf_c
        else:
            nx_c = np.array([])
            ny_c = np.array([])
            sf_c = np.array([])

        nx = np.concatenate((nx_s, nx_c))
        ny = np.concatenate((ny_s, ny_c))
        sf = np.concatenate((sf_s, sf_c))

    # Generate an array of size 'features', with values either 0,1,2, (pseudo)randomly set
    if color_prob and in_channels > 1:
        counts = [round(p*features) for p in color_prob]
        counts[2] += 512 - sum(counts)  # Y/B gets one less
        color = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
        np.random.shuffle(color)
    else:
        color = np.random.randint(low=0, high=in_channels, size=features, dtype=np.int8)

    return sf, ori, phase, nx, ny, color
