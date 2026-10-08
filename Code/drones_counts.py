"""
This script contains the script to generate figure 2 in the paper "Striking acoustic signals: quantifying low-frequency musical drones in 
conspiracy documentaries" (Tomasini et al.) submitted to DHNB 2027. 

The core function is `count_hits()` which returns the number of hits for a film soundtrack given a certain threshold. This is then looped
upon for the five documentaries for 20 threshold values between 0.04 and 0.07. For a question of copyright, we cannot publish the sound data used
for the paper, but they are available upon request.

The algorithm makes use of the package libfmp developed by Meinard Müller and Frank Zalkow. 
"""

import datetime
import librosa
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema 
import matplotlib.pyplot as plt

import libfmp.b
import libfmp.c7


def time_string_from_float(time: float):
    time_string = str(datetime.timedelta(minutes=time)).split('.')[0]

    return time_string


def compute_cens_from_file(fn_wav, Fs=22050, N=4410, H=2205, ell=21, d=5):
    """Compute CENS features from file

    Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        fn_wav (str): Filename of wav file
        Fs (scalar): Feature rate of wav file (Default value = 22050)
        N (int): Window size for STFT (Default value = 4410)
        H (int): Hop size for STFT (Default value = 2205)
        ell (int): Smoothing length (Default value = 21)
        d (int): Downsampling factor (Default value = 5)

    Returns:
        X_CENS (np.ndarray): CENS features
        L (int): Length of CENS feature sequence
        Fs_CENS (scalar): Feature rate of CENS features
        x_duration (float): Duration (seconds) of wav file
    """
    x, Fs = librosa.load(fn_wav, sr=Fs)
    x_duration = x.shape[0] / Fs
    X_chroma = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
    X_CENS, Fs_CENS = libfmp.c7.compute_cens_from_chromagram(X_chroma, Fs=Fs/H, ell=ell, d=d)
    L = X_CENS.shape[1]
    return X_CENS, L, Fs_CENS, x_duration

def compute_matching_function_dtw(X, Y, stepsize=2):
    """Compute CENS features from file

    Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        X (np.ndarray): Query feature sequence (given as K x N matrix)
        Y (np.ndarray): Database feature sequence (given as K x M matrix)
        stepsize (int): Parameter for step size condition (1 or 2) (Default value = 2)

    Returns:
        Delta (np.ndarray): DTW-based matching function
        C (np.ndarray): Cost matrix
        D (np.ndarray): Accumulated cost matrix
    """
    C = libfmp.c7.cost_matrix_dot(X, Y)
    if stepsize == 1:
        D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw(C)
    if stepsize == 2:
        D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw_21(C)
    N, M = C.shape
    Delta = D[-1, :] / N
    return Delta, C, D

def matches_dtw(pos, D, stepsize=2):
    """Derives matches from positions for DTW-based strategy

    Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        pos (np.ndarray): End positions of matches
        D (np.ndarray): Accumulated cost matrix
        stepsize (int): Parameter for step size condition (1 or 2) (Default value = 2)

    Returns:
        matches (np.ndarray): Array containing matches (start, end)
    """
    matches = np.zeros((len(pos), 2)).astype(int)
    for k in range(len(pos)):
        t = pos[k]
        matches[k, 1] = t
        if stepsize == 1:
            P = libfmp.c7.compute_optimal_warping_path_subsequence_dtw(D, m=t)
        if stepsize == 2:
            P = libfmp.c7.compute_optimal_warping_path_subsequence_dtw_21(D, m=t)
        s = P[0, 1]
        matches[k, 0] = s
    return matches
    
def compute_plot_matching_function_DTW(fn_wav_X, fn_wav_Y, 
                                       ell=21, d=5, stepsize=2, tau=0.2, num=5, ylim=[0,0.35], create_plot=False):
    #ann, _ = libfmp.c4.read_structure_annotation(fn_ann)
    #color_ann = {'Theme': [0, 0, 1, 0.1], 'Match': [0, 0, 1, 0.2]}
    X, N, Fs_X, x_duration = compute_cens_from_file(fn_wav_X, ell=ell, d=d)
    Y, M, Fs_Y, y_duration = compute_cens_from_file(fn_wav_Y, ell=ell, d=d)
    Delta, C, D = compute_matching_function_dtw(X, Y, stepsize=stepsize)
    pos = libfmp.c7.mininma_from_matching_function(Delta, rho=2*N//3, tau=tau, num=num)
    matches = matches_dtw(pos, D, stepsize=stepsize)
    if create_plot:
        fig, ax = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1], 
                                                'height_ratios': [1, 1]}, figsize=(8, 4))
        cmap = libfmp.b.compressed_gray_cmap(alpha=-10, reverse=True)
        libfmp.b.plot_matrix(C, Fs=Fs_X, ax=[ax[0]], ylabel='Time (seconds)',
                            # title='Cost matrix $C$ with ground truth annotations (blue rectangles)', 
                            colorbar=False, cmap=cmap)
    # libfmp.b.plot_segments_overlay(ann, ax=ax[0], alpha=0.2, time_max=y_duration, 
    #                                colors = color_ann, print_labels=False)a
    
        title = r'Matching function $\Delta_\mathrm{DTW}$ with matches (red rectangles)'
        libfmp.b.plot_signal(Delta,  ax=ax[1], Fs=Fs_X, color='k', title=title, ylim=ylim)
        ax[1].grid()
        libfmp.c7.plot_matches(ax[1], matches, Delta, Fs=Fs_X, s_marker='', t_marker='o')
        plt.tight_layout()  
        plt.show()
    #    print(y_duration, Fs_Y)
    
    return Delta


def give_drones_hits(movie_path, drone_path, threshold, create_full_plot = False, create_secondary_plot = False):
    D_dtw = compute_plot_matching_function_DTW(drone_path, movie_path, stepsize = 1, ylim=[0, 0.5], tau=3, create_plot=create_full_plot)
    local_minima_idx = argrelextrema(D_dtw, np.less, order=50)[0]
    indeces_below_threshold_bool = D_dtw[local_minima_idx] < threshold
    indeces_below_threshold = local_minima_idx[indeces_below_threshold_bool]
    all_minima_D_dtw = D_dtw[indeces_below_threshold]
    if create_secondary_plot:
        plt.plot(D_dtw)
        plt.ylim(0, 0.25)
        plt.plot(indeces_below_threshold, all_minima_D_dtw, 'ro')

    minima_in_sec = indeces_below_threshold / 2
    minima_in_minutes = minima_in_sec / 60

    print(f"The algorithm found matches at: {minima_in_minutes}")
    
    return minima_in_minutes


def count_hits(movie_path, threshold):
    all_minima = []
    for file in os.listdir('./drones_samples_5s/'):
        minima_for_this_drone = give_drones_hits(movie_path, f'./drones_samples_5s/{file}', threshold)
        for element in minima_for_this_drone:
            all_minima.append(element)

    total_hits = len(all_minima)

    return total_hits


paths = ['./SoundData/conspiracyoflies_files/conspiracyoflies_(Instrumental).flac', 
         './SoundData/loosechange_files/loose_change_(Instrumental).flac',
         './SoundData/ruletheworld_files/rule_the_world_(Instrumental).flac',
         './SoundData/wargames_files/wargames_(Instrumental)_model_bs_roformer_ep_317_sdr_12.flac',
         './SoundData/zeitgeist_files/zeitgeist_background.flac']

thresholds = np.linspace(0.04, 0.07, num=20)
labels = ['Conspiracy of Lies', 'Loose Change', 'Rule The World', 'War Games', 'Zeitgeist']
i_lab = 4

plt.figure()

for path in paths:
    label = labels[0]
    
    counts = []
    for threshold in thresholds:
        count = count_hits(path, threshold)
        counts.append(count)

    plt.plot(thresholds, counts, label=label)
    np.savetxt(f'./thresholds_{i_lab}.txt', thresholds, fmt="%s")
    np.savetxt(f'./counts_{i_lab}.txt', counts, fmt="%s")

    i_lab += 1



plt.legend()
plt.show()
plt.savefig('../Figures/hits_by_threshold.png')

