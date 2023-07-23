import tifffile
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import find_objects, gaussian_filter, label
import pandas as pd
import os
import cv2
from sklearn import linear_model
from sklearn.metrics import r2_score
from skimage.filters import apply_hysteresis_threshold

from scipy.signal import butter, filtfilt
import pickle
import json
import re
from datetime import datetime
plt.rcParams.update({'figure.max_open_warning': 0})
from scipy.ndimage import find_objects, gaussian_filter, label
from operator import itemgetter
from itertools import groupby
from sklearn.neighbors import KernelDensity
import csv
from unidip import UniDip

# functions
def create_directory(path):
    """
    Create directory if it does not exists.
    :param path: path of the directory
    :return: False if directory already exists, True otherwise
    """
    if os.path.exists(path):
        return False
    os.makedirs(path)
    return True


def heart_rate_quality_filter(fish_dir, fish, date, trial_index, ROI, duration=1300, starting_sigma=4, max_sigma=13,
                              rolling_index=9, std_threshold=0.55):
    """
    run averaged pixel activity of a trial and check its quality, also return quality boolean and trace as a list,
    also generate the heart rate plot
    """
    # Read the video
    heart_video = glob.glob(fish_dir + '*heart_' + str(trial_index) + '.*')
    video = cv2.VideoCapture(heart_video[0])
    # loop to read frame by frame and record intensity of the trial
    averaged_data = np.zeros(duration)
    npy_temp = np.zeros(duration)

    ret = True
    for index in range(0, duration):
        ret, img = video.read()
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            crop = gray[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2]]
            intensity = np.average(crop)
            averaged_data[index] = intensity

    sigma = starting_sigma
    while sigma < max_sigma:
        max_min_idx = True
        data = gaussian_filter(averaged_data, sigma=sigma)
        diffed = np.diff(data)
        smoothed = pd.Series(diffed).rolling(rolling_index, min_periods=0, center=True).mean().values
        signed = np.sign(diffed)
        second_diff = np.diff(signed)
        local_maxima = np.where(second_diff < 0)[0] + 1
        local_minima = np.where(second_diff > 0)[0] + 1

        # IBI calculation
        for j in range(1, local_maxima.shape[0]):
            npy_temp[local_maxima[j - 1]:local_maxima[j]] = 100 / (local_maxima[j] - local_maxima[j - 1])
        npy_temp[0:local_maxima[0]] = 100 / (local_maxima[1] - local_maxima[0])
        npy_temp[local_maxima[j]:] = 100 / (local_maxima[-1] - local_maxima[-2])

        #     locals()['npy_'+fish+'_'+date][i-1,0:local_maxima[0]] = 100/(local_maxima[1]-local_maxima[0])
        if np.std(npy_temp) < std_threshold:
            for max_i in range(1, local_maxima.shape[0]):
                if data[local_maxima[max_i]] < data[local_maxima[max_i - 1]] / 4 + data[
                    local_minima[np.where(local_maxima[max_i - 1] < local_minima)[0][0]]] * 3 / 4:
                    max_min_idx = False
            for min_i in range(1, local_minima.shape[0]):
                if data[local_minima[min_i]] > data[
                    local_maxima[np.where(local_minima[min_i - 1] < local_maxima)[0][0]]] * 3 / 4 + data[
                    local_minima[min_i - 1]] / 4:
                    max_min_idx = False
            for max_i in range(0, local_maxima.shape[0] - 1):
                if data[local_maxima[max_i]] < data[local_maxima[max_i + 1]] / 4 + data[
                    local_minima[np.where(local_maxima[max_i] < local_minima)[0][0]]] * 3 / 4:
                    max_min_idx = False
            for min_i in range(0, local_minima.shape[0] - 1):
                if data[local_minima[min_i]] > data[
                    local_maxima[np.where(local_minima[min_i] < local_maxima)[0][0]]] * 3 / 4 + data[
                    local_minima[min_i + 1]] / 4:
                    max_min_idx = False
            #         normalized_data = (data-np.min(data))/(np.max(data)-np.min(data))
            #         std_sum_min_max_temp.append(np.std(normalized_data[local_maxima])+np.std(normalized_data[local_minima]))
            if max_min_idx:
                plt.figure(figsize=(10, 4))
                plt.plot(data)
                print(np.std(npy_temp))
                for j in local_maxima:
                    plt.axvline(j, color='r', alpha=0.2)
                plt.title('Pixel Activity of T' + str(trial_index) + '_' + fish + '_' + date)
                plt.ylabel('Pixel Intensity (a.u)')
                plt.xlabel('Frame')
                plt.savefig(fish_dir + date + '_' + fish + '_T' + str(trial_index) + '.png')
                plt.show()
            else:
                #                 no_idx.append(trial)
                plt.figure(figsize=(10, 4))
                plt.plot(data, color='g')
                print(np.std(npy_temp))
                for j in local_maxima:
                    plt.axvline(j, color='r', alpha=0.2)
                plt.title('Pixel Activity of T' + str(trial_index) + '_' + fish + '_' + date)
                plt.ylabel('Pixel Intensity (a.u)')
                plt.xlabel('Frame')
                plt.savefig(fish_dir + date + '_' + fish + '_T' + str(trial_index) + '.png')
                plt.show()
        else:
            max_min_idx = False
            #             no_idx.append(trial)
            plt.figure(figsize=(10, 4))
            plt.plot(data, color='k')
            print(np.std(npy_temp))
            for j in local_maxima:
                plt.axvline(j, color='r', alpha=0.2)
            plt.title('Pixel Activity of T' + str(trial_index) + '_' + fish + '_' + date)
            plt.ylabel('Pixel Intensity (a.u)')
            plt.xlabel('Frame')
            plt.savefig(fish_dir + date + '_' + fish + '_T' + str(trial_index) + '.png')
            plt.show()

        sigma = sigma + 1
        if max_min_idx:
            video.release()
            return max_min_idx, npy_temp
    video.release()
    return max_min_idx, npy_temp


def create_brady_info(fish_dir, ROI):
    # first create the directory and bradyinfo excel
    fish = fish_dir[-9:-1]
    print(fish)
    date = fish_dir[-27:-19]
    print(date)
    vsinfo_dir = glob.glob(fish_dir + 'vsinfo*')
    vsinfo = pd.read_excel(vsinfo_dir[0])
    expinfo = pd.read_excel(vsinfo_dir[0], sheet_name='ExpInfo')
    duration = expinfo.Total_Trial_Duration[0] * expinfo.Stimulus_Frame_Rate[0]
    bradyinfo = pd.DataFrame()
    bradyinfo['fish index'] = [fish] * vsinfo.shape[0]
    bradyinfo['trial index'] = range(1, vsinfo.shape[0] + 1)
    bradyinfo['roi'] = [ROI] * vsinfo.shape[0]
    # run quality filter for every trial
    heart_rate_array = np.zeros((vsinfo.shape[0], duration))
    quality_list = []
    for trial in range(1, vsinfo.shape[0] + 1):
        quality, trace = heart_rate_quality_filter(fish_dir, fish, date, trial, ROI)
        heart_rate_array[trial - 1, :] = trace
        quality_list.append(quality)

        fig, (ax1) = plt.subplots(1, 1, figsize=(16, 6), sharex=True, sharey=True)
        ax1.plot(trace, linewidth=3, c='b')
        x1 = np.arange(900, 1000)
        plt.fill_between(x1, 4, -4, linewidth=1, color='lightcoral', alpha=0.2)
        plt.ylim(0, 4)
        plt.xticks([0, 900, 1000, 1300])  # Set label locations.
        plt.title('Heart Rate of T' + str(trial))
        plt.ylabel('Normalized Heart Rate')
        plt.xlabel('frame')
        plt.show()
        fig.savefig(fish_dir + 'Heart Rate of ' + date + '_' + fish + '_T' + str(trial) + '.png', dpi=300)

    bradyinfo['Video_Quality'] = quality_list
    heart_rate_dataframe = pd.DataFrame(heart_rate_array)

    bradyinfo_path = fish_dir + 'Bradyinfo_' + date + '_' + fish + '.xlsx'
    writer = pd.ExcelWriter(bradyinfo_path, engine='xlsxwriter')
    bradyinfo.to_excel(writer, sheet_name='Bradyinfo', index=False)
    heart_rate_dataframe.to_excel(writer, sheet_name='heart_rate_trace', index=False)
    writer.save()


def QC_fixer(fish_dir, trial_index, ROI_list):
    # loop to try all ROIs
    print(trial_index)
    fish = fish_dir[-9:-1]
    print(fish)
    date = fish_dir[-27:-19]
    print(date)
    quality = False
    ROI_G = ROI_list[0]
    for ROI in ROI_list:
        if quality == False:
            quality, trace = heart_rate_quality_filter(fish_dir, fish, date, trial_index, ROI)
            if quality == True:
                ROI_G = ROI
    return quality, trace, ROI_G


def make_regressor(duration, pre_cutoff, onset_time, on_duration, stimulus_duration):
    fig, (ax1) = plt.subplots(1, 1, figsize=(16, 6), sharex=True, sharey=True)
    regressor = np.ones(duration - pre_cutoff)
    regressor[onset_time - pre_cutoff:onset_time - pre_cutoff + on_duration] = 0
    ax1.plot(regressor, linewidth=3, c='b')
    x1 = np.arange(onset_time - pre_cutoff, onset_time - pre_cutoff + stimulus_duration)
    plt.fill_between(x1, 2, -2, linewidth=1, color='lightcoral', alpha=0.2)
    # x2 = numpy.arange(1200,1500)
    # plt.fill_between(x2,2,-2,linewidth = 1,color = 'cyan',alpha = 0.2)
    plt.ylim(-0.1, 2)
    # plt.axhline(y = 0.8,xmin = 0, xmax =1 , alpha = 0.4)
    # plt.xticks([0,900,1000,1100,1200,1300,1400,1500,0])  # Set label locations.
    #     plt.title('lum trial:Heart Rate of T' + str(i))
    plt.ylabel('Normalized Heart Rate')
    plt.xlabel('frame')
    plt.show()

    return regressor


def brady_linear_regression(regressor, heart_rate_trace, pre_cutoff, onset_time):
    df = pd.DataFrame(heart_rate_trace[pre_cutoff:] / np.mean(heart_rate_trace[pre_cutoff:onset_time]))
    X = pd.DataFrame(regressor)
    regr = linear_model.LinearRegression()

    x = regr.fit(X, df)
    coeff = regr.coef_[0][0]
    intercept = x.intercept_[0]
    r2 = r2_score(df, x.predict(X), multioutput='raw_values')
    r2 = r2[0]

    brady = False

    if coeff > 0.09 and r2 > 0.13:
        brady = True

    return coeff, intercept, r2, brady

def strike_detection(fish_dir, trial_index, ROI, date, fish, duration=1250, low_thresh=2, high_thresh=4, sigma=4):
    # Read the video
    heart_video = glob.glob(fish_dir + '*heart_' + str(trial_index) + '.*')
    video = cv2.VideoCapture(heart_video[0])
    # loop to read frame by frame and record intensity of the trial
    averaged_data = np.zeros(duration)

    ret = True
    for index in range(0, duration):
        ret, img = video.read()
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            crop = gray[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2]]
            intensity = np.average(crop)
            averaged_data[index] = intensity
    # detect bouts
    d_curve = np.gradient(averaged_data)
    abs_d_curve = np.abs(d_curve)
    filtered_abs_d_curve = gaussian_filter(abs_d_curve, sigma)
    thresholded = apply_hysteresis_threshold(filtered_abs_d_curve, low_thresh, high_thresh)
    bouts = [(i[0].start, i[0].stop) for i in find_objects(label(thresholded)[0])]

    plt.figure(figsize=(10, 4))
    plt.plot(abs_d_curve, color='b')
    plt.plot(filtered_abs_d_curve, color='g')
    plt.plot(thresholded, color='r')
    plt.legend(['gradient', 'filter', 'threshold'])
    plt.title('Strike Detection of T' + str(trial_index) + '_' + fish + '_' + date)
    plt.ylabel('Pixel Intensity (a.u)')
    plt.xlabel('Frame')
    plt.ylim(0, 15)
    plt.savefig(fish_dir + date + '_' + fish + '_T' + str(trial_index) + '_Strike_Detection.png')
    plt.show()

    for bout in bouts:
        if bout[0] >= 850 and bout[0] <= 1000:
            return True, bouts
    else:
        return False, bouts


def strike_screening(fish_dir, ROI, duration=1250, low_thresh=2, high_thresh=4, sigma=4):
    # first load vsinfo and bradyinfo
    fish = fish_dir[-9:-1]
    print(fish)
    date = fish_dir[-27:-19]
    print(date)
    vsinfo_dir = glob.glob(fish_dir + 'vsinfo*')
    vsinfo = pd.read_excel(vsinfo_dir[0])
    expinfo = pd.read_excel(vsinfo_dir[0], sheet_name='ExpInfo')
    duration = expinfo.Total_Trial_Duration[0] * expinfo.Stimulus_Frame_Rate[0]
    bradyinfo_path = glob.glob(fish_dir + '/Brady*')
    bradyinfo = pd.read_excel(bradyinfo_path[0])
    heart_rate_dataframe = pd.read_excel(bradyinfo_path[0], sheet_name='heart_rate_trace')
    # run quality filter for every trial
    strike_list = []
    strike_bool_list = []
    for trial in range(1, vsinfo.shape[0] + 1):
        strike, strike_bool = strike_detection(fish_dir, trial, ROI, date, fish,duration, low_thresh, high_thresh, sigma)
        strike_list.append(strike)
        strike_bool_list.append(strike_bool)
    bradyinfo['Strike_ROI'] = [ROI] * vsinfo.shape[0]
    bradyinfo['Strike'] = strike_bool_list
    bradyinfo['Strike_Bouts'] = strike_list

    writer = pd.ExcelWriter(bradyinfo_path[0], engine='xlsxwriter')
    bradyinfo.to_excel(writer, sheet_name='Bradyinfo', index=False)
    heart_rate_dataframe.to_excel(writer, sheet_name='heart_rate_trace', index=False)
    writer.save()


def create_brady_info_pixelwise(fish_dir, ROI):
    # first create the directory and bradyinfo excel
    fish = fish_dir[-9:-1]
    print(fish)
    date = fish_dir[-27:-19]
    print(date)
    vsinfo_dir = glob.glob(fish_dir + 'vsinfo*')
    vsinfo = pd.read_excel(vsinfo_dir[0])
    expinfo = pd.read_excel(vsinfo_dir[0], sheet_name='ExpInfo')
    duration = expinfo.Total_Trial_Duration[0] * expinfo.Stimulus_Frame_Rate[0]
    bradyinfo = pd.DataFrame()
    bradyinfo['fish index'] = [fish] * vsinfo.shape[0]
    bradyinfo['trial index'] = range(1, vsinfo.shape[0] + 1)
    bradyinfo['roi'] = [ROI] * vsinfo.shape[0]
    # run quality filter for every trial
    heart_rate_array = np.zeros((vsinfo.shape[0], duration))
    quality_list = []
    for trial in range(1, vsinfo.shape[0] + 1):
        quality, trace = heart_rate_pixelwise_trace(fish_dir, fish, date, trial, ROI)
        heart_rate_array[trial - 1, :] = trace
        quality_list.append(quality)
    bradyinfo['Good_Pixel_Number'] = quality_list
    heart_rate_dataframe = pd.DataFrame(heart_rate_array)

    bradyinfo_path = fish_dir + 'Bradyinfo_' + date + '_' + fish + '.xlsx'
    writer = pd.ExcelWriter(bradyinfo_path, engine='xlsxwriter')
    bradyinfo.to_excel(writer, sheet_name='Bradyinfo', index=False)
    heart_rate_dataframe.to_excel(writer, sheet_name='heart_rate_trace', index=False)
    writer.save()


def heart_rate_pixelwise_trace(fish_dir, fish, date, trial_index, ROI, duration=1300):
    """
    run averaged pixel activity of a trial and check its quality, also return quality boolean and trace as a list,
    also generate the heart rate plot
    """
    # Read the video
    heart_video = glob.glob(fish_dir + '*heart_' + str(trial_index) + '.*')
    video = cv2.VideoCapture(heart_video[0])
    # loop to read frame by frame get a 3d volumn
    video_array = np.zeros((duration, ROI[3], ROI[2]))

    ret = True
    for index in range(0, duration):
        ret, img = video.read()
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            crop = gray[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2]]
            video_array[index, :, :] = crop
    video.release()
    good_pixel_array = np.zeros((duration, ROI[3], ROI[2]))

    for x in range(0, ROI[3]):
        for y in range(0, ROI[2]):
            pixel_trace = video_array[:, x, y]
            pixel_index = [x, y]
            good_pixel_array[:, x, y] = pixel_heart_rate(fish_dir, fish, date, trial_index, pixel_index, pixel_trace)

    averaged_heart_rate = np.zeros(duration)
    for frame in range(0, duration):
        pixels_of_frame = good_pixel_array[frame, :, :]
        good_pixels = pixels_of_frame.ravel()[np.flatnonzero(pixels_of_frame)].tolist()
        averaged_heart_rate[frame] = np.average(good_pixels)

    plt.figure(figsize=(12, 7))
    plt.plot(averaged_heart_rate[200:])
    x1 = np.arange(700, 800)
    plt.fill_between(x1, 4, -4, linewidth=1, color='violet', alpha=0.3)
    plt.ylim(0, 4)
    print(np.std(averaged_heart_rate))
    # plt.title('Averaged Heart Rate of T' + str(trial_index) + '_' + fish + '_' + date,fontsize = 20)
    plt.ylabel('Heart Rate per Minute',fontsize=26)
    plt.yticks([0,0.5,1,1.5,2,2.5,3,3.5,4],
               [0, 30,60,90,120,150,180,210,240],fontsize=22)
    plt.xlabel('Time(Seconds)', fontsize=26)
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], fontsize=22)
    plt.savefig(fish_dir + 'Averaged Heart Rate of_' + date + '_' + fish + '_T' + str(trial_index) + '.png',bbox_inches="tight",dpi=300)
    #     plt.show()

    return len(good_pixels), averaged_heart_rate


def pixel_heart_rate(fish_dir, fish, date, trial_index, pixel_index, pixel_trace, duration=1300, starting_sigma=4,
                     max_sigma=13, rolling_index=9, std_threshold=0.55):
    #     print(pixel_index)
    npy_temp = np.zeros(duration)
    sigma = starting_sigma
    while sigma < max_sigma:
        max_min_idx = True
        data = gaussian_filter(pixel_trace, sigma=sigma)
        diffed = np.diff(data)
        smoothed = pd.Series(diffed).rolling(rolling_index, min_periods=0, center=True).mean().values
        signed = np.sign(diffed)
        second_diff = np.diff(signed)
        local_maxima = np.where(second_diff < 0)[0] + 1
        local_minima = np.where(second_diff > 0)[0] + 1

        local_max_screen = []
        local_min_screen = []

        for max_i in range(1, local_maxima.shape[0]):
            min_between = local_minima[np.where(local_maxima[max_i - 1] < local_minima)]
            min_between = min_between[np.where(local_maxima[max_i] > min_between)]

            if len(min_between) == 1:
                local_min_screen.append(min_between[0])
            if len(min_between) > 1:
                local_min_screen.append(int(np.average(min_between)))

        for min_i in range(1, local_minima.shape[0]):
            max_between = local_maxima[np.where(local_minima[min_i - 1] < local_maxima)]
            max_between = max_between[np.where(local_minima[min_i] > max_between)]

            if len(max_between) == 1:
                local_max_screen.append(max_between[0])
            if len(max_between) > 1:
                local_max_screen.append(int(np.average(max_between)))

        local_maxima = np.asarray(local_max_screen)
        local_minima = np.asarray(local_min_screen)

        if local_maxima.shape[0] < 5 or local_minima.shape[0] < 5:
            return np.zeros(duration)

        #         plt.figure(figsize=(10, 4))
        #         plt.plot(data[0:1300], color='k')
        # #         print(np.std(npy_temp))
        #         for j in local_maxima:
        #             plt.axvline(j, color='r', alpha=0.2)
        #         for j in local_minima:
        #             plt.axvline(j, color='b', alpha=0.2)

        #         plt.title('Pixel Activity of T' + str(trial_index) + '_' + fish + '_' + date + 'Pixel_No_' + str(pixel_index))
        #         plt.ylabel('Pixel Intensity (a.u)')
        #         plt.xlabel('Frame')
        #         plt.show()

        # IBI calculation
        for j in range(1, len(local_maxima)):
            npy_temp[local_maxima[j - 1]:local_maxima[j]] = 100 / (local_maxima[j] - local_maxima[j - 1])
        npy_temp[0:local_maxima[0]] = 100 / (local_maxima[1] - local_maxima[0])
        npy_temp[local_maxima[j]:] = 100 / (local_maxima[-1] - local_maxima[-2])

        #     locals()['npy_'+fish+'_'+date][i-1,0:local_maxima[0]] = 100/(local_maxima[1]-local_maxima[0])
        if np.std(npy_temp) < std_threshold:
            for max_i in range(1, local_maxima.shape[0]):
                if data[local_maxima[max_i]] < data[local_maxima[max_i - 1]] / 4 + data[
                    local_minima[np.where(local_maxima[max_i - 1] < local_minima)[0][0]]] * 3 / 4:
                    max_min_idx = False
            for min_i in range(1, local_minima.shape[0]):
                if data[local_minima[min_i]] > data[
                    local_maxima[np.where(local_minima[min_i - 1] < local_maxima)[0][0]]] * 3 / 4 + data[
                    local_minima[min_i - 1]] / 4:
                    max_min_idx = False
            for max_i in range(0, local_maxima.shape[0] - 1):
                if data[local_maxima[max_i]] < data[local_maxima[max_i + 1]] / 4 + data[
                    local_minima[np.where(local_maxima[max_i] < local_minima)[0][0]]] * 3 / 4:
                    max_min_idx = False
            for min_i in range(0, local_minima.shape[0] - 1):
                if data[local_minima[min_i]] > data[
                    local_maxima[np.where(local_minima[min_i] < local_maxima)[0][0]]] * 3 / 4 + data[
                    local_minima[min_i + 1]] / 4:
                    max_min_idx = False

        #             if max_min_idx:
        #                 plt.figure(figsize=(10, 4))
        #                 plt.plot(data[0:1250])
        #                 print(np.std(npy_temp))
        #                 for j in local_maxima:
        #                     plt.axvline(j, color='r', alpha=0.2)
        #                 plt.title('Pixel Activity of T' + str(trial_index) + '_' + fish + '_' + date + 'Pixel_No_' + str(pixel_index))
        #                 plt.ylabel('Pixel Intensity (a.u)')
        #                 plt.xlabel('Frame')
        #                 plt.show()
        #             else:
        #                 print('try again')
        #                 plt.figure(figsize=(10, 4))
        #                 plt.plot(data[0:1250], color='g')
        #                 print(np.std(npy_temp))
        #                 for j in local_maxima:
        #                     plt.axvline(j, color='r', alpha=0.2)
        #                 plt.title('Pixel Activity of T' + str(trial_index) + '_' + fish + '_' + date + 'Pixel_No_' + str(pixel_index))
        #                 plt.ylabel('Pixel Intensity (a.u)')
        #                 plt.xlabel('Frame')
        #                 plt.show()
        else:
            max_min_idx = False
        #             plt.figure(figsize=(10, 4))
        #             plt.plot(data[0:1250], color='k')
        #             print(np.std(npy_temp))
        #             for j in local_maxima:
        #                 plt.axvline(j, color='r', alpha=0.2)
        #             plt.title('Pixel Activity of T' + str(trial_index) + '_' + fish + '_' + date + 'Pixel_No_' + str(pixel_index))
        #             plt.ylabel('Pixel Intensity (a.u)')
        #             plt.xlabel('Frame')
        #             plt.show()

        sigma = sigma + 1
        if max_min_idx:
            #             print(pixel_index)
            return npy_temp
    return np.zeros(duration)

def swim_detection(fish_dir, trial_index, ROI, date, fish, duration=1250, low_thresh=2, high_thresh=4, sigma=4):
    # Read the video
    heart_video = glob.glob(fish_dir + '*Trial' + str(trial_index) + '.*')
    video = cv2.VideoCapture(heart_video[0])
    # loop to read frame by frame and record intensity of the trial
    averaged_data = np.zeros(duration)

    ret = True
    for index in range(0, duration):
        ret, img = video.read()
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            crop = gray[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2]]
            intensity = np.average(crop)
            averaged_data[index] = intensity
    # detect bouts
    d_curve = np.gradient(averaged_data)
    abs_d_curve = np.abs(d_curve)
    filtered_abs_d_curve = gaussian_filter(abs_d_curve, sigma)
    thresholded = apply_hysteresis_threshold(filtered_abs_d_curve, low_thresh, high_thresh)
    bouts = [(i[0].start, i[0].stop) for i in find_objects(label(thresholded)[0])]

    plt.figure(figsize=(10, 4))
    plt.plot(abs_d_curve, color='b')
    plt.plot(filtered_abs_d_curve, color='g')
    plt.plot(thresholded, color='r')
    plt.legend(['gradient', 'filter', 'threshold'])
    plt.title('Swim Detection of T' + str(trial_index) + '_' + fish + '_' + date)
    plt.ylabel('Pixel Intensity (a.u)')
    plt.xlabel('Frame')
    plt.ylim(0, 15)
    plt.savefig(fish_dir + date + '_' + fish + '_T' + str(trial_index) + '_Swim_Detection.png')
    plt.show()

    for bout in bouts:
        if bout[0] >= 850 and bout[0] <= 1000:
            return True, bouts
    else:
        return False, bouts


def swim_screening(fish_dir, ROI, duration=1250, low_thresh=2, high_thresh=4, sigma=4):
    # first load vsinfo and bradyinfo
    fish = fish_dir[-9:-1]
    print(fish)
    date = fish_dir[-27:-19]
    print(date)
    vsinfo_dir = glob.glob(fish_dir + 'vsinfo*')
    vsinfo = pd.read_excel(vsinfo_dir[0])
    expinfo = pd.read_excel(vsinfo_dir[0], sheet_name='ExpInfo')
    duration = expinfo.Total_Trial_Duration[0] * expinfo.Stimulus_Frame_Rate[0]
    bradyinfo_path = glob.glob(fish_dir + '/Brady*')
    bradyinfo = pd.read_excel(bradyinfo_path[0])
    heart_rate_dataframe = pd.read_excel(bradyinfo_path[0], sheet_name='heart_rate_trace')
    # run quality filter for every trial
    swim_list = []
    swim_bool_list = []
    for trial in range(1, vsinfo.shape[0] + 1):
        swim_bool, swim = swim_detection(fish_dir, trial, ROI, date, fish,duration, low_thresh, high_thresh, sigma)
        swim_list.append(swim)
        swim_bool_list.append(swim_bool)
    bradyinfo['Swim_ROI'] = [ROI] * vsinfo.shape[0]
    bradyinfo['Swim'] = swim_bool_list
    bradyinfo['Swim_Bouts'] = swim_list

    writer = pd.ExcelWriter(bradyinfo_path[0], engine='xlsxwriter')
    bradyinfo.to_excel(writer, sheet_name='Bradyinfo', index=False)
    heart_rate_dataframe.to_excel(writer, sheet_name='heart_rate_trace', index=False)
    writer.save()


def get_bimodel_distribution(fish_dir, bandwidth=2, min_threshold=0, max_threshold=50):
    eye_angle_all = pd.DataFrame()
    vsinfo_dir = glob.glob(fish_dir + '\\vsinfo*')
    vsinfo = pd.read_excel(vsinfo_dir[0])

    for i in range(0, vsinfo.shape[0]):
        file = glob.glob(fish_dir + '\\*Trial' + str(i + 1) + '.avi.h5')[0]
        df_eye = pd.read_hdf(file, key='eye')
        df_eye['trial_idx'] = i
        eye_angle_all = eye_angle_all.append(df_eye.iloc[:, -4:])

    eye_angle_all['convergence_angle'] = eye_angle_all['left_eye']['angle'] + eye_angle_all['right_eye']['angle']
    ax = sns.distplot(eye_angle_all['convergence_angle'], kde=True)
    plt.show()

    data = np.array(eye_angle_all['convergence_angle'])
    data = data[~np.isnan(data)]

    # kde
    min_angle = np.floor(np.nanmin(data))
    max_angle = np.ceil(np.nanmax(data))
    bin_edges = np.arange(min_angle, max_angle + 1)
    counts, bin_edges = np.histogram(data, bins=bin_edges)

    # perform kernel density estimation
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.expand_dims(data, 1))
    # get the log counts
    log_counts = kde.score_samples(np.expand_dims(bin_edges, 1))
    # convert logarithmic values to absolute counts
    kde_counts = np.exp(log_counts)
    # find the value of the mode
    mode = bin_edges[np.argmax(kde_counts)]

    # find convergence threshold
    diffed = np.diff(kde_counts)
    # smooth the differentiated data
    smoothed = pd.Series(diffed).rolling(7, min_periods=0, center=True).mean().values
    # take the sign of the smoothed data (is the function increasing or decreasing)
    signed = np.sign(smoothed)
    # take the derivative of the sign of the first derivative
    second_diff = np.diff(signed)
    # find the indices of local minima (i.e. where the sign of first derivative goes from negative to positive)
    local_minima = np.where(second_diff > 0)[0] + 1
    # find values of the antimodes
    antimodes = bin_edges[local_minima]
    try:
        # Try to find an antimode within the threshold range
        threshold = antimodes[(antimodes > min_threshold) & (antimodes < max_threshold)][0]
    except IndexError:  # local minimum does not exist within the threshold range
        print('No local minimum within limits!')
        threshold = False

    print('The Threshold for ' + fish_dir + ' is ' + str(threshold))
    return eye_angle_all, threshold


def draw_threshold(fish_dir, threshold, bandwidth=2):
    eye_angle_all = pd.DataFrame()
    vsinfo_dir = glob.glob(fish_dir + '\\vsinfo*')
    vsinfo = pd.read_excel(vsinfo_dir[0])

    for i in range(0, vsinfo.shape[0]):
        file = glob.glob(fish_dir + '\\*Trial' + str(i + 1) + '.avi.h5')[0]
        df_eye = pd.read_hdf(file, key='eye')
        df_eye['trial_idx'] = i
        eye_angle_all = eye_angle_all.append(df_eye.iloc[:, -4:])

    eye_angle_all['convergence_angle'] = eye_angle_all['left_eye']['angle'] + eye_angle_all['right_eye']['angle']

    data = np.array(eye_angle_all['convergence_angle'])
    data = data[~np.isnan(data)]

    # kde
    min_angle = np.floor(np.nanmin(data))
    max_angle = np.ceil(np.nanmax(data))
    bin_edges = np.arange(min_angle, max_angle + 1)
    counts, bin_edges = np.histogram(data, bins=bin_edges)

    # perform kernel density estimation
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.expand_dims(data, 1))
    # get the log counts
    log_counts = kde.score_samples(np.expand_dims(bin_edges, 1))
    # convert logarithmic values to absolute counts
    kde_counts = np.exp(log_counts)
    # find the value of the mode
    mode = bin_edges[np.argmax(kde_counts)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    fig.suptitle('Eye convergence threshold = ' + str(threshold) + ' degree')

    converged = bin_edges >= threshold

    ax1.plot(bin_edges, kde_counts * len(data), linewidth=1)
    ax1.fill_between(bin_edges[converged], 0, kde_counts[converged] * len(data))

    ax1.set_title('Kernel density estimation')
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Counts')

    ax2.hist(data, bins=bin_edges)
    ax2.plot([threshold, threshold], [0, counts.max()], c='k', linestyle='dashed')

    ax2.set_title('Raw counts')
    ax2.set_xlabel('Angle (degrees)')

    ax1.set_xlim(-25, 100)
    ax1.set_xticks(np.arange(-25, 125, 25))

    fig.savefig(fish_dir + '//eye_convergence_threshold.png', dpi=300)


def get_pc_trial(fish_dir, threshold):
    bradyinfo_path = glob.glob(fish_dir + '/Brady*')
    bradyinfo = pd.read_excel(bradyinfo_path[0])
    heart_rate_dataframe = pd.read_excel(bradyinfo_path[0], sheet_name='heart_rate_trace')

    PC_list = np.full((bradyinfo.shape[0]), False)
    eye_bout_ranges_list = np.full((bradyinfo.shape[0]), False)

    for i in range(0, bradyinfo.shape[0]):
        file = glob.glob(fish_dir + '\\*Trial' + str(i + 1) + '.avi.h5')[0]
        title = file.split("\\")[-1][:-7]
        df_eye = pd.read_hdf(file, key='eye')
        x = df_eye.shape[0]

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, squeeze=True)
        fig.suptitle(title)

        df_eye.left_eye.angle.plot(ax=axes[0], c='k', linewidth=0.3)
        axes[0].set_ylabel('Left Eye Angle', rotation=90, labelpad=44)
        axes[0].set_ylim([-45, 45])

        df_eye.right_eye.angle.plot(ax=axes[1], c='k', linewidth=0.3)
        axes[1].set_ylabel('Right Eye Angle', rotation=90, labelpad=44)
        axes[1].set_ylim([-45, 45])

        eye_bouts = []
        for frame in range(0, x):
            if df_eye.left_eye.angle[frame] + df_eye.right_eye.angle[frame] >= threshold:
                eye_bouts.append(frame)

        eye_bouts_ranges = []
        if eye_bouts:
            print(np.array(eye_bouts)[np.array(eye_bouts) > 900].shape[0])
            if np.array(eye_bouts)[np.array(eye_bouts) > 900].shape[0] > 40:
                print(i)
                PC_list[i] = True
                print(PC_list)

            for key, g in groupby(enumerate(eye_bouts), lambda x: x[1] - x[0]):
                group = list(map(itemgetter(1), g))
                if len(group) > 1:
                    eye_bouts_ranges.append((group[0], group[-1]))
                else:
                    eye_bouts_ranges.append((group[0], group[0]))
        print(eye_bouts_ranges)
        eye_bout_ranges_list[i] = eye_bouts_ranges

        for bout in eye_bouts_ranges:
            axes[0].axvspan(*(np.array(bout)), alpha=.1, color='r')
            axes[1].axvspan(*(np.array(bout)), alpha=.1, color='r')
        plt.xlim(0, x)

        plt.xlabel('Frames')
        plt.show()

        fig.savefig(fish_dir+ '\\eye_plotting_' + title + '.png', dpi=300)

    return PC_list, eye_bout_ranges_list

