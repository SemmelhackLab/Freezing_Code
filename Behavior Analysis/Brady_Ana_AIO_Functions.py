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
import math
from scipy.signal import butter, filtfilt, find_peaks
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
        quality, trace = heart_rate_pixelwise_trace(fish_dir, fish, date, trial, ROI, duration)
        heart_rate_array[trial - 1, :] = trace
        quality_list.append(quality)
    bradyinfo['Good_Pixel_Number'] = quality_list
    heart_rate_dataframe = pd.DataFrame(heart_rate_array)

    bradyinfo_path = fish_dir + 'Bradyinfo_' + date + '_' + fish + '.xlsx'
    writer = pd.ExcelWriter(bradyinfo_path, engine='xlsxwriter')
    bradyinfo.to_excel(writer, sheet_name='Bradyinfo', index=False)
    heart_rate_dataframe.to_excel(writer, sheet_name='heart_rate_trace', index=False)
    writer.save()


def heart_rate_pixelwise_trace(fish_dir, fish, date, trial_index, ROI, duration=2200):
    """
    run averaged pixel activity of a trial and check its quality, also return quality boolean and trace as a list,
    also generate the heart rate plot
    """
    # Read the video
    heart_video = glob.glob(fish_dir + 'Side_Camera/' + '*heart_' + str(trial_index) + '.*')
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
            good_pixel_array[:, x, y] = pixel_heart_rate(fish_dir, fish, date, trial_index, pixel_index, pixel_trace,duration)

    averaged_heart_rate = np.zeros(duration)
    for frame in range(0, duration):
        pixels_of_frame = good_pixel_array[frame, :, :]
        good_pixels = pixels_of_frame.ravel()[np.flatnonzero(pixels_of_frame)].tolist()
        averaged_heart_rate[frame] = np.average(good_pixels)

    plt.figure(figsize=(12, 7))
    plt.plot(averaged_heart_rate[0:])
    x1 = np.arange(900, 1300)
    plt.fill_between(x1, 4, -4, linewidth=1, color='violet', alpha=0.3)
    plt.ylim(0, 4)
    print(np.std(averaged_heart_rate))
    # plt.title('Averaged Heart Rate of T' + str(trial_index) + '_' + fish + '_' + date,fontsize = 20)
    plt.ylabel('Heart Rate per Minute',fontsize=26)
    plt.yticks([0,0.5,1,1.5,2,2.5,3,3.5,4],
               [0, 30,60,90,120,150,180,210,240],fontsize=22)
    plt.xlabel('Time(Seconds)', fontsize=26)
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200],
               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22], fontsize=22)
    plt.savefig(fish_dir + 'Side_Camera/' + 'Averaged Heart Rate of_' + date + '_' + fish + '_T' + str(trial_index) + '.png',bbox_inches="tight",dpi=300)
    #     plt.show()

    return len(good_pixels), averaged_heart_rate


def pixel_heart_rate(fish_dir, fish, date, trial_index, pixel_index, pixel_trace, duration=2200, starting_sigma=4,
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
        else:
            max_min_idx = False


        sigma = sigma + 1
        if max_min_idx:
            #             print(pixel_index)
            return npy_temp
    return np.zeros(duration)


def calculate_HR_to_ceiling(bradyinfo_list,median_window_len,max_window_len)
    for bradyinfo_dir in bradyinfo_list:
        heart_rate = pd.read_excel(bradyinfo_dir,sheet_name = 'heart_rate_trace')
        trial_no = heart_rate.shape[0]
        np_HR_to_ceiling = np.zeros((trial_no,heart_rate.shape[1]))
        np_HR_reduction = np.zeros((trial_no,heart_rate.shape[1]))
        for t in range(0,trial_no):
            raw_HR = heart_rate.iloc[t,:].to_numpy()
            filtered_HR = scipy.ndimage.median_filter(raw_HR, median_window_len,mode = 'nearest')
            ceiling = scipy.ndimage.filters.maximum_filter1d(filtered_HR,max_window_len,mode = 'nearest')

            HR_to_ceiling = raw_HR-ceiling
            np_HR_to_ceiling[t,:] = HR_to_ceiling

        df_HR_to_ceiling = pd.DataFrame(np_HR_to_ceiling)
        with pd.ExcelWriter(bradyinfo_dir,engine="openpyxl",mode = 'a',if_sheet_exists = 'replace') as writer:
            df_HR_to_ceiling.to_excel(writer,sheet_name = 'HR_to_ceiling',index=False)
            

def identify_bradycardia(bradyinfo_list,std_times=3):
    brady_threshold = list()
    for bradyinfo_dir in bradyinfo_list:
        HR_to_ceiling = pd.read_excel(bradyinfo_dir,sheet_name = 'HR_to_ceiling')
        HR = pd.read_excel(bradyinfo_dir,sheet_name = 'heart_rate_trace')
        bradyinfo = pd.read_excel(bradyinfo_dir,sheet_name = 'Bradyinfo')
        roi = eval(bradyinfo['roi'][0])
        total_pixel_no = roi[2]*roi[3]

        baseline_HR_to_ceiling = []
        baseline_HR = []
        vsinfo_dir = glob.glob(main_dir+'/'+date+'/behavior/'+fish+'/vsinfo*')[0]
        vsinfo = pd.read_excel(vsinfo_dir,sheet_name = 'VsInfo')
        for t in range(0,HR_to_ceiling.shape[0]):
            if bradyinfo['Good_Pixel_Number'][t]/total_pixel_no>0.05:
                if vsinfo['Left_Stimulus_Type'][t]=='n' and vsinfo['Right_Stimulus_Type'][t]=='n':
                    for f in HR_to_ceiling.iloc[t,:]:
                        baseline_HR_to_ceiling.append(f)

        temp_avg_baseline = np.average(baseline_HR_to_ceiling)
        temp_std_baseline = np.std(baseline_HR_to_ceiling)
        brady_threshold.append(temp_avg_baseline-std_times*temp_std_baseline)

    # write the brady bouts into bradyinfo 
    for bradyinfo_dir,brady_threshold_fish in zip(bradyinfo_list,brady_threshold):
        bradyinfo = pd.read_excel(bradyinfo_dir,sheet_name = 'Bradyinfo')
        bradyinfo['Brady_bouts'] = '[]'
        HR_to_ceiling = pd.read_excel(bradyinfo_dir,sheet_name = 'HR_to_ceiling')
        heart_rate_trace = pd.read_excel(bradyinfo_dir,sheet_name = 'heart_rate_trace')
        for t in range(0,HR_to_ceiling.shape[0]):
            thresholded = HR_to_ceiling.iloc[t,:]<brady_threshold_fish
            thresholded_invert = np.invert(thresholded)
            neg_bouts = [(i[0].start, i[0].stop) for i in find_objects(label(thresholded_invert)[0])]
            for i in range(0, len(neg_bouts)):
                temp_bout_length = neg_bouts[i][1] - neg_bouts[i][0]
                if temp_bout_length <= 20:
                    thresholded[neg_bouts[i][0]:neg_bouts[i][1]] = True

            brady_bouts_unfiltered = [(i[0].start, i[0].stop) for i in find_objects(label(thresholded)[0])]
            list_false_positive = []
            for i in range(0, len(brady_bouts_unfiltered)):
                temp_bout_length = brady_bouts_unfiltered[i][1] - brady_bouts_unfiltered[i][0]
                if temp_bout_length < 50:
                    list_false_positive.append(i)
            brady_bouts = [i for j, i in enumerate(brady_bouts_unfiltered) if j not in list_false_positive]
            bradyinfo['Brady_bouts'][t] = brady_bouts

        writer = pd.ExcelWriter(bradyinfo_dir, engine='xlsxwriter')
        bradyinfo.to_excel(writer, sheet_name='Bradyinfo',index = False)
        heart_rate_trace.to_excel(writer, sheet_name='heart_rate_trace',index = False)
        HR_to_ceiling.to_excel(writer, sheet_name='HR_to_ceiling',index = False)
        writer.save()
        writer.close()

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
        if bout[0] >= 850 and bout[0] <= 1300:
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
        quality, trace = heart_rate_pixelwise_trace(fish_dir, fish, date, trial, ROI, duration)
        heart_rate_array[trial - 1, :] = trace
        quality_list.append(quality)
    bradyinfo['Good_Pixel_Number'] = quality_list
    heart_rate_dataframe = pd.DataFrame(heart_rate_array)

    bradyinfo_path = fish_dir + 'Bradyinfo_' + date + '_' + fish + '.xlsx'
    writer = pd.ExcelWriter(bradyinfo_path, engine='xlsxwriter')
    bradyinfo.to_excel(writer, sheet_name='Bradyinfo', index=False)
    heart_rate_dataframe.to_excel(writer, sheet_name='heart_rate_trace', index=False)
    writer.save()


def heart_rate_pixelwise_trace(fish_dir, fish, date, trial_index, ROI, duration=2200):
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
            good_pixel_array[:, x, y] = pixel_heart_rate(fish_dir, fish, date, trial_index, pixel_index, pixel_trace,duration)

    averaged_heart_rate = np.zeros(duration)
    for frame in range(0, duration):
        pixels_of_frame = good_pixel_array[frame, :, :]
        good_pixels = pixels_of_frame.ravel()[np.flatnonzero(pixels_of_frame)].tolist()
        averaged_heart_rate[frame] = np.average(good_pixels)

    plt.figure(figsize=(12, 7))
    plt.plot(averaged_heart_rate[0:])
    x1 = np.arange(900, 1300)
    plt.fill_between(x1, 4, -4, linewidth=1, color='violet', alpha=0.3)
    plt.ylim(0, 4)
    print(np.std(averaged_heart_rate))
    # plt.title('Averaged Heart Rate of T' + str(trial_index) + '_' + fish + '_' + date,fontsize = 20)
    plt.ylabel('Heart Rate per Minute',fontsize=26)
    plt.yticks([0,0.5,1,1.5,2,2.5,3,3.5,4],
               [0, 30,60,90,120,150,180,210,240],fontsize=22)
    plt.xlabel('Time(Seconds)', fontsize=26)
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200],
               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22], fontsize=22)
    plt.savefig(fish_dir + 'Averaged Heart Rate of_' + date + '_' + fish + '_T' + str(trial_index) + '.png',bbox_inches="tight",dpi=300)
    #     plt.show()

    return len(good_pixels), averaged_heart_rate


def pixel_heart_rate(fish_dir, fish, date, trial_index, pixel_index, pixel_trace, duration=2200, starting_sigma=4,
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
        if bout[0] >= 890 and bout[0] <= 1000:
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

def Bout_Reader(fish_dir, duration=2200, low_thresh=2, high_thresh=4, sigma_angles=0, sigma=3,num_points=10,bout_sigma=3,bout_threshold=1):
    # first load vsinfo and bradyinfo
    fish = fish_dir[-9:-1]
    print(fish)
    date = fish_dir[-27:-19]
    print(date)
    vsinfo_dir = glob.glob(fish_dir + 'vsinfo*')
    vsinfo = pd.read_excel(vsinfo_dir[0])
    expinfo = pd.read_excel(vsinfo_dir[0], sheet_name='ExpInfo')
    duration = expinfo.Total_Trial_Duration[0] * expinfo.Stimulus_Frame_Rate[0]
######
#     bradyinfo_path = glob.glob(fish_dir + '/Brady*')
#     bradyinfo = pd.read_excel(bradyinfo_path[0])
#     heart_rate_dataframe = pd.read_excel(bradyinfo_path[0], sheet_name='heart_rate_trace')
######
    tail_curvature_array = np.zeros((vsinfo.shape[0], duration))
    # run quality filter for every trial
    swim_bouts_list = []
    swim_bouts_amp_curvature_list = []
    swim_bouts_amp_tip_angle_list = []
    swim_bouts_amp_middle_angle_list = []
    swim_bouts_avg_velocity_list = []
    swim_bouts_max_velocity_list = []
    swim_bouts_time_list = []
    swim_bouts_fre_list = []
    swim_bool_list = []
    swim_bouts_int_list = []
    
    for trial in range(1, vsinfo.shape[0] + 1):
        swim_bool, swim_trace, swim_bouts, swim_bouts_amp_tip_angle,swim_bouts_amp_middle_angle,swim_bouts_amp_curvature,swim_bouts_avg_velocity,swim_bouts_max_velocity,swim_bouts_time, swim_bouts_fre, swim_bouts_int = Bout_Reader_Trial(fish_dir, trial, date, fish, duration, low_thresh, high_thresh, sigma_angles, sigma, num_points,bout_sigma,bout_threshold)
        tail_curvature_array[trial-1,0:duration] = swim_trace.tolist()[0:duration]
        swim_bool_list.append(swim_bool)
        swim_bouts_list.append(swim_bouts)
        swim_bouts_amp_tip_angle_list.append(swim_bouts_amp_tip_angle)
        swim_bouts_amp_middle_angle_list.append(swim_bouts_amp_middle_angle)
        swim_bouts_amp_curvature_list.append(swim_bouts_amp_curvature)
        swim_bouts_avg_velocity_list.append(swim_bouts_avg_velocity)
        swim_bouts_max_velocity_list.append(swim_bouts_max_velocity)
        swim_bouts_time_list.append(swim_bouts_time)
        swim_bouts_fre_list.append(swim_bouts_fre)
        swim_bouts_int_list.append(swim_bouts_int)
#####
#     bradyinfo['Swim'] = swim_bool_list
#     bradyinfo['Swim_Bouts'] = swim_bouts_list
#     bradyinfo['Swim_Bouts_Amplitude_Angle'] =swim_bouts_amp_angle_list
#     bradyinfo['Swim_Bouts_Amplitude_Curvature'] =swim_bouts_amp_curvature_list
#     bradyinfo['Swim_Bouts_Time'] =swim_bouts_time_list
#     bradyinfo['Swim_Bouts_Frequency'] = swim_bouts_fre_list
#     bradyinfo['Swim_Bouts_Integral'] = swim_bouts_int_list
#     print(bradyinfo)
#####
    df_boutinfo = pd.DataFrame()
    df_boutinfo['Trial'] = range(1, vsinfo.shape[0] + 1)
    df_boutinfo['Swim'] = swim_bool_list
    df_boutinfo['Swim_Bouts'] = swim_bouts_list
    df_boutinfo['Swim_Bouts_Amplitude_Curvature'] =swim_bouts_amp_curvature_list
    df_boutinfo['Swim_Bouts_Amplitude_Tip_Angle'] =swim_bouts_amp_tip_angle_list
    df_boutinfo['Swim_Bouts_Amplitude_Middle_Angle'] =swim_bouts_amp_middle_angle_list
    df_boutinfo['Swim_Bouts_Avg_Velocity'] =swim_bouts_avg_velocity_list
    df_boutinfo['Swim_Bouts_Max_Velocity'] =swim_bouts_max_velocity_list
    df_boutinfo['Swim_Bouts_Time'] =swim_bouts_time_list
    df_boutinfo['Swim_Bouts_Frequency'] = swim_bouts_fre_list
    df_boutinfo['Swim_Bouts_Integral'] = swim_bouts_int_list
    
    
    tail_curvature_dataframe = pd.DataFrame(tail_curvature_array)
######
#     writer = pd.ExcelWriter(bradyinfo_path[0], engine='xlsxwriter')
    
#     bradyinfo.to_excel(writer, sheet_name='Bradyinfo', index=False)
#     heart_rate_dataframe.to_excel(writer, sheet_name='heart_rate_trace', index=False)
#     writer.save()
#     writer.close()
######
    boutsinfo_path = fish_dir + 'Boutsinfo_' + date + '_' + fish + '.xlsx'
    writer = pd.ExcelWriter(boutsinfo_path, engine='xlsxwriter')
    df_boutinfo.to_excel(writer, sheet_name='boutinfo', index=False)
    tail_curvature_dataframe.to_excel(writer, sheet_name='tail_curvature_trace', index=False)
    writer.save()
    writer.close()

def get_vectorized_tail_h5(file, sigma):
    df_tail = pd.read_hdf(file, key='tail')
    points = np.stack([df_tail.loc(axis=1)[:,'x'].iloc[:,:],df_tail.loc(axis=1)[:,'y'].iloc[:,:]],axis=-1)
    filtered = gaussian_filter(points, sigma=[0, sigma, 0], mode='nearest')
    diff = filtered[:, :-1] - filtered[:, 1:]
    heading = 0
    heading_vector = np.array([np.cos(heading), np.sin(heading)])
    angles = np.arctan2(np.cross(diff, heading_vector), np.dot(diff, heading_vector))
    for i in range(0,angles.shape[0]):
        for j in range(0,angles.shape[1]):
            if angles[i][j] >-math.pi-0.00001 and angles[i][j] <-math.pi/2:
                angles[i][j] = angles[i][j]+2*math.pi
    
    tip_diff = filtered[:, 0]-filtered[:, -1]
    tip_angle = np.arctan2(np.cross(tip_diff, heading_vector), np.dot(tip_diff, heading_vector))
    
    for i in range(0,tip_angle.shape[0]):
        if tip_angle[i] >-math.pi-0.00001 and tip_angle[i] <-math.pi/2:
            tip_angle[i] = tip_angle[i]+2*math.pi
            
            
    middle_diff = filtered[:, 0]-filtered[:, 9]
    middle_angle = np.arctan2(np.cross(middle_diff, heading_vector), np.dot(middle_diff, heading_vector))
    
    for i in range(0,middle_angle.shape[0]):
        if middle_angle[i] >-math.pi-0.00001 and middle_angle[i] <-math.pi/2:
            middle_angle[i] = middle_angle[i]+2*math.pi
    # diff_angles = angles[:, :-1] - angles[:, 1:]
    # bad_frames = []
    # for rows in range(1,diff_angles.shape[0]-1):
    #     if np.max(np.abs(diff_angles[rows]))>1.04:
    #         # bad_frames.append(rows)
    #         angles[rows] = angles[rows-1]/2 +angles[rows+1]/2
    # return angles,bad_frames
    return tip_angle,middle_angle,angles

def get_curvature(vectorized, num_points):
    return vectorized[:, -num_points:].mean(-1)

def Bout_Reader_Trial(fish_dir, trial_index, date, fish, duration=2200, low_thresh =1, high_thresh =3, sigma_angles = 0, sigma = 3, num_points = 10,bout_sigma=3,bout_prominence=1):
    # read tail tracking h5 file
    h5file_dir = glob.glob(fish_dir + '/Top_Camera/*Trial'+ str(trial_index) + '.mp4.h5')[0]
    tip_angle,middle_angle,angles = get_vectorized_tail_h5(h5file_dir,sigma = sigma_angles)
    # calculate curvature and save as averaged data
    averaged_data = get_curvature(angles,num_points)
    # detect bouts
    d_curve = np.gradient(averaged_data)
    abs_d_curve = np.abs(d_curve)
    filtered_abs_d_curve = gaussian_filter(abs_d_curve, sigma)
    thresholded = apply_hysteresis_threshold(filtered_abs_d_curve, low_thresh, high_thresh)
    # merge the bouts who are near with each other
    thresholded_invert = np.invert(thresholded)
    neg_bouts = [(i[0].start, i[0].stop) for i in find_objects(label(thresholded_invert)[0])]
    for i in range(0, len(neg_bouts)):
        temp_bout_length = neg_bouts[i][1] - neg_bouts[i][0]
        if temp_bout_length <= 25:
            thresholded[neg_bouts[i][0]:neg_bouts[i][1]] = True
    
    bouts = [(i[0].start, i[0].stop) for i in find_objects(label(thresholded)[0])]
    
        
    list_false_positive = []
    for i in range(0, len(bouts)):
        temp_bout_length = bouts[i][1] - bouts[i][0]
        if temp_bout_length < 8:
            list_false_positive.append(i)
    bouts_filtered = [i for j, i in enumerate(bouts) if j not in list_false_positive]
    
    
    
    # plotting
    plt.figure(figsize=(10, 4))
    plt.plot(averaged_data/4, color='b')
    plt.plot(filtered_abs_d_curve, color='g')
    plt.plot(thresholded*high_thresh, color='r')
    plt.legend(['curvature', 'gradient', 'threshold'])
    plt.title('Swim Detection of T' + str(trial_index) + '_' + fish + '_' + date)
    plt.ylabel('Curvature Gradient (a.u)')
    plt.xlabel('Frame')
    plt.ylim(0, 1)
    plt.savefig(fish_dir + date + '_' + fish + '_T' + str(trial_index) + '_Swim_Detection0.png')
    plt.show()

    # find the max curvature in each bout
    amp_tip_angle_list = []
    amp_middle_angle_list = []
    amp_curvature_list = []
    avg_velocity_list = []
    max_velocity_list = []
    time_list = []
    fre_list = []
    int_list = []

    trans_curvature = np.rad2deg(averaged_data-math.pi/2)
    trans_tip_angle = np.rad2deg(tip_angle-math.pi/2)
    trans_middle_angle = np.rad2deg(middle_angle-math.pi/2)
    
    for bout in bouts_filtered:
        bout_curv_data = gaussian_filter(trans_curvature[bout[0]:bout[1]],bout_sigma)
        bout_tip_angle_data = gaussian_filter(trans_tip_angle[bout[0]:bout[1]],bout_sigma)
        bout_middle_angle_data = gaussian_filter(trans_middle_angle[bout[0]:bout[1]],bout_sigma)
        bout_velocity = filtered_abs_d_curve[bout[0]:bout[1]]
        fre, peaks = get_bout_fre(bout_curv_data,prominence=bout_prominence)
        
        amp_curvature_list.append(np.abs(bout_curv_data[np.argmax(np.abs(bout_curv_data))]))
        amp_tip_angle_list.append(np.abs(bout_tip_angle_data[np.argmax(np.abs(bout_tip_angle_data))]))
        amp_middle_angle_list.append(np.abs(bout_middle_angle_data[np.argmax(np.abs(bout_middle_angle_data))]))
        avg_velocity_list.append(np.average(bout_velocity))
        max_velocity_list.append(np.max(bout_velocity))
        fre_list.append(np.abs(fre)*200)
        time_list.append(bout[1]-bout[0])
        int_list.append(np.abs(np.sum(np.abs(bout_tip_angle_data))))

        plt.figure(figsize=(10, 4))
        plt.plot(bout_curv_data, color='b')
        plt.plot(peaks, bout_curv_data[peaks], "x",color='r')

        # for j in peaks:
        #     plt.axvline(j, color='r', alpha=0.2)
        plt.legend(['curvature','bout peaks'])
        plt.title('Reading of Bout_' + str(bout[0]) + '_T_' + str(trial_index) + '_' + fish + '_' + date)
        plt.ylabel('Curvature')
        plt.xlabel('Frame')
        plt.ylim(-180, 180)
        plt.savefig(fish_dir + date + '_' + fish + '_T' + str(trial_index) + '_Bout_' + str(bout[0]) + '0.png')
        plt.show()

    for bout in bouts_filtered:
        if bout[0] >= 1780 and bout[0] <= 2600:
            return True, averaged_data, bouts_filtered, amp_tip_angle_list,amp_middle_angle_list,amp_curvature_list,avg_velocity_list,max_velocity_list,time_list, fre_list, int_list
    else:
        return False, averaged_data, bouts_filtered, amp_tip_angle_list,amp_middle_angle_list,amp_curvature_list,avg_velocity_list,max_velocity_list,time_list, fre_list, int_list

def get_bout_fre(bout_curv_data,prominence=1):
    peaks, _ = find_peaks(bout_curv_data,prominence = prominence)
    mins, _ = find_peaks(-bout_curv_data,prominence = prominence)
    if len(mins)>len(peaks):
        peaks = mins
    frequency = len(peaks)/len(bout_curv_data)
    return frequency, peaks

def get_bimodel_distribution(fish_dir, bandwidth=2, min_threshold=0, max_threshold=50):
    eye_angle_all = pd.DataFrame()
    vsinfo_dir = glob.glob(fish_dir + '\\vsinfo*')
    vsinfo = pd.read_excel(vsinfo_dir[0])

    for i in range(0, vsinfo.shape[0]):
        file = glob.glob(fish_dir + '\\Top_Camera\\*Trial' + str(i + 1) + '.mp4.h5')[0]
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

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.expand_dims(data, 1))
    log_counts = kde.score_samples(np.expand_dims(bin_edges, 1))
    kde_counts = np.exp(log_counts)
    mode = bin_edges[np.argmax(kde_counts)]

    diffed = np.diff(kde_counts)
    smoothed = pd.Series(diffed).rolling(7, min_periods=0, center=True).mean().values
    signed = np.sign(smoothed)
    second_diff = np.diff(signed)
    local_minima = np.where(second_diff > 0)[0] + 1
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
        file = glob.glob(fish_dir + '\\Top_Camera\\*Trial' + str(i + 1) + '.mp4.h5')[0]
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
    eye_bout_ranges_list = []

    for i in range(0, bradyinfo.shape[0]):
        file = glob.glob(fish_dir + '\\Top_Camera\\*Trial' + str(i + 1) + '.mp4.h5')[0]
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
        eye_thresholded = np.full(x, False)
        for frame in range(0, x):
            if df_eye.left_eye.angle[frame] + df_eye.right_eye.angle[frame] >= threshold:
                eye_bouts.append(frame)
                eye_thresholded[frame] = True

        eye_bouts_ranges = []
        if eye_bouts:
            print(np.array(eye_bouts)[np.logical_and(np.array(eye_bouts) > 900, np.array(eye_bouts) < 1300)].shape[0])
            if np.array(eye_bouts)[np.logical_and(np.array(eye_bouts) > 900, np.array(eye_bouts) < 1300)].shape[0] > 40:
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
        
        
        
        eye_thresholded_invert = np.invert(eye_thresholded)
        eye_neg_bouts = [(i[0].start, i[0].stop) for i in find_objects(label(eye_thresholded_invert)[0])]
        for i in range(0, len(eye_neg_bouts)):
            temp_bout_length = eye_neg_bouts[i][1] - eye_neg_bouts[i][0]
            if temp_bout_length <= 60:
                eye_thresholded[eye_neg_bouts[i][0]:eye_neg_bouts[i][1]] = True

        eye_bouts_unfiltered = [(i[0].start, i[0].stop) for i in find_objects(label(eye_thresholded)[0])]
        list_false_positive = []
        for i in range(0, len(eye_bouts_unfiltered)):
            temp_bout_length = eye_bouts_unfiltered[i][1] - eye_bouts_unfiltered[i][0]
            if temp_bout_length < 150:
                list_false_positive.append(i)
        eye_bouts_filtered = [i for j, i in enumerate(eye_bouts_unfiltered) if j not in list_false_positive]
        print(eye_bouts_filtered)
        
        eye_bout_ranges_list.append(eye_bouts_filtered)

        for bout in eye_bouts_ranges:
            axes[0].axvspan(*(np.array(bout)), alpha=.1, color='r')
            axes[1].axvspan(*(np.array(bout)), alpha=.1, color='r')
        plt.xlim(0, x)

        plt.xlabel('Frames')
        plt.show()

        fig.savefig(fish_dir+ '\\eye_plotting_' + title + '.png', dpi=300)

    return PC_list, eye_bout_ranges_list

