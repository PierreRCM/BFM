import numpy as np
import pandas as pd
from fitEllipse import XYtoThetaSimple
import warnings

def log_bin_average(freq, average_psd, bin_number=20):

    df_PSD = pd.DataFrame(data=np.array(average_psd).T, index=freq, columns=["PSD"])
    log_freq = np.logspace(np.log10(freq[0]), np.log10(freq[-1]+0.01), bin_number)
    #
    digitized = np.digitize(df_PSD.index, log_freq)  # find which freq stand in which bin
    #
    df_PSD["digitized"] = digitized
    an_array = []
    array_to_mean = []
    n_log_freq = []
    for i in range(len(df_PSD)):

        if i != 0:
            if df_PSD["digitized"].iloc[i] == df_PSD["digitized"].iloc[i - 1] or len(an_array) == 0:

                an_array.append(df_PSD["PSD"].iloc[i])
            else:
                n_log_freq.append(log_freq[df_PSD["digitized"].iloc[i]])
                array_to_mean.append(an_array)
                an_array = []
        else:
            an_array.append(df_PSD["PSD"].iloc[0])

    bin_mean = []

    for arr in array_to_mean:
        bin_mean.append(np.array(arr).mean())

    if bin_number != len(bin_mean):

        warnings.warn("Some bins are empty")

    return n_log_freq, bin_mean


def reorganize_data(list):

    matrix = np.zeros([len(list), len(list[0])])

    for i, array in enumerate(list):
        for j, value in enumerate(array):

            matrix[i, j] = value

    return matrix


def average_psd_maker(data, FPS=10000, seg_number=100, data_type="x_y"):
    """Input: data
       Perform PSD for each equal length array, then compute the average
       Output: len(df)/seg_number array-like"""

    equal_lenght_df = []

    if data_type == "x_y":

        posx = data[0]
        posy = data[1]
        m_df = pd.DataFrame(np.array([posx, posy]).T)
        new_index_len = int(len(m_df.index)/seg_number)*seg_number
        m_df = m_df.iloc[:new_index_len]
        seg_number = seg_number
        m_df.columns = ["x", "y"]
        posx = m_df["x"]
        posy = m_df["y"]

        equal_lenght_df = [
            pd.DataFrame(np.array([posx[int(x * len(posx) / seg_number):int((x + 1) * len(posx) / seg_number)],
                                   posy[
                                   int(x * len(posx) / seg_number):int((x + 1) * len(posy) / seg_number)]]).T,
                         columns=["x", "y"]) for x in range(seg_number)]
    elif data_type == "angle":

        angle = data
        m_df = pd.DataFrame(np.array([angle]).T)
        new_index_len = int(len(m_df.index) / seg_number) * seg_number
        m_df = m_df.iloc[:new_index_len]
        seg_number = seg_number
        m_df.columns = ["angle"]
        angle = m_df["angle"]

        equal_lenght_df = [
            pd.DataFrame(np.array([angle[int(x * len(angle) / seg_number):int((x + 1) * len(angle) / seg_number)]]).T,
                         columns=["angle"]) for x in range(seg_number)]

    psd_list = []
    list_vmean = []
    for a_df in equal_lenght_df:

        angle_theta = []
        a_df.index = a_df.index / FPS

        if data_type == "x_y":
            angle_theta = XYtoThetaSimple(a_df["x"], a_df["y"])
        elif data_type == "angle":
            angle_theta = a_df["angle"]

        vitesse = np.diff(angle_theta) * FPS
        vmean = vitesse.mean()
        list_vmean.append(vmean)
        dft = np.abs(np.fft.fft(vitesse))
        psd = (dft ** 2)
        freqs = np.fft.fftfreq(vitesse.size, 1 / FPS)
        PSD_speed, freqs = np.fft.fftshift(psd), np.fft.fftshift(freqs)
        PSD_speed = PSD_speed / (len(vitesse) * FPS)
        # Make it one sided:

        freqs, PSD_speed = freqs[int(len(freqs) / 2) + 1:],PSD_speed[int(len(freqs) / 2) + 1:]
        PSD_speed *= 2

        psd_list.append(PSD_speed)

    vmean = np.array(list_vmean).mean()
    psd_mat = reorganize_data(psd_list)
    average_psd = [psd_mat[:, x].mean() for x in range(psd_mat.shape[1])]

    return average_psd, freqs, vmean