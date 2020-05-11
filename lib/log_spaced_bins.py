import numpy as np
from fitEllipse import XYtoThetaSimple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from filters import run_win_smooth


def turn_into_dataframe(data, exp):
    """Input: Data, nested dictionnary,
       Output: pandas Dataframe, index: time in second, 2 columns : ["x", "y"] """

    posx_1 = data[exp]["x"]
    posy_1 = data[exp]["y"]
    FPS = data[exp]["FPS"]
    df_xy = pd.DataFrame(np.array([posx_1, posy_1]).T)
    df_xy.columns = ["x", "y"]  # reduce size of data

    return df_xy


def reorganize_data(list):
    """Input: list of data
       Output: 2D array, easier to use"""
    matrix = np.zeros([len(list), len(list[0])])

    for i, array in enumerate(list):
        for j, value in enumerate(array):

            matrix[i, j] = value

    return matrix


def average_psd_maker(m_df, FPS, seg_number=100):
    """Input: data, number of equal length array
       Perform PSD for each equal length array, then compute the average
       Output: len(df)/seg_number array-like"""

    new_index_len = int(len(m_df.index)/seg_number)*seg_number
    m_df = m_df.iloc[:new_index_len]
    seg_number = seg_number
    posx = m_df["x"]
    posy = m_df["y"]

    equal_lenght_df = [
        pd.DataFrame(np.array([posx[int(x * len(posx) / seg_number):int((x + 1) * len(posx) / seg_number)],
                               posy[
                               int(x * len(posx) / seg_number):int((x + 1) * len(posy) / seg_number)]]).T,
                     columns=["x", "y"]) for x in range(seg_number)]
    psd_list = []

    for a_df in equal_lenght_df:

        a_df.index = a_df.index / FPS
        angle_theta = XYtoThetaSimple(a_df["x"], a_df["y"])
        vitesse = np.gradient(angle_theta) * FPS
        vitesse -= vitesse.mean()
        n = max(a_df.index)
        N = len(a_df.index)
        step = n / N
        PSD_speed = abs(np.fft.fft(vitesse)) ** 2 / (N * FPS)
        freq = np.linspace(0.0, 1 / step, N)
        psd_list.append(PSD_speed)
        # plt.plot(freq, PSD_speed)
        # plt.show()
    psd_mat = reorganize_data(psd_list)
    average_psd = [psd_mat[:, x].mean() for x in range(psd_mat.shape[1])]

    return average_psd, freq


def fit_data(x, y):
    """call curve fit, output: fitted parameters"""
    global sp
    parameters, pcov = curve_fit(model, x[:len(sp) // 2], y[:len(sp) // 2], p0=[3.58840285e+02, 5.37790538e-05, 4.08722267e-01],
                                         # average fitting parameters value
                                        bounds=(0, [np.inf, np.inf, np.inf]))
    return parameters


def model(f, a, gamma, kl):

    global vmean
    Kb = 1.380649E-23
    T = 300
    tc = gamma/kl

    return ((a*vmean)/((2*np.pi*f*tc)**2 + 1)) + ((2*Kb*T)/gamma)*((2*np.pi*f*tc)**2/((2*np.pi*f*tc)**2 + 1))


data = np.load("../data/D_WT_1000SS.p")  # opening data
exp_number = 3  # experiment to choose in the data
FPS = data[exp_number]["FPS"]
df = turn_into_dataframe(data, exp_number)

angle_theta = XYtoThetaSimple(df["x"], df["y"])
# angle_degree = (angle_theta*180/np.pi)  # Turn the angle into degree
#############
# Perform gradient : get instantaneous speed
sp = np.gradient(angle_theta)[2:]  # Crop to first points because gradients give weird result on them
angle_theta = angle_theta[2:]
df = df[2:]
vmean = sp.mean()
############
# Average PSD

average_psd, freq = average_psd_maker(df, FPS, seg_number=100)
log_freq = np.logspace(np.log10(freq[1]), np.log10(freq[:len(freq)//2][-1]+1), 200)  # Create log space bins from my freq

df_PSD = pd.DataFrame(data=np.array(average_psd).T, index=freq, columns=["PSD"]).iloc[:len(average_psd)//2]

digitized = np.digitize(df_PSD.index, log_freq)  # find which freq stand in which bin
df_PSD["digitized"] = digitized
array_to_mean = []

an_array = []
n_log_freq = []

for i in range(len(df_PSD)):

    if i != 0:
        if df_PSD["digitized"].iloc[i] == df_PSD["digitized"].iloc[i-1] or len(an_array) == 0:

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

fig, [ax1, ax2] = plt.subplots(2, 1)

ax1.set_title("key number : 0, bin: 300, PSD_average: 100 equal length segment")
ax1.set_xlabel("Freq, Hz")
ax1.scatter(n_log_freq[1:], bin_mean[1:], s=1)
ax1.set_xscale("log")
ax2.set_title("key number : 0, PSD_average: 100 equal length segment, no binning")
ax2.set_xlabel("Freq, Hz")
ax2.loglog(freq[:len(freq)//2], average_psd[:len(freq)//2])
plt.show()
