import matplotlib.pyplot as plt
import numpy as np
from average_psd_maker import HandleData
import pandas as pd
import os

path = os.getcwd().split("/")
path.pop(-1)
path_1 = path.copy()
path_1.append("simulation")

path_1 = "/".join(path_1)
path = "/".join(path)
path_to_save = path +"/result/fit_sim"
k_to_test = np.linspace(5E-20, 5E-18, 15)
drag_to_test = [500E-9, 1000E-9]
step_to_test = np.linspace(20, 1000, 15)


dist_beadsurf = 5000E-9
eta = 0.954e-3
axis_offset = 200E-9

for r in drag_to_test:

    faxen_1 = 1 - (1 / 8.) * (r / dist_beadsurf) ** 3
    faxen_2 = 1 - (9 / 16.) * (r / dist_beadsurf) + (1. / 8) * (r / dist_beadsurf) ** 3 - (
            45 / 256.) * (r / dist_beadsurf) ** 4 - (1. / 16) * (r / dist_beadsurf) ** 5
    drag_bead = (8 * np.pi * eta * r ** 3) / faxen_1 + (
            6 * np.pi * eta * r * axis_offset ** 2) / faxen_2

    for i, step in enumerate(step_to_test):
        k_df = pd.DataFrame()
        drag_df = pd.DataFrame()
        step_df = pd.DataFrame()
        a = 2 * np.pi / step
        for i, k in enumerate(k_to_test):

            df = pd.read_csv(path_1 + "/{:.2E}_{:.2E}_{:.2E}".format(k, r, step))
            bead_angle = df.iloc[:, 1].to_numpy()  # select first column turn into an array

            b = HandleData(bead_angle, data_type="angle")
            bounds = [(0, 1), (1E-22, 1E-19), (9.99E-22, 1E-15)]
            print("{}, {}, {}".format(a, drag_bead, k))
            sim = [a, drag_bead, k]
            list_v, *list_para = b.segment_data(bead_angle, bounds, sim, segment_length=500000, sub_seg_number=5,
                                                slide_coef=3,
                                                data_type="angle")

            k_sim = list_para[2]
            drag_sim = list_para[1]
            step_sim = list_para[0]

            r_k = [(k_s - k) / k for k_s in k_sim]  # calculate r for each k of the output k list fitted
            k_df = pd.concat([k_df, pd.DataFrame(data=r_k)], ignore_index=True, axis=1)
            r_drag = [(drag_s - drag_bead)/drag_bead for drag_s in drag_sim]
            drag_df = pd.concat([drag_df, pd.DataFrame(data=r_drag)], ignore_index=True, axis=1)
            r_step = [(step_s - a)/a for step_s in step_sim]
            step_df = pd.concat([step_df, pd.DataFrame(data=r_step)], ignore_index=True, axis=1)

        print("stepsize = {}".format(2*np.pi/a))
        lim_k = abs(k_df).max().max()
        lim_drag = abs(drag_df).max().max()
        lim_step = abs(step_df).max().max()
        fig, ax1 = plt.subplots(1, 3, figsize=(10, 6))
        ax1[1].set_title("Parameters over K_hook stepsize={:.2E} bead_radius = {:.2E}".format(2 * np.pi / step, r))
        ax1[0].set_yscale("symlog")
        ax1[1].set_yscale("symlog")
        ax1[2].set_yscale("symlog")
        ax1[0].grid()
        ax1[1].grid()
        ax1[2].grid()
        ax1[0].set_ylim(ymin=-10 * abs(lim_k),
                        ymax=10 * abs(lim_k))
        ax1[1].set_ylim(ymin=-10 * abs(lim_drag),
                        ymax=10 * abs(lim_drag))
        ax1[2].set_ylim(ymin=-10 * abs(lim_step),
                        ymax=10 * abs(lim_step))

        for i in range(len(k_df.index)):

            ax1[0].set_ylabel("r(K_hook)")
            ax1[0].plot(k_to_test, k_df.iloc[i, :], linewidth=0, marker="o", markersize=8)
            ax1[1].set_ylabel("r(Drag)")
            ax1[1].plot(k_to_test, drag_df.iloc[i, :], linewidth=0, marker="o", markersize=8)
            ax1[2].set_ylabel("r(stepsize)")
            ax1[2].plot(k_to_test, step_df.iloc[i, :], linewidth=0, marker="o", markersize=8)
        ax1[0].set_xlabel("K_hook")
        ax1[1].set_xlabel("K_hook")
        ax1[2].set_xlabel("K_hook")

        # plt.show()
        plt.savefig(path_to_save + "/k_fct_r_a/{:.2E}_{:.2E}_MAE_seg500k_sub5_s3.png".format(r, step))
        plt.close(fig)
#

# for r in drag_to_test:
#     faxen_1 = 1 - (1 / 8.) * (r / dist_beadsurf) ** 3
#     faxen_2 = 1 - (9 / 16.) * (r / dist_beadsurf) + (1. / 8) * (r / dist_beadsurf) ** 3 - (
#             45 / 256.) * (r / dist_beadsurf) ** 4 - (1. / 16) * (r / dist_beadsurf) ** 5
#     drag_bead = (8 * np.pi * eta * r ** 3) / faxen_1 + (
#             6 * np.pi * eta * r * axis_offset ** 2) / faxen_2
#     for k in k_to_test:
#
#         k_df = pd.DataFrame()
#         drag_df = pd.DataFrame()
#         step_df = pd.DataFrame()
#
#         for step in step_to_test:
#             a = 2 * np.pi / step
#             df = pd.read_csv(path_1 + "/{:.2E}_{:.2E}_{:.2E}".format(k, r, step))
#             bead_angle = df.iloc[:, 1].to_numpy()  # select first column turn into an array
#
#             b = HandleData(bead_angle, data_type="angle")
#
#             bounds = [(0, 1), (1E-22, 1E-18), (9.99E-25, 1E-14)]
#             sim = [a, drag_bead, k]
#             list_v, *list_para = b.segment_data(bead_angle, bounds, sim, segment_length=200000, sub_seg_number=4,
#                                                 slide_coef=2,
#                                                 data_type="angle")
#
#
#             k_sim = list_para[2]
#             drag_sim = list_para[1]
#             step_sim = list_para[0]
#
#             r_k = [(k_s - k) / k for k_s in k_sim]  # select last para_list to calculate r_k
#             k_df = pd.concat([k_df, pd.DataFrame(data=r_k)], ignore_index=True, axis=1)
#             r_drag = [(drag_s - drag_bead)/drag_bead for drag_s in drag_sim]
#             drag_df = pd.concat([drag_df, pd.DataFrame(data=r_drag)], ignore_index=True, axis=1)
#             r_step = [(step_s - a)/a for step_s in step_sim]
#             step_df = pd.concat([step_df, pd.DataFrame(data=r_step)], ignore_index=True, axis=1)
#
#         lim_k = abs(k_df).max().max()
#         lim_drag = abs(drag_df).max().max()
#         lim_step = abs(step_df).max().max()
#
#         fig, ax1 = plt.subplots(1, 3, figsize=(10, 6))
#
#         ax1[1].set_title("Parameters over stepsize stiffness={:.2E} bead_radius = {:.2E}".format(k, r))
#         ax1[0].set_yscale("symlog")
#         ax1[1].set_yscale("symlog")
#         ax1[2].set_yscale("symlog")
#         ax1[0].grid()
#         ax1[1].grid()
#         ax1[2].grid()
#         ax1[0].set_ylim(ymin=-10 * abs(lim_k),
#                         ymax=10 * abs(lim_k))
#         ax1[1].set_ylim(ymin=-10 * abs(lim_drag),
#                         ymax=10 * abs(lim_drag))
#         ax1[2].set_ylim(ymin=-10 * abs(lim_step),
#                         ymax=10 * abs(lim_step))
#         for i in range(len(k_df.index)):
#             ax1[0].set_ylabel("r(K_hook)")
#             ax1[0].plot(step_to_test, k_df.iloc[i, :], linewidth=0, marker="o", markersize=8)
#             ax1[1].set_ylabel("r(Drag)")
#             ax1[1].plot(step_to_test, drag_df.iloc[i, :], linewidth=0, marker="o", markersize=8)
#             ax1[2].set_ylabel("r(stepsize)")
#             ax1[2].plot(step_to_test, step_df.iloc[i, :], linewidth=0, marker="o", markersize=8)
#
#         ax1[0].set_xlabel("stepsize")
#         ax1[1].set_xlabel("stepsize")
#         ax1[2].set_xlabel("stepsize")
#         plt.show()
        # plt.savefig(path_to_save + "/a_fct_r_k/{:.2E}_{:.2E}_2.png".format(r, k))
        # plt.close()

# for k in k_to_test:
#     for step in step_to_test:
#
#         k_df = pd.DataFrame()
#         drag_df = pd.DataFrame()
#         step_df = pd.DataFrame()
#
#         a = 2 * np.pi / step
#         for r in drag_to_test:
#
#             faxen_1 = 1 - (1 / 8.) * (r / dist_beadsurf) ** 3
#             faxen_2 = 1 - (9 / 16.) * (r / dist_beadsurf) + (1. / 8) * (r / dist_beadsurf) ** 3 - (
#                     45 / 256.) * (r / dist_beadsurf) ** 4 - (1. / 16) * (r / dist_beadsurf) ** 5
#             drag_bead = (8 * np.pi * eta * r ** 3) / faxen_1 + (
#                     6 * np.pi * eta * r * axis_offset ** 2) / faxen_2
#             df = pd.read_csv(path_1 + "/{:.2E}_{:.2E}_{:.2E}".format(k, r, step))
#             bead_angle = df.iloc[:, 1].to_numpy()  # select first column turn into an array
#
#             b = HandleData(bead_angle, data_type="angle")
#
#             bounds = [(0, 1), (1E-22, 1E-19), (9.99E-25, 1E-14)]
#
#             list_v, *list_para = b.segment_data(bead_angle, bounds, segment_length=200000, sub_seg_number=4,
#                                                 slide_coef=2,
#                                                 data_type="angle")
#
#             k_sim = list_para[2]
#             drag_sim = list_para[1]
#             step_sim = list_para[0]
#
#             r_k = [(k_s - k) / k for k_s in k_sim]  # select last para_list to calculate r_k
#             k_df = pd.concat([pd.DataFrame(data=r_k), k_df], ignore_index=True, axis=1)
#             r_drag = [(drag_s - drag_bead)/drag_bead for drag_s in drag_sim]
#             drag_df = pd.concat([pd.DataFrame(data=r_drag), drag_df], ignore_index=True, axis=1)
#             r_step = [(step_s - a)/a for step_s in step_sim]
#             step_df = pd.concat([pd.DataFrame(data=r_step), step_df], ignore_index=True, axis=1)
#
#         fig, ax1 = plt.subplots(1, 3, figsize=(10, 6))
#
#         lim_k = abs(k_df).max().max()
#         lim_drag = abs(drag_df).max().max()
#         lim_step = abs(step_df).max().max()
#         ax1[1].set_title("Parameters over bead_radius stepsize={:.2E} stiffness={:.2E}".format(2*np.pi/step, k))
#
#         ax1[0].set_yscale("symlog")
#         ax1[1].set_yscale("symlog")
#         ax1[2].set_yscale("symlog")
#         ax1[0].grid()
#         ax1[1].grid()
#         ax1[2].grid()
#         ax1[0].set_ylim(ymin=-10 * abs(lim_k),
#                         ymax=10 * abs(lim_k))
#         ax1[1].set_ylim(ymin=-10 * abs(lim_drag),
#                         ymax=10 * abs(lim_drag))
#         ax1[2].set_ylim(ymin=-10 * abs(lim_step),
#                         ymax=10 * abs(lim_step))
#
#         for i in range(len(k_df.index)):
#             ax1[0].set_ylabel("r(K_hook)")
#             ax1[0].plot(drag_to_test, k_df.iloc[i, :], linewidth=0, marker="o", markersize=8)
#             ax1[1].set_ylabel("r(Drag)")
#             ax1[1].plot(drag_to_test, drag_df.iloc[i, :], linewidth=0, marker="o", markersize=8)
#             ax1[2].set_ylabel("r(stepsize)")
#             ax1[2].plot(drag_to_test, step_df.iloc[i, :], linewidth=0, marker="o", markersize=8)
#         ax1[0].set_xlabel("Radius")
#         ax1[1].set_xlabel("Radius")
#         ax1[2].set_xlabel("Radius")
#
#         plt.savefig(path_to_save + "/r_fct_k_a/{:.2E}_{:.2E}_2.png".format(k, step))




