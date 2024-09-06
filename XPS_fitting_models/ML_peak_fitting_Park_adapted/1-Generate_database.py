# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:13:51 2024

@author: ajulien & Romaric
"""

# %% Importation of packages
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import pickle
import os


# %% Inputs
main_data_folder = "C:/Users/rsallustre/Documents/XPS_fitting/"
session_name = "Fifteenth_test_23/08/2024"
data_folder = main_data_folder+session_name+"/"
database_folder = data_folder+"Database/"
os.makedirs(database_folder, exist_ok=True)

# Setting x-axis range and grid (setting the size of data)
energy_range_n = 401
energy_range = np.linspace(0, 15, energy_range_n)

# Definition of the sizes of the sub-databases
number_5 = 400
number_4 = 875
number_3 = 200
number_2 = 360
number_1 = 100


# %% Definition of Pseudo-Voigt function
def y(a, b, c, x):
    beta = 5.09791537e-01
    gamma = 4.41140472e-01
    y = c * ((0.7 * np.exp(-np.log(2) * (x - a) ** 2 / (beta * b) ** 2))
             + (0.3 / (1 + (x - a) ** 2 / (gamma * b) ** 2)))
    return y
# %% Sub database for peaks composed of five peaks
#
#
# Randomly generate entries of the database, composed of five peaks, each determined by three parameters
peak5_param = np.zeros((number_5, 5, 3))
for i in range(number_5):
    peak5_param[i][0] = [2 + 11.0 * np.random.rand(),  # random.uniform(2,13,1) #low & high variables
                         0.3 + 1.6 * np.random.rand(),
                         0.05 + np.random.rand()]

    peak5_param[i][1] = [2 + 11.0 * np.random.rand(),
                         0.3 + 1.6 * np.random.rand(),
                         0.05 + np.random.rand()]

    peak5_param[i][2] = [2 + 11.0 * np.random.rand(),
                         0.3 + 1.6 * np.random.rand(),
                         0.05 + np.random.rand()]

    peak5_param[i][3] = [2 + 11.0 * np.random.rand(),
                         0.3 + 1.6 * np.random.rand(),
                         0.05 + np.random.rand()]

    peak5_param[i][4] = [2 + 11.0 * np.random.rand(),
                         0.3 + 1.6 * np.random.rand(),
                         0.05 + np.random.rand()]

# Convert into python list
peak5_param2 = peak5_param.copy()
peak5_param2 = peak5_param2.tolist()

# Delete entries when peaks are too close from each other
t = 0
count = 0
R = 0.2
for j in peak5_param:  # Accessing 1 data out of 100
    # 0123
    if (j[0][0] - R * j[0][1] < j[4][0] < j[0][0] + R * j[0][1] or
        j[1][0] - R * j[1][1] < j[4][0] < j[1][0] + R * j[1][1] or
        j[2][0] - R * j[2][1] < j[4][0] < j[2][0] + R * j[2][1] or
            j[3][0] - R * j[3][1] < j[4][0] < j[3][0] + R * j[3][1]):
        del peak5_param2[count + t]
        t = t - 1

    # 0124
    elif (j[0][0] - R * j[0][1] < j[3][0] < j[0][0] + R * j[0][1] or
          j[1][0] - R * j[1][1] < j[3][0] < j[1][0] + R * j[1][1] or
          j[2][0] - R * j[2][1] < j[3][0] < j[2][0] + R * j[2][1] or
          j[4][0] - R * j[4][1] < j[3][0] < j[4][0] + R * j[4][1]):
        del peak5_param2[count + t]
        t = t - 1

    # 0134
    elif (j[0][0] - R * j[0][1] < j[2][0] < j[0][0] + R * j[0][1] or
          j[1][0] - R * j[1][1] < j[2][0] < j[1][0] + R * j[1][1] or
          j[3][0] - R * j[3][1] < j[2][0] < j[3][0] + R * j[3][1] or
          j[4][0] - R * j[4][1] < j[2][0] < j[4][0] + R * j[4][1]):
        del peak5_param2[count + t]
        t = t - 1

    # 0234
    elif (j[0][0] - R * j[0][1] < j[1][0] < j[0][0] + R * j[0][1] or
          j[2][0] - R * j[2][1] < j[1][0] < j[2][0] + R * j[2][1] or
          j[3][0] - R * j[3][1] < j[1][0] < j[3][0] + R * j[3][1] or
          j[4][0] - R * j[4][1] < j[1][0] < j[4][0] + R * j[4][1]):
        del peak5_param2[count + t]
        t = t - 1

    # 1233
    elif (j[1][0] - R * j[1][1] < j[0][0] < j[1][0] + R * j[1][1] or
          j[2][0] - R * j[2][1] < j[0][0] < j[2][0] + R * j[2][1] or
          j[3][0] - R * j[3][1] < j[0][0] < j[3][0] + R * j[3][1] or
          j[4][0] - R * j[4][1] < j[0][0] < j[4][0] + R * j[4][1]):
        del peak5_param2[count + t]
        t = t - 1

    count += 1

five_peaks_str = '\n'.join(["Sub database of sums of five peaks",
                            str(number_5)+" generated entries",
                            str(number_5+t)+" sufficiently separated peaks"])
print(five_peaks_str)


# %% Plot first entries of the database
# for i in range(5):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(energy_range, y(peak5_param2[i][0][0], peak5_param2[i][0][1], peak5_param2[i][0][2], energy_range) +
#              y(peak5_param2[i][1][0], peak5_param2[i][1][1], peak5_param2[i][1][2], energy_range) +
#              y(peak5_param2[i][2][0], peak5_param2[i][2][1], peak5_param2[i][2][2], energy_range) +
#              y(peak5_param2[i][3][0], peak5_param2[i][3][1], peak5_param2[i][3][2], energy_range) +
#              y(peak5_param2[i][4][0], peak5_param2[i][4][1], peak5_param2[i][4][2], energy_range), c="black")

#     for j in range(len(peak5_param2[i])):
#         plt.plot(energy_range, y(peak5_param2[i][j][0], peak5_param2[i][j][1], peak5_param2[i][j][2], energy_range))
#         plt.title(i)


# %% Add the five peaks together of a given database entry plus gaussian noise
peak5_graph = []
noise_level = 0.25
for i in peak5_param2:
    # Adding the five peaks together of a given database entry
    total_y = 0
    for j in i:
        total_y += y(j[0], j[1], j[2], energy_range)

    # Adding noise to the five peaks
    noise_graph = []
    for k in range(energy_range_n):
        noise_graph.append(np.random.rand() * noise_level - noise_level * 0.5)
    peak5_graph.append(total_y + noise_graph)


# %% Plot first entries of the database with noise
# for i in range(5):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(energy_range, peak5_graph[i], c="black")

#     for j in range(len(peak5_param2[i])):
#         plt.plot(energy_range, y(peak5_param2[i][j][0], peak5_param2[i][j][1], peak5_param2[i][j][2], energy_range))
#         plt.title(i)


# %% Identify peak with largest underlying area
peak5_label = []  # The peak with largest area is stored in a list
for i in peak5_param2:  # Loop over entries in 5 peaks database
    graph_sum = []
    for j in i:  # Loop over the 5 peaks of the entry
        # Compute underlying area
        graph_sum.append(sum(y(j[0], j[1], j[2], energy_range)))

    for k in i:  # Loop over the 5 peaks of the entry
        if sum(y(k[0], k[1], k[2], energy_range)) == max(graph_sum):
            # If the peak is asscociated to largest area, store its inputs
            peak5_label.append(deepcopy(k))


# %% Plot first entries of the database with noise and identified peak with largest area
for i in range(3):
    plt.figure(figsize=(10, 5))
    plt.plot(energy_range, peak5_graph[i], c="black")

    for j in peak5_param2[i]:
        plt.plot(energy_range, y(j[0], j[1], j[2], energy_range))

    plt.plot(energy_range, y(peak5_label[i][0], peak5_label[i][1],
             peak5_label[i][2], energy_range), c="red", label="biggest area")
    plt.grid(True)
    plt.xlabel("Energy")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title(i)
    plt.savefig(database_folder+"Generated_spectrum_5_peaks_" +
                str(i)+".jpg", dpi=300)

    plt.figure(figsize=(10, 5))
    plt.plot(energy_range, peak5_graph[i], c="black")
    plt.plot(energy_range, y(peak5_label[i][0], peak5_label[i][1],
             peak5_label[i][2], energy_range), c="red", label="Main contribution")
    plt.grid(True)
    plt.xlabel("Energy")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title(i)
    plt.savefig(database_folder+"Generated_spectrum_5_peaks_" +
                str(i)+"_A.jpg", dpi=300)

    plt.figure(figsize=(10, 5))
    plt.plot(energy_range, peak5_graph[i], c="black")
    plt.grid(True)
    plt.xlabel("Energy")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title(i)
    plt.savefig(database_folder+"Generated_spectrum_5_peaks_" +
                str(i)+"_B.jpg", dpi=300)

# %% Complete the label list with the number of peaks for each entry
for i in peak5_label:
    i.append(5)


#
#
# %% Sub database for peaks composed of four peaks
#
#
peak4_param = np.zeros((number_4, 4, 3))

for i in range(number_4):
    peak4_param[i][0] = [
        2 + 11.0 * np.random.rand(),
        0.3 + 1.6 * np.random.rand(),
        0.05 + np.random.rand(),
    ]
    peak4_param[i][1] = [
        2 + 11.0 * np.random.rand(),
        0.3 + 1.6 * np.random.rand(),
        0.05 + np.random.rand(),
    ]
    peak4_param[i][2] = [
        2 + 11.0 * np.random.rand(),
        0.3 + 1.6 * np.random.rand(),
        0.05 + np.random.rand(),
    ]
    peak4_param[i][3] = [
        2 + 11.0 * np.random.rand(),
        0.3 + 1.6 * np.random.rand(),
        0.05 + np.random.rand(),
    ]

peak4_param2 = peak4_param.copy()
peak4_param2 = peak4_param2.tolist()

t = 0
count = 0
R = 0.2

for j in peak4_param:  # 100개중 1개의 data 접근
    #     print('--',count,'--')

    # 012
    if (
        j[0][0] - R * j[0][1] < j[3][0] < j[0][0] + R * j[0][1]
        or j[1][0] - R * j[1][1] < j[3][0] < j[1][0] + R * j[1][1]
        or j[2][0] - R * j[2][1] < j[3][0] < j[2][0] + R * j[2][1]
    ):
        del peak4_param2[count + t]
        t = t - 1
    #         print('delete')

    # 013
    elif (
        j[0][0] - R * j[0][1] < j[2][0] < j[0][0] + R * j[0][1]
        or j[1][0] - R * j[1][1] < j[2][0] < j[1][0] + R * j[1][1]
        or j[3][0] - R * j[3][1] < j[2][0] < j[3][0] + R * j[3][1]
    ):
        del peak4_param2[count + t]
        t = t - 1
    #         print('delete')

    # 023
    elif (
        j[0][0] - R * j[0][1] < j[1][0] < j[0][0] + R * j[0][1]
        or j[2][0] - R * j[2][1] < j[1][0] < j[2][0] + R * j[2][1]
        or j[3][0] - R * j[3][1] < j[1][0] < j[3][0] + R * j[3][1]
    ):
        del peak4_param2[count + t]
        t = t - 1
    #         print('delete')

    # 123
    elif (
        j[1][0] - R * j[1][1] < j[0][0] < j[1][0] + R * j[1][1]
        or j[2][0] - R * j[2][1] < j[0][0] < j[2][0] + R * j[2][1]
        or j[3][0] - R * j[3][1] < j[0][0] < j[3][0] + R * j[3][1]
    ):
        del peak4_param2[count + t]
        t = t - 1
    #         print('delete')

    count += 1

four_peaks_str = '\n'.join(["Sub database of sums of four peaks",
                            str(number_4)+" generated entries",
                            str(number_4+t)+" sufficiently separated peaks"])
print(four_peaks_str)


# for i in range(5):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(
#         energy_range,
#         y(
#             peak4_param2[i][0][0],
#             peak4_param2[i][0][1],
#             peak4_param2[i][0][2],
#             energy_range,
#         )
#         + y(
#             peak4_param2[i][1][0],
#             peak4_param2[i][1][1],
#             peak4_param2[i][1][2],
#             energy_range,
#         )
#         + y(
#             peak4_param2[i][2][0],
#             peak4_param2[i][2][1],
#             peak4_param2[i][2][2],
#             energy_range,
#         )
#         + y(
#             peak4_param2[i][3][0],
#             peak4_param2[i][3][1],
#             peak4_param2[i][3][2],
#             energy_range,
#         ),
#         c="black",
#     )

#     for j in range(len(peak4_param2[i])):
#         plt.plot(
#             energy_range,
#             y(
#                 peak4_param2[i][j][0],
#                 peak4_param2[i][j][1],
#                 peak4_param2[i][j][2],
#                 energy_range,
#             ),
#         )
#         plt.title(i)


# for i in range(5):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(
#         energy_range,
#         y(
#             peak4_param2[i][0][0],
#             peak4_param2[i][0][1],
#             peak4_param2[i][0][2],
#             energy_range,
#         )
#         + y(
#             peak4_param2[i][1][0],
#             peak4_param2[i][1][1],
#             peak4_param2[i][1][2],
#             energy_range,
#         )
#         + y(
#             peak4_param2[i][2][0],
#             peak4_param2[i][2][1],
#             peak4_param2[i][2][2],
#             energy_range,
#         )
#         + y(
#             peak4_param2[i][3][0],
#             peak4_param2[i][3][1],
#             peak4_param2[i][3][2],
#             energy_range,
#         ),
#         c="black",
#     )
#     for j in range(len(peak4_param2[i])):
#         plt.plot(
#             energy_range,
#             y(
#                 peak4_param2[i][j][0],
#                 peak4_param2[i][j][1],
#                 peak4_param2[i][j][2],
#                 energy_range,
#             ),
#         )
#         plt.title(i)


peak4_graph = []

for i in peak4_param2:  # 58개 data중 1개 불러오기
    total_y = 0

    for j in i:  # 1개의 data안의 4개의 peak을 각각 불러오기
        total_y += y(j[0], j[1], j[2], energy_range)

    noise_level = 0.02
    noise_graph = []
    for k in range(energy_range_n):
        noise_graph.append(np.random.rand() * noise_level - noise_level * 0.5)

    peak4_graph.append(total_y + noise_graph)


# for i in range(5):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(energy_range, peak4_graph[i], c="black")

#     for j in range(len(peak4_param2[i])):
#         plt.plot(
#             energy_range,
#             y(
#                 peak4_param2[i][j][0],
#                 peak4_param2[i][j][1],
#                 peak4_param2[i][j][2],
#                 energy_range,
#             ),
#         )
#         plt.title(i)


peak4_label = []
for i in peak4_param2:  # 58개의 데이터중 1개의 데이터 뽑기
    graph_sum = []
    for j in i:  # 1개의 data안에 있는 4개의 peak의 label불러오기
        graph_sum.append(sum(y(j[0], j[1], j[2], energy_range)))

    for k in i:
        if sum(y(k[0], k[1], k[2], energy_range)) == max(graph_sum):
            peak4_label.append(deepcopy(k))


# peak 4개 따로

for i in range(3):
    plt.figure(figsize=(10, 5))
    plt.plot(energy_range, peak4_graph[i], c="black")
    for j in peak4_param2[i]:
        plt.plot(energy_range, y(j[0], j[1], j[2], energy_range))

    plt.plot(energy_range, y(peak4_label[i][0], peak4_label[i][1],
             peak4_label[i][2], energy_range), c="red", label="biggest area")
    plt.grid(True)
    plt.xlabel("Energy")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title(i)
    plt.savefig(database_folder+"Generated_spectrum_4_peaks_" +
                str(i)+".jpg", dpi=300)


for i in peak4_label:
    i.append(4)


#
#
# %% Sub database for peaks composed of three peaks
#
#
peak3_param = np.zeros((number_3, 3, 3))
for i in range(number_3):
    peak3_param[i][0] = [
        2 + 11.0 * np.random.rand(),
        0.3 + 1.6 * np.random.rand(),
        0.05 + np.random.rand(),
    ]
    peak3_param[i][1] = [
        2 + 11.0 * np.random.rand(),
        0.3 + 1.6 * np.random.rand(),
        0.05 + np.random.rand(),
    ]
    peak3_param[i][2] = [
        2 + 11.0 * np.random.rand(),
        0.3 + 1.6 * np.random.rand(),
        0.05 + np.random.rand(),
    ]

peak3_param2 = peak3_param.copy()
peak3_param2 = peak3_param2.tolist()


# peak 3개 따로

t = 0
count = 0
R = 0.2

for j in peak3_param:  # 100개중 1개의 data 접근
    #     print('--',count,'--')

    # 01
    if (
        j[0][0] - R * j[0][1] < j[2][0] < j[0][0] + R * j[0][1]
        or j[1][0] - R * j[1][1] < j[2][0] < j[1][0] + R * j[1][1]
    ):
        del peak3_param2[count + t]
        t = t - 1
    #         print('delete')

    # 02
    elif (
        j[0][0] - R * j[0][1] < j[1][0] < j[0][0] + R * j[0][1]
        or j[2][0] - R * j[2][1] < j[1][0] < j[2][0] + R * j[2][1]
    ):
        del peak3_param2[count + t]
        t = t - 1
    #         print('delete')

    # 12
    elif (
        j[1][0] - R * j[1][1] < j[0][0] < j[1][0] + R * j[1][1]
        or j[2][0] - R * j[2][1] < j[0][0] < j[2][0] + R * j[2][1]
    ):
        del peak3_param2[count + t]
        t = t - 1
    #         print('delete')

    count += 1

three_peaks_str = '\n'.join(["Sub database of sums of three peaks",
                             str(number_3)+" generated entries",
                             str(number_3+t)+" sufficiently separated peaks"])
print(three_peaks_str)


# for i in range(5):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(
#         energy_range,
#         y(
#             peak3_param2[i][0][0],
#             peak3_param2[i][0][1],
#             peak3_param2[i][0][2],
#             energy_range,
#         )
#         + y(
#             peak3_param2[i][1][0],
#             peak3_param2[i][1][1],
#             peak3_param2[i][1][2],
#             energy_range,
#         )
#         + y(
#             peak3_param2[i][2][0],
#             peak3_param2[i][2][1],
#             peak3_param2[i][2][2],
#             energy_range,
#         ),
#         c="black",
#     )

#     for j in range(len(peak3_param2[i])):
#         plt.plot(
#             energy_range,
#             y(
#                 peak3_param2[i][j][0],
#                 peak3_param2[i][j][1],
#                 peak3_param2[i][j][2],
#                 energy_range,
#             ),
#         )
#         plt.title(i)

# for i in range(5):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(
#         energy_range,
#         y(
#             peak3_param2[i][0][0],
#             peak3_param2[i][0][1],
#             peak3_param2[i][0][2],
#             energy_range,
#         )
#         + y(
#             peak3_param2[i][1][0],
#             peak3_param2[i][1][1],
#             peak3_param2[i][1][2],
#             energy_range,
#         )
#         + y(
#             peak3_param2[i][2][0],
#             peak3_param2[i][2][1],
#             peak3_param2[i][2][2],
#             energy_range,
#         ),
#         c="black",
#     )
#     for j in range(len(peak3_param2[i])):
#         plt.plot(
#             energy_range,
#             y(
#                 peak3_param2[i][j][0],
#                 peak3_param2[i][j][1],
#                 peak3_param2[i][j][2],
#                 energy_range,
#             ),
#         )
#         plt.title(i)


# peak 3개 따로
peak3_graph = []

for i in peak3_param2:  # 58개 data중 1개 불러오기
    total_y = 0

    for j in i:  # 1개의 data안의 4개의 peak을 각각 불러오기
        total_y += y(j[0], j[1], j[2], energy_range)

    noise_level = 0.02
    noise_graph = []
    for k in range(energy_range_n):
        noise_graph.append(np.random.rand() * noise_level - noise_level * 0.5)

    peak3_graph.append(total_y + noise_graph)

# for i in range(5):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(energy_range, peak3_graph[i], c="black")

#     for j in range(len(peak3_param2[i])):
#         plt.plot(
#             energy_range,
#             y(
#                 peak3_param2[i][j][0],
#                 peak3_param2[i][j][1],
#                 peak3_param2[i][j][2],
#                 energy_range,
#             ),
#         )
#         plt.title(i)


peak3_label = []
for i in peak3_param2:  # 58개의 데이터중 1개의 데이터 뽑기
    graph_sum = []
    for j in i:  # 1개의 data안에 있는 3개의 peak의 label불러오기
        graph_sum.append(sum(y(j[0], j[1], j[2], energy_range)))

    for k in i:
        if sum(y(k[0], k[1], k[2], energy_range)) == max(graph_sum):
            peak3_label.append(deepcopy(k))


for i in range(3):
    plt.figure(figsize=(10, 5))
    plt.plot(energy_range, peak3_graph[i], c="black")
    for j in peak3_param2[i]:
        plt.plot(energy_range, y(j[0], j[1], j[2], energy_range))

    plt.plot(energy_range, y(peak3_label[i][0], peak3_label[i][1],
             peak3_label[i][2], energy_range), c="red", label="biggest area")
    plt.grid(True)
    plt.xlabel("Energy")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title(i)
    plt.savefig(database_folder+"Generated_spectrum_3_peaks_" +
                str(i)+".jpg", dpi=300)


for i in peak3_label:
    i.append(3)


#
#
# %% Sub database for peaks composed of two peaks
#
#
peak2_param = np.zeros((number_2, 2, 3))
for i in range(number_2):
    peak2_param[i][0] = [
        2 + 11.0 * np.random.rand(),
        0.3 + 1.6 * np.random.rand(),
        0.05 + np.random.rand(),
    ]
    peak2_param[i][1] = [
        2 + 11.0 * np.random.rand(),
        0.3 + 1.6 * np.random.rand(),
        0.05 + np.random.rand(),
    ]

peak2_param2 = peak2_param.copy()
peak2_param2 = peak2_param2.tolist()


# peak 2개 따로

t = 0
count = 0
R = 0.2

for j in peak2_param:  # 100개중 1개의 data 접근
    #     print('--',count,'--')

    # 0
    if j[0][0] - R * j[0][1] < j[1][0] < j[0][0] + R * j[0][1]:
        del peak2_param2[count + t]
        t = t - 1
    #         print('delete')

    # 1
    elif j[1][0] - R * j[1][1] < j[0][0] < j[1][0] + R * j[1][1]:
        del peak2_param2[count + t]
        t = t - 1
    #         print('delete')

    count += 1

two_peaks_str = '\n'.join(["Sub database of sums of two peaks",
                           str(number_2)+" generated entries",
                           str(number_2+t)+" sufficiently separated peaks"])
print(two_peaks_str)

# for i in range(5):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(
#         energy_range,
#         y(
#             peak2_param2[i][0][0],
#             peak2_param2[i][0][1],
#             peak2_param2[i][0][2],
#             energy_range,
#         )
#         + y(
#             peak2_param2[i][1][0],
#             peak2_param2[i][1][1],
#             peak2_param2[i][1][2],
#             energy_range,
#         ),
#         c="black",
#     )

#     for j in range(len(peak2_param2[i])):
#         plt.plot(
#             energy_range,
#             y(
#                 peak2_param2[i][j][0],
#                 peak2_param2[i][j][1],
#                 peak2_param2[i][j][2],
#                 energy_range,
#             ),
#         )
#         plt.title(i)


# for i in range(5):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(
#         energy_range,
#         y(
#             peak2_param2[i][0][0],
#             peak2_param2[i][0][1],
#             peak2_param2[i][0][2],
#             energy_range,
#         )
#         + y(
#             peak2_param2[i][1][0],
#             peak2_param2[i][1][1],
#             peak2_param2[i][1][2],
#             energy_range,
#         ),
#         c="black",
#     )
#     for j in range(len(peak2_param2[i])):
#         plt.plot(
#             energy_range,
#             y(
#                 peak2_param2[i][j][0],
#                 peak2_param2[i][j][1],
#                 peak2_param2[i][j][2],
#                 energy_range,
#             ),
#         )
#         plt.title(i)


peak2_graph = []

for i in peak2_param2:  # 58개 data중 1개 불러오기
    total_y = 0

    for j in i:  # 1개의 data안의 2개의 peak을 각각 불러오기
        total_y += y(j[0], j[1], j[2], energy_range)

    noise_level = 0.02
    noise_graph = []
    for k in range(energy_range_n):
        noise_graph.append(np.random.rand() * noise_level - noise_level * 0.5)

    peak2_graph.append(total_y + noise_graph)


# for i in range(10):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(energy_range, peak2_graph[i], c="black")

#     for j in range(len(peak2_param2[i])):
#         plt.plot(
#             energy_range,
#             y(
#                 peak2_param2[i][j][0],
#                 peak2_param2[i][j][1],
#                 peak2_param2[i][j][2],
#                 energy_range,
#             ),
#         )
#         plt.title(i)


peak2_label = []
for i in peak2_param2:  # 58개의 데이터중 1개의 데이터 뽑기
    graph_sum = []
    for j in i:  # 1개의 data안에 있는 2개의 peak의 label불러오기
        graph_sum.append(sum(y(j[0], j[1], j[2], energy_range)))

    for k in i:
        if sum(y(k[0], k[1], k[2], energy_range)) == max(graph_sum):
            peak2_label.append(deepcopy(k))


# peak 2개 따로

for i in range(3):
    plt.figure(figsize=(10, 5))
    plt.plot(energy_range, peak2_graph[i], c="black")
    for j in peak2_param2[i]:
        plt.plot(energy_range, y(j[0], j[1], j[2], energy_range))

    plt.plot(energy_range, y(peak2_label[i][0], peak2_label[i][1],
             peak2_label[i][2], energy_range), c="red", label="biggest area")
    plt.grid(True)
    plt.xlabel("Energy")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title(i)
    plt.savefig(database_folder+"Generated_spectrum_2_peaks_" +
                str(i)+".jpg", dpi=300)


for i in peak2_label:
    i.append(2)


#
#
# %% Sub database for peaks composed of one peak
#
#
t = 0
count = 0
R = 0.2

peak1_param = np.zeros((number_1, 1, 3))

for i in range(number_1):
    peak1_param[i][0] = [
        2 + 11.0 * np.random.rand(),
        0.3 + 1.6 * np.random.rand(),
        0.05 + np.random.rand(),
    ]


peak1_param2 = peak1_param.copy()
peak1_param2 = peak1_param2.tolist()

one_peaks_str = '\n'.join(["Sub database of single peaks",
                           str(number_1)+" generated entries"])
print(one_peaks_str)


# for i in range(5):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(
#         energy_range,
#         y(
#             peak1_param2[i][0][0],
#             peak1_param2[i][0][1],
#             peak1_param2[i][0][2],
#             energy_range,
#         ),
#         c="black",
#     )

#     for j in range(len(peak1_param2[i])):
#         plt.plot(
#             energy_range,
#             y(
#                 peak1_param2[i][j][0],
#                 peak1_param2[i][j][1],
#                 peak1_param2[i][j][2],
#                 energy_range,
#             ),
#         )
#         plt.title(i)


# for i in range(5):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(
#         energy_range,
#         y(
#             peak1_param2[i][0][0],
#             peak1_param2[i][0][1],
#             peak1_param2[i][0][2],
#             energy_range,
#         ),
#         c="black",
#     )
#     for j in range(len(peak1_param2[i])):
#         plt.plot(
#             energy_range,
#             y(
#                 peak1_param2[i][j][0],
#                 peak1_param2[i][j][1],
#                 peak1_param2[i][j][2],
#                 energy_range,
#             ),
#         )
#         plt.title(i)


peak1_graph = []

for i in peak1_param2:  # 58개 data중 1개 불러오기
    total_y = 0

    for j in i:  # 1개의 data안의 1개의 peak을 각각 불러오기
        total_y += y(j[0], j[1], j[2], energy_range)

    noise_level = 0.02
    noise_graph = []
    for k in range(energy_range_n):
        noise_graph.append(np.random.rand() * noise_level - noise_level * 0.5)

    peak1_graph.append(total_y + noise_graph)


# for i in range(5):
#     plt.figure(figsize=(10, 5))
#     plt.ylim(0, 2)
#     plt.plot(energy_range, peak1_graph[i], c="black")

#     for j in range(len(peak1_param2[i])):
#         plt.plot(
#             energy_range,
#             y(
#                 peak1_param2[i][j][0],
#                 peak1_param2[i][j][1],
#                 peak1_param2[i][j][2],
#                 energy_range,
#             ),
#         )
#         plt.title(i)


peak1_label = []
for i in peak1_param2:  # 58개의 데이터중 1개의 데이터 뽑기
    graph_sum = []
    for j in i:  # 1개의 data안에 있는 1개의 peak의 label불러오기
        graph_sum.append(sum(y(j[0], j[1], j[2], energy_range)))

    for k in i:
        if sum(y(k[0], k[1], k[2], energy_range)) == max(graph_sum):
            peak1_label.append(deepcopy(k))


for i in range(3):
    plt.figure(figsize=(10, 5))
    plt.plot(energy_range, peak1_graph[i], c="black")
    for j in peak1_param2[i]:
        plt.plot(energy_range, y(j[0], j[1], j[2], energy_range))

    plt.plot(energy_range, y(peak1_label[i][0], peak1_label[i][1],
             peak1_label[i][2], energy_range), c="red", label="biggest area")
    plt.grid(True)
    plt.xlabel("Energy")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title(i)
    plt.savefig(database_folder+"Generated_spectrum_1_peak_" +
                str(i)+".jpg", dpi=300)


for i in peak1_label:
    i.append(1)


#
#
# %% Assembling the sub databases into one for machine learning
#
#
peak_label = (peak1_label + peak2_label +
              peak3_label + peak4_label + peak5_label)
peak_graph = (peak1_graph + peak2_graph +
              peak3_graph + peak4_graph + peak5_graph)
peak_param = (peak1_param2 + peak2_param2 +
              peak3_param2 + peak4_param2 + peak5_param2)

full_database_str = "Size of total database: "+str(len(peak_label))
print(full_database_str)

# Save database in pickle file
with open(database_folder+"Main_database.pkl", 'wb') as f:
    pickle.dump([energy_range, peak_label, peak_graph, peak_param], f)


# %% Shuffle database
before_shuffle = []
for i in zip(peak_graph, peak_label, peak_param):
    before_shuffle.append(i)

random.shuffle(before_shuffle)

after_shuffle_peak_graph = []
after_shuffle_peak_label = []
after_shuffle_peak_param = []
for i in range(len(before_shuffle)):
    after_shuffle_peak_graph.append(before_shuffle[i][0])
    after_shuffle_peak_label.append(before_shuffle[i][1])
    after_shuffle_peak_param.append(before_shuffle[i][2])


# %% Split database into training, validation and test databases
# Ratios for training:validation:test : 8:1:1
train_peak_label = after_shuffle_peak_label[: int(
    0.8*len(after_shuffle_peak_param))]
val_peak_label = after_shuffle_peak_label[int(
    0.8*len(after_shuffle_peak_param)): int(0.9*len(after_shuffle_peak_graph))]
test_peak_label = after_shuffle_peak_label[int(
    0.9*len(after_shuffle_peak_param)):]

train_peak = after_shuffle_peak_graph[: int(0.8*len(after_shuffle_peak_graph))]
val_peak = after_shuffle_peak_graph[int(
    0.8*len(after_shuffle_peak_graph)): int(0.9*len(after_shuffle_peak_graph))]
test_peak = after_shuffle_peak_graph[int(0.9 * len(after_shuffle_peak_graph)):]

train_peak_param = after_shuffle_peak_param[: int(
    0.8*len(after_shuffle_peak_param))]
val_peak_param = after_shuffle_peak_param[int(
    0.8*len(after_shuffle_peak_param)): int(0.9*len(after_shuffle_peak_graph))]
test_peak_param = after_shuffle_peak_param[int(
    0.9*len(after_shuffle_peak_param)):]

# Convolution: reshape database of spectra into 3d array: (database size x energy_range length x 1)
train_peak = np.array(train_peak).reshape(
    np.array(train_peak).shape[0], np.array(train_peak).shape[1], 1)
val_peak = np.array(val_peak).reshape(
    np.array(val_peak).shape[0], np.array(val_peak).shape[1], 1)
test_peak = np.array(test_peak).reshape(
    np.array(test_peak).shape[0], np.array(test_peak).shape[1], 1)

sub_databases_str = '\n'.join(["Size of training database: "+str(train_peak.shape[0]),
                               "Size of validation database: " +
                               str(val_peak.shape[0]),
                               "Size of test database: "+str(test_peak.shape[0])])
print(sub_databases_str)


# %% Save training, validation and test databases
with open(database_folder+"Training_database.pkl", 'wb') as f:
    pickle.dump([energy_range, train_peak_label,
                train_peak, train_peak_param], f)

with open(database_folder+"Validation_database.pkl", 'wb') as f:
    pickle.dump([energy_range, val_peak_label, val_peak, val_peak_param], f)

with open(database_folder+"Test_database.pkl", 'wb') as f:
    pickle.dump([energy_range, test_peak_label, test_peak, test_peak_param], f)

with open(database_folder+"Database_summary.txt", 'w') as f:
    f.writelines(five_peaks_str)
    f.writelines('\n')
    f.writelines(four_peaks_str)
    f.writelines('\n')
    f.writelines(three_peaks_str)
    f.writelines('\n')
    f.writelines(two_peaks_str)
    f.writelines('\n')
    f.writelines(one_peaks_str)
    f.writelines('\n')
    f.writelines(full_database_str)
    f.writelines('\n')
    f.writelines(sub_databases_str)
    f.close()
