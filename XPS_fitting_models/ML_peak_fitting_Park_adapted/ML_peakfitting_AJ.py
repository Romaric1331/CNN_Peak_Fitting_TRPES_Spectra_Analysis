#!/usr/bin/env python
# coding: utf-8

#%% Set working folder
import os
os.chdir("C:/Users/ajulien/Documents/Codes/GitHub/IA_spectro/2021Park_ML-peak-fitting/")


#%% Importation of libraries
from tensorflow.python.client import device_lib
import keras
import tensorflow as tf
# from keras import backend
import numpy as np
import random
import matplotlib.pyplot as plt


#%% Check compatibility with system
device_lib.list_local_devices()

print(keras.__version__)
print(tf.__version__)

print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))


#%% Setting x-axis range and grid (setting the size of data)
x = np.linspace(0, 15, 401)


#%% Definition of Pseudo-Voigt function
def y(a, b, c, x):
    beta = 5.09791537e-01
    gamma = 4.41140472e-01
    y = c * ((0.7 * np.exp(-np.log(2) * (x - a) ** 2 / (beta * b) ** 2))
        + (0.3 / (1 + (x - a) ** 2 / (gamma * b) ** 2)))
    return y


#%% Dataset creation - from 5 peak to 1 peak spectra
# Randomly generate entries of the database, composed of five peaks, each determined by three parameters
number = 50  # number of entries in the database
peak5_param = np.zeros((number, 5, 3))
for i in range(number):
    peak5_param[i][0] = [2 + 11.0 * np.random.rand(),
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

print("Number of deleted peaks: ", -t)
print("Size of remaining database: ", number + t)


#%% Plot some examples of entries of the database
for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 2)
    plt.plot(x, y(peak5_param2[i][0][0], peak5_param2[i][0][1], peak5_param2[i][0][2], x) +
             y(peak5_param2[i][1][0], peak5_param2[i][1][1], peak5_param2[i][1][2], x) +
             y(peak5_param2[i][2][0], peak5_param2[i][2][1], peak5_param2[i][2][2], x) +
             y(peak5_param2[i][3][0], peak5_param2[i][3][1], peak5_param2[i][3][2], x) +
             y(peak5_param2[i][4][0], peak5_param2[i][4][1], peak5_param2[i][4][2], x), c="black")
    
    for j in range(len(peak5_param2[i])):
        plt.plot(x, y(peak5_param2[i][j][0], peak5_param2[i][j][1], peak5_param2[i][j][2], x))
        plt.title(i)


#%% Add the five peaks together of a given database entry plus gaussian noise
peak5_graph = []
noise_level = 0.02
for i in peak5_param2:  # Loading 1 data out of 58
    # Adding the five peaks together of a given database entry
    total_y = 0
    for (j) in (i):  
        total_y += y(j[0], j[1], j[2], x)
    
    # Adding noise to the five peaks
    noise_graph = []
    for k in range(401):
        noise_graph.append(np.random.rand() * noise_level - noise_level * 0.5)
    peak5_graph.append(total_y + noise_graph) 


#%% Plot 
for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 1)
    plt.plot(x, peak5_graph[i], c="black")

    for j in range(len(peak5_param2[i])):
        plt.plot(
            x,
            y(
                peak5_param2[i][j][0],
                peak5_param2[i][j][1],
                peak5_param2[i][j][2],
                x,
            ),
        )
        plt.title(i)

# peak 5개 따로

peak5_label = []
for (
    i
) in (
    peak5_param2
):  # 58개의 데이터중 1개의 데이터 뽑기 - "Extracting 1 data out of 58 data."
    graph_sum = []
    for (
        j
    ) in (
        i
    ):  # 1개의 data안에 있는 5개의 peak의 label불러오기 - "Loading labels of 5 peaks within 1 data."
        graph_sum.append(sum(y(j[0], j[1], j[2], x)))

    for k in i:
        if sum(y(k[0], k[1], k[2], x)) == max(graph_sum):
            peak5_label.append(k)


# peak 5개 따로

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.plot(x, peak5_graph[i], c="black")

    for j in peak5_param2[i]:
        plt.plot(x, y(j[0], j[1], j[2], x))

    plt.plot(
        x,
        y(peak5_label[i][0], peak5_label[i][1], peak5_label[i][2], x),
        c="red",
        label="biggest area",
    )
    plt.title(i)
    plt.legend()


# In[8]:


print("peak5 data의 갯수 : ", number + t)


# In[9]:


num = 0
for i in peak5_label:
    i.append(5)
    num += 1
num


# In[10]:


# peak 4개 따로

number = 40  # 400000
peak4_param = np.zeros((number, 4, 3))

for i in range(number):
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


# peak 4개 따로
peak4_param2 = peak4_param.copy()

peak4_param2 = peak4_param2.tolist()


# peak 4개 따로

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

print("delete number", -t)
print(len(peak4_param2))
# peakk 4개 따로

import matplotlib.pyplot as plt

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 2)
    plt.plot(
        x,
        y(
            peak4_param2[i][0][0],
            peak4_param2[i][0][1],
            peak4_param2[i][0][2],
            x,
        )
        + y(
            peak4_param2[i][1][0],
            peak4_param2[i][1][1],
            peak4_param2[i][1][2],
            x,
        )
        + y(
            peak4_param2[i][2][0],
            peak4_param2[i][2][1],
            peak4_param2[i][2][2],
            x,
        )
        + y(
            peak4_param2[i][3][0],
            peak4_param2[i][3][1],
            peak4_param2[i][3][2],
            x,
        ),
        c="black",
    )

    for j in range(len(peak4_param2[i])):
        plt.plot(
            x,
            y(
                peak4_param2[i][j][0],
                peak4_param2[i][j][1],
                peak4_param2[i][j][2],
                x,
            ),
        )
        plt.title(i)

# peak 4개 따로

import matplotlib.pyplot as plt

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 2)
    plt.plot(
        x,
        y(
            peak4_param2[i][0][0],
            peak4_param2[i][0][1],
            peak4_param2[i][0][2],
            x,
        )
        + y(
            peak4_param2[i][1][0],
            peak4_param2[i][1][1],
            peak4_param2[i][1][2],
            x,
        )
        + y(
            peak4_param2[i][2][0],
            peak4_param2[i][2][1],
            peak4_param2[i][2][2],
            x,
        )
        + y(
            peak4_param2[i][3][0],
            peak4_param2[i][3][1],
            peak4_param2[i][3][2],
            x,
        ),
        c="black",
    )
    for j in range(len(peak4_param2[i])):
        plt.plot(
            x,
            y(
                peak4_param2[i][j][0],
                peak4_param2[i][j][1],
                peak4_param2[i][j][2],
                x,
            ),
        )
        plt.title(i)


# peak 4개 따로
peak4_graph = []

for i in peak4_param2:  # 58개 data중 1개 불러오기
    total_y = 0

    for j in i:  # 1개의 data안의 4개의 peak을 각각 불러오기
        total_y += y(j[0], j[1], j[2], x)

    noise_level = 0.02
    noise_graph = []
    for k in range(401):
        noise_graph.append(np.random.rand() * noise_level - noise_level * 0.5)

    peak4_graph.append(total_y + noise_graph)


# peak 4개 따로

import matplotlib.pyplot as plt

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 2)
    plt.plot(x, peak4_graph[i], c="black")

    for j in range(len(peak4_param2[i])):
        plt.plot(
            x,
            y(
                peak4_param2[i][j][0],
                peak4_param2[i][j][1],
                peak4_param2[i][j][2],
                x,
            ),
        )
        plt.title(i)


# peak 4개 따로

peak4_label = []
for i in peak4_param2:  # 58개의 데이터중 1개의 데이터 뽑기
    graph_sum = []
    for j in i:  # 1개의 data안에 있는 4개의 peak의 label불러오기
        graph_sum.append(sum(y(j[0], j[1], j[2], x)))

    for k in i:
        if sum(y(k[0], k[1], k[2], x)) == max(graph_sum):
            peak4_label.append(k)


# peak 4개 따로

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.plot(x, peak4_graph[i], c="black")

    for j in peak4_param2[i]:
        plt.plot(x, y(j[0], j[1], j[2], x))

    plt.plot(
        x,
        y(peak4_label[i][0], peak4_label[i][1], peak4_label[i][2], x),
        c="red",
        label="biggest area",
    )
    plt.title(i)
    plt.legend()


# In[11]:


print("peak4 data의 갯수 : ", number + t)


# In[12]:


num = 0
for i in peak4_label:
    i.append(4)
    num += 1
num


# In[13]:


# peak 3개 따로

# number = 4000000
number = 35  # 350000
peak3_param = np.zeros((number, 3, 3))

for i in range(number):
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


# peak 3개 따로
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

print("delete number", -t)
print(len(peak3_param2))

# peakk 3개 따로

import matplotlib.pyplot as plt

x = np.linspace(0, 15, 401)
for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 2)
    plt.plot(
        x,
        y(
            peak3_param2[i][0][0],
            peak3_param2[i][0][1],
            peak3_param2[i][0][2],
            x,
        )
        + y(
            peak3_param2[i][1][0],
            peak3_param2[i][1][1],
            peak3_param2[i][1][2],
            x,
        )
        + y(
            peak3_param2[i][2][0],
            peak3_param2[i][2][1],
            peak3_param2[i][2][2],
            x,
        ),
        c="black",
    )

    for j in range(len(peak3_param2[i])):
        plt.plot(
            x,
            y(
                peak3_param2[i][j][0],
                peak3_param2[i][j][1],
                peak3_param2[i][j][2],
                x,
            ),
        )
        plt.title(i)

# peak 3개 따로

import matplotlib.pyplot as plt

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 2)
    plt.plot(
        x,
        y(
            peak3_param2[i][0][0],
            peak3_param2[i][0][1],
            peak3_param2[i][0][2],
            x,
        )
        + y(
            peak3_param2[i][1][0],
            peak3_param2[i][1][1],
            peak3_param2[i][1][2],
            x,
        )
        + y(
            peak3_param2[i][2][0],
            peak3_param2[i][2][1],
            peak3_param2[i][2][2],
            x,
        ),
        c="black",
    )
    for j in range(len(peak3_param2[i])):
        plt.plot(
            x,
            y(
                peak3_param2[i][j][0],
                peak3_param2[i][j][1],
                peak3_param2[i][j][2],
                x,
            ),
        )
        plt.title(i)


# peak 3개 따로
peak3_graph = []

for i in peak3_param2:  # 58개 data중 1개 불러오기
    total_y = 0

    for j in i:  # 1개의 data안의 4개의 peak을 각각 불러오기
        total_y += y(j[0], j[1], j[2], x)

    noise_level = 0.02
    noise_graph = []
    for k in range(401):
        noise_graph.append(np.random.rand() * noise_level - noise_level * 0.5)

    peak3_graph.append(total_y + noise_graph)


# peak 3개 따로

import matplotlib.pyplot as plt

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 2)
    plt.plot(x, peak3_graph[i], c="black")

    for j in range(len(peak3_param2[i])):
        plt.plot(
            x,
            y(
                peak3_param2[i][j][0],
                peak3_param2[i][j][1],
                peak3_param2[i][j][2],
                x,
            ),
        )
        plt.title(i)


# peak 3개 따로

peak3_label = []
for i in peak3_param2:  # 58개의 데이터중 1개의 데이터 뽑기
    graph_sum = []
    for j in i:  # 1개의 data안에 있는 3개의 peak의 label불러오기
        graph_sum.append(sum(y(j[0], j[1], j[2], x)))

    for k in i:
        if sum(y(k[0], k[1], k[2], x)) == max(graph_sum):
            peak3_label.append(k)


# peak 3개 따로

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.plot(x, peak3_graph[i], c="black")

    for j in peak3_param2[i]:
        plt.plot(x, y(j[0], j[1], j[2], x))

    plt.plot(
        x,
        y(peak3_label[i][0], peak3_label[i][1], peak3_label[i][2], x),
        c="red",
        label="biggest area",
    )
    plt.title(i)
    plt.legend()


# In[14]:


print("peak3 data의 갯수 : ", number + t)


# In[15]:


num = 0
for i in peak3_label:
    i.append(3)
    num += 1
num


# In[16]:


# peak 2개 따로

number = 32  # 320000
peak2_param = np.zeros((number, 2, 3))

for i in range(number):
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


# peak 2개 따로
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

print("delete number", -t)
print(len(peak2_param2))

# peakk 2개 따로

import matplotlib.pyplot as plt

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 2)
    plt.plot(
        x,
        y(
            peak2_param2[i][0][0],
            peak2_param2[i][0][1],
            peak2_param2[i][0][2],
            x,
        )
        + y(
            peak2_param2[i][1][0],
            peak2_param2[i][1][1],
            peak2_param2[i][1][2],
            x,
        ),
        c="black",
    )

    for j in range(len(peak2_param2[i])):
        plt.plot(
            x,
            y(
                peak2_param2[i][j][0],
                peak2_param2[i][j][1],
                peak2_param2[i][j][2],
                x,
            ),
        )
        plt.title(i)

# peak 2개 따로

import matplotlib.pyplot as plt

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 2)
    plt.plot(
        x,
        y(
            peak2_param2[i][0][0],
            peak2_param2[i][0][1],
            peak2_param2[i][0][2],
            x,
        )
        + y(
            peak2_param2[i][1][0],
            peak2_param2[i][1][1],
            peak2_param2[i][1][2],
            x,
        ),
        c="black",
    )
    for j in range(len(peak2_param2[i])):
        plt.plot(
            x,
            y(
                peak2_param2[i][j][0],
                peak2_param2[i][j][1],
                peak2_param2[i][j][2],
                x,
            ),
        )
        plt.title(i)


# peak 2개 따로
peak2_graph = []

for i in peak2_param2:  # 58개 data중 1개 불러오기
    total_y = 0

    for j in i:  # 1개의 data안의 2개의 peak을 각각 불러오기
        total_y += y(j[0], j[1], j[2], x)

    noise_level = 0.02
    noise_graph = []
    for k in range(401):
        noise_graph.append(np.random.rand() * noise_level - noise_level * 0.5)

    peak2_graph.append(total_y + noise_graph)


# peak 2개 따로

import matplotlib.pyplot as plt

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 2)
    plt.plot(x, peak2_graph[i], c="black")

    for j in range(len(peak2_param2[i])):
        plt.plot(
            x,
            y(
                peak2_param2[i][j][0],
                peak2_param2[i][j][1],
                peak2_param2[i][j][2],
                x,
            ),
        )
        plt.title(i)


# peak 2개 따로

peak2_label = []
for i in peak2_param2:  # 58개의 데이터중 1개의 데이터 뽑기
    graph_sum = []
    for j in i:  # 1개의 data안에 있는 2개의 peak의 label불러오기
        graph_sum.append(sum(y(j[0], j[1], j[2], x)))

    for k in i:
        if sum(y(k[0], k[1], k[2], x)) == max(graph_sum):
            peak2_label.append(k)


# peak 2개 따로

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.plot(x, peak2_graph[i], c="black")

    for j in peak2_param2[i]:
        plt.plot(x, y(j[0], j[1], j[2], x))

    plt.plot(
        x,
        y(peak2_label[i][0], peak2_label[i][1], peak2_label[i][2], x),
        c="red",
        label="biggest area",
    )
    plt.title(i)
    plt.legend()


# In[21]:


print("peak2 data의 갯수 : ", number + t)


# In[22]:


num = 0
for i in peak2_label:
    i.append(2)
    num += 1
num


# In[17]:


# peak 1개 따로

t = 0
count = 0
R = 0.2

number = 30  # 300000
peak1_param = np.zeros((number, 1, 3))

for i in range(number):
    peak1_param[i][0] = [
        2 + 11.0 * np.random.rand(),
        0.3 + 1.6 * np.random.rand(),
        0.05 + np.random.rand(),
    ]


# peak 1개 따로
peak1_param2 = peak1_param.copy()

peak1_param2 = peak1_param2.tolist()


# peakk 2개 따로

import matplotlib.pyplot as plt

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 2)
    plt.plot(
        x,
        y(
            peak1_param2[i][0][0],
            peak1_param2[i][0][1],
            peak1_param2[i][0][2],
            x,
        ),
        c="black",
    )

    for j in range(len(peak1_param2[i])):
        plt.plot(
            x,
            y(
                peak1_param2[i][j][0],
                peak1_param2[i][j][1],
                peak1_param2[i][j][2],
                x,
            ),
        )
        plt.title(i)

# peak 1개 따로

import matplotlib.pyplot as plt

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 2)
    plt.plot(
        x,
        y(
            peak1_param2[i][0][0],
            peak1_param2[i][0][1],
            peak1_param2[i][0][2],
            x,
        ),
        c="black",
    )
    for j in range(len(peak1_param2[i])):
        plt.plot(
            x,
            y(
                peak1_param2[i][j][0],
                peak1_param2[i][j][1],
                peak1_param2[i][j][2],
                x,
            ),
        )
        plt.title(i)


# peak 1개 따로
peak1_graph = []

for i in peak1_param2:  # 58개 data중 1개 불러오기
    total_y = 0

    for j in i:  # 1개의 data안의 1개의 peak을 각각 불러오기
        total_y += y(j[0], j[1], j[2], x)

    noise_level = 0.02
    noise_graph = []
    for k in range(401):
        noise_graph.append(np.random.rand() * noise_level - noise_level * 0.5)

    peak1_graph.append(total_y + noise_graph)


# peak 1개 따로

import matplotlib.pyplot as plt

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 2)
    plt.plot(x, peak1_graph[i], c="black")

    for j in range(len(peak1_param2[i])):
        plt.plot(
            x,
            y(
                peak1_param2[i][j][0],
                peak1_param2[i][j][1],
                peak1_param2[i][j][2],
                x,
            ),
        )
        plt.title(i)


# peak 1개 따로

peak1_label = []
for i in peak1_param2:  # 58개의 데이터중 1개의 데이터 뽑기
    graph_sum = []
    for j in i:  # 1개의 data안에 있는 1개의 peak의 label불러오기
        graph_sum.append(sum(y(j[0], j[1], j[2], x)))

    for k in i:
        if sum(y(k[0], k[1], k[2], x)) == max(graph_sum):
            peak1_label.append(k)


# peak 1개 따로

for i in range(10):
    plt.figure(figsize=(10, 5))
    plt.plot(x, peak1_graph[i], c="black")

    for j in peak1_param2[i]:
        plt.plot(x, y(j[0], j[1], j[2], x))

    plt.plot(
        x,
        y(peak1_label[i][0], peak1_label[i][1], peak1_label[i][2], x),
        c="red",
        label="biggest area",
    )
    plt.title(i)
    plt.legend()


# In[18]:


print("peak1 data의 갯수 : ", number + t)


# In[19]:


num = 0
for i in peak1_label:
    i.append(1)
    num += 1
num


# In[34]:


# Assembling the data labels for ML data sets

peak_label = (
    peak1_label + peak2_label + peak3_label + peak4_label + peak5_label
)

peak_graph = (
    peak1_graph + peak2_graph + peak3_graph + peak4_graph + peak5_graph
)

peak_param = (
    peak1_param2 + peak2_param2 + peak3_param2 + peak4_param2 + peak5_param2
)


# ## Beginning of Machine Learning part

# In[61]:


# In future: shuffle TRPES data here

before_shuffle = []
for i in zip(peak_graph, peak_label, peak_param):
    before_shuffle.append(i)
# print(before_shuffle)
print(before_shuffle[0][2][0])
print(before_shuffle[1][2][0])

random.shuffle(before_shuffle)

after_shuffle_peak_graph = []
after_shuffle_peak_label = []
after_shuffle_peak_param = []
for i in range(len(before_shuffle)):
    after_shuffle_peak_graph.append(before_shuffle[i][0])
    after_shuffle_peak_label.append(before_shuffle[i][1])
    after_shuffle_peak_param.append(before_shuffle[i][2])
# 함수형 api를 위해 label 재분배 - "# Redistributing labels for functional API."
center = []
width = []
amp = []
peak_number = []

for i in range(len(peak_label)):
    try:
        print("center: " + str(after_shuffle_peak_label[i][0]))
        center.append(after_shuffle_peak_label[i][0])
        # print("width: " + str(after_shuffle_peak_label[i][1]))
        width.append(after_shuffle_peak_label[i][1])
        # print("amp: " + str(after_shuffle_peak_label[i][2]))
        amp.append(after_shuffle_peak_label[i][2])
        # print("peak_number: " + str(after_shuffle_peak_label[i][3]))
        peak_number.append(after_shuffle_peak_label[i][3])
    except:
        print(
            "Reached error at " + str(i) + " step",
        )


# In[60]:


# In[27]:


# train : val : test => 8: 1: 1로 나누기 -  "# Splitting train : val : test => 8: 1: 1."

train_peak = after_shuffle_peak_graph[
    : int(0.8 * len(after_shuffle_peak_graph))
]
val_peak = after_shuffle_peak_graph[
    int(0.8 * len(after_shuffle_peak_graph)) : int(
        0.9 * len(after_shuffle_peak_graph)
    )
]
test_peak = after_shuffle_peak_graph[
    int(0.9 * len(after_shuffle_peak_graph)) :
]

train_center = center[: int(0.8 * len(center))]
val_center = center[int(0.8 * len(center)) : int(0.9 * len(center))]
test_center = center[int(0.9 * len(center)) :]


train_width = width[: int(0.8 * len(width))]
val_width = width[int(0.8 * len(width)) : int(0.9 * len(width))]
test_width = width[int(0.9 * len(width)) :]

train_amp = amp[: int(0.8 * len(amp))]
val_amp = amp[int(0.8 * len(amp)) : int(0.9 * len(amp))]
test_amp = amp[int(0.9 * len(amp)) :]

train_peak_number = peak_number[: int(0.8 * len(peak_number))]
val_peak_number = peak_number[
    int(0.8 * len(peak_number)) : int(0.9 * len(peak_number))
]
test_peak_number = peak_number[int(0.9 * len(peak_number)) :]

train_peak_param = after_shuffle_peak_param[
    : int(0.8 * len(after_shuffle_peak_param))
]
val_peak_param = after_shuffle_peak_param[
    int(0.8 * len(after_shuffle_peak_param)) : int(
        0.9 * len(after_shuffle_peak_graph)
    )
]
test_peak_param = after_shuffle_peak_param[
    int(0.9 * len(after_shuffle_peak_param)) :
]

# conv1d를 위해 reshape로 1차원 늘리기

train_peak = np.array(train_peak).reshape(
    np.array(train_peak).shape[0], np.array(train_peak).shape[1], 1
)
val_peak = np.array(val_peak).reshape(
    np.array(val_peak).shape[0], np.array(val_peak).shape[1], 1
)
test_peak = np.array(test_peak).reshape(
    np.array(test_peak).shape[0], np.array(test_peak).shape[1], 1
)

print(train_peak.shape)
print(val_peak.shape)
print(test_peak.shape)


# In[28]:


####################### 돌리지마 멈춰 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ -  " Stop running "


# In[1]:


from keras.utils import plot_model
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    # Flatten,
    BatchNormalization,
    # Dropout,
    # Add,
    # advanced_activations,
)

# from keras.layers.convolutional import Conv1D
# from keras.layers.pooling import (
#     # MaxPooling1D,
#     # GlobalMaxPooling1D,
#     # GlobalAveragePooling1D,
#     AveragePooling1D,
# )

from keras.layers import (
    # MaxPooling1D,
    # GlobalMaxPooling1D,
    # GlobalAveragePooling1D,
    AveragePooling1D,
)
from keras.layers import Concatenate  # exchanged layers.merge with layers
from keras import layers


# In[2]:


from keras.layers import Activation, Multiply  # , Reshape


# In[ ]:


# finish
import numpy as np


# ## architecture

# ### SE- Dense- Resnet
#
# #### Densnet
# - concept : channel의 reuse
# - How
# - -  i) transition layer 사용하여 parameter 경량화 (=composition factor 0.5=논문에서 추천한 값)
# - - ii) concatenate로  projection block 을 connecting
# - review
# - - i) 모든 residual block에 connecting 하는 것보다  projection conection쓰이는 곳에만 연결하는게 더 좋은 효과
#
# #### The translation of the Korean text into English is:
#
# - Concept: Reuse of channels in channel-wise attention.
# - How:
#   - i) Use transition layer to reduce the number of parameters (= recommended value of composition factor 0.5 in the paper).
#   - ii) Connect projection block with concatenate.
# - Review:
#   - i) Connecting only where the projection connection is used is more effective than connecting to all residual blocks.
#
#

# In[6]:


# sparse dense block with no se block

x = np.linspace(0, 15, 401)

input_data = Input(shape=(len(x), 1))
r = 16
shortcut = 0
se = 0
Cf = 0.5
cardinality = 16

# resnet 1차
x = layers.Conv1D(
    32, 4, strides=2, padding="same", kernel_initializer="he_normal"
)(input_data)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    32, 4, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    32, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.MaxPooling1D(3, strides=2)(x)  # 나누기 2
# 443
# --------------------------------------


shortcut_dense = x

shortcut = x
shortcut = layers.Conv1D(
    64, 1, strides=1, padding="valid", kernel_initializer="he_normal"
)(shortcut)

# ------------------ first layer-1
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    64, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    64, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2


se = layers.GlobalAveragePooling1D()(x)  # global pooling
se = Dense(64 // r, kernel_initializer="he_normal")(se)  # FC
se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
se = Dense(64, kernel_initializer="he_normal")(se)  # FC
se = Activation("sigmoid")(se)  # Sigmoid

x = Multiply()([x, se])  # Scale

x = layers.Add()([x, shortcut])  # x.shape = (100,256)


# ----------------- first layer-2

shortcut = x


x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    64, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    64, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2

se = layers.GlobalAveragePooling1D()(x)
se = Dense(64 // r, kernel_initializer="he_normal")(se)
se = layers.LeakyReLU(alpha=0.01)(se)
se = Dense(64, kernel_initializer="he_normal")(se)
se = Activation("sigmoid")(se)  # Sigmoid
# se= Reshape([1,100])(se)

x = Multiply()([x, se])

x = layers.Add()([x, shortcut])

x = Concatenate()([x, shortcut_dense])  # Added layer into the concatenate

# se = layers.GlobalAveragePooling1D()(x)
# se = Dense(96 // r, kernel_initializer = 'he_normal')(se)
# se = layers.LeakyReLU(alpha = 0.01)(se)
# se = Dense(96, kernel_initializer = 'he_normal')(se)
# se = Activation('sigmoid')(se) # Sigmoid

# x = Multiply()([x,se])  # scale

# -----------------------------transition layer


x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(96 * Cf, 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # 나누기 2
x = AveragePooling1D(3, padding="same", strides=2)(x)  # overlapped pooling

se = layers.GlobalAveragePooling1D()(x)
se = Dense(48 // r, kernel_initializer="he_normal")(se)
se = layers.LeakyReLU(alpha=0.01)(se)
se = Dense(48, kernel_initializer="he_normal")(se)
se = Activation("sigmoid")(se)  # Sigmoid
x = Multiply()([x, se])  # Scale


# --------------------------------------
# ----------------- second layer-1

shortcut_dense = x

shortcut = x
shortcut = layers.Conv1D(
    128, 1, strides=1, padding="valid", kernel_initializer="he_normal"
)(shortcut)


x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    128, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    128, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2


se = layers.GlobalAveragePooling1D()(x)  # global pooling
se = Dense(128 // r, kernel_initializer="he_normal")(se)  # FC
se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
se = Dense(128, kernel_initializer="he_normal")(se)  # FC
se = Activation("sigmoid")(se)  # Sigmoid

x = Multiply()([x, se])  # Scale

x = layers.Add()([x, shortcut])


# ----------------- second layer-2

shortcut = x

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    128, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    128, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2


se = layers.GlobalAveragePooling1D()(x)
se = Dense(128 // r, kernel_initializer="he_normal")(se)
se = layers.LeakyReLU(alpha=0.01)(se)
se = Dense(128, kernel_initializer="he_normal")(se)
se = Activation("sigmoid")(se)  # Sigmoid

x = Multiply()([x, se])

x = layers.Add()([x, shortcut])

x = Concatenate()([x, shortcut_dense])

# se = layers.GlobalAveragePooling1D()(x)
# se = Dense(176 // r, kernel_initializer = 'he_normal')(se)
# se = layers.LeakyReLU(alpha = 0.01)(se)
# se = Dense(176, kernel_initializer = 'he_normal')(se)
# se = Activation('sigmoid')(se) # Sigmoid

# x = Multiply()([x,se])  # scale

# transition layer---------------------------------


x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    176 * Cf, 1, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2
x = AveragePooling1D(3, padding="same", strides=2)(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(88 // r, kernel_initializer="he_normal")(se)
se = layers.LeakyReLU(alpha=0.01)(se)
se = Dense(88, kernel_initializer="he_normal")(se)
se = Activation("sigmoid")(se)  # Sigmoid
x = Multiply()([x, se])  # Scale


# --------------------------------------
# ----------------- third layer-1


shortcut_dense = x

shortcut = x
shortcut = layers.Conv1D(
    256, 1, strides=1, padding="valid", kernel_initializer="he_normal"
)(shortcut)

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    256, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    256, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2


se = layers.GlobalAveragePooling1D()(x)  # global pooling
se = Dense(256 // r, kernel_initializer="he_normal")(se)  # FC
se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
se = Dense(256, kernel_initializer="he_normal")(se)  # FC
se = Activation("sigmoid")(se)  # Sigmoid

x = Multiply()([x, se])  # Scale

x = layers.Add()([x, shortcut])

# ----------------- third layer-2

shortcut = x

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    256, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    256, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2


se = layers.GlobalAveragePooling1D()(x)
se = Dense(256 // r, kernel_initializer="he_normal")(se)
se = layers.LeakyReLU(alpha=0.01)(se)
se = Dense(256, kernel_initializer="he_normal")(se)
se = Activation("sigmoid")(se)  # Sigmoid

x = Multiply()([x, se])

x = layers.Add()([x, shortcut])

x = Concatenate()([x, shortcut_dense])

# se = layers.GlobalAveragePooling1D()(x)
# se = Dense(344 // r, kernel_initializer = 'he_normal')(se)
# se = layers.LeakyReLU(alpha = 0.01)(se)
# se = Dense(344, kernel_initializer = 'he_normal')(se)
# se = Activation('sigmoid')(se) # Sigmoid

# x = Multiply()([x,se])  # scale


# transition layer---------------------------------


x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    334 * Cf, 1, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2
x = AveragePooling1D(3, padding="same", strides=2)(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(167 // r, kernel_initializer="he_normal")(se)
se = layers.LeakyReLU(alpha=0.01)(se)
se = Dense(167, kernel_initializer="he_normal")(se)
se = Activation("sigmoid")(se)  # Sigmoid
x = Multiply()([x, se])  # Scale


# --------------------------------------
# ----------------- four layer-1


shortcut_dense = x

shortcut = x
shortcut = layers.Conv1D(
    512, 1, strides=1, padding="valid", kernel_initializer="he_normal"
)(shortcut)


x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    512, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    512, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2

se = layers.GlobalAveragePooling1D()(x)  # global pooling
se = Dense(512 // r, kernel_initializer="he_normal")(se)  # FC
se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
se = Dense(512, kernel_initializer="he_normal")(se)  # FC
se = Activation("sigmoid")(se)  # Sigmoid

x = Multiply()([x, se])  # Scale

x = layers.Add()([x, shortcut])

# ----------------- four layer-2

shortcut = x

x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    512, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2

x = layers.BatchNormalization()(x)  # 786,994
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    512, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2


se = layers.GlobalAveragePooling1D()(x)
se = Dense(512 // r, kernel_initializer="he_normal")(se)
se = layers.LeakyReLU(alpha=0.01)(se)
se = Dense(512, kernel_initializer="he_normal")(se)
se = Activation("sigmoid")(se)  # Sigmoid

x = Multiply()([x, se])

x = layers.Add()([x, shortcut])

x = Concatenate()([x, shortcut_dense])

# se = layers.GlobalAveragePooling1D()(x)
# se = Dense(679 // r, kernel_initializer = 'he_normal')(se)
# se = layers.LeakyReLU(alpha = 0.01)(se)
# se = Dense(679, kernel_initializer = 'he_normal')(se)
# se = Activation('sigmoid')(se) # Sigmoid

# x = Multiply()([x,se])  # scale

# transition layer---------------------------------


x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    679 * Cf, 1, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2
x = AveragePooling1D(3, padding="same", strides=2)(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(339 // r, kernel_initializer="he_normal")(se)
se = layers.LeakyReLU(alpha=0.01)(se)
se = Dense(339, kernel_initializer="he_normal")(se)
se = Activation("sigmoid")(se)  # Sigmoid
x = Multiply()([x, se])  # Scale


# --------------------------------------
# --------------------------------------


x = layers.GlobalAveragePooling1D()(x)

# and BN을 확인해보자
# pre-activation했는지
# transition layer사용해야되는지 -> identity layer만 사용햐도되는지


total_center1 = Dense(
    100, name="total_center1", kernel_initializer="he_normal"
)(x)
center_Batchnormalization = BatchNormalization()(total_center1)
total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
total_center3 = Dense(
    1,
    activation="linear",
    name="total_center3",
    kernel_initializer="he_normal",
)(total_center1_act)

total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(
    x
)
width_Batchnormalization = BatchNormalization()(total_width1)
total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
total_width3 = Dense(
    1, activation="linear", name="total_width3", kernel_initializer="he_normal"
)(total_width1_act)

total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
amp_Batchnormalization = BatchNormalization()(total_amp1)
total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
total_amp3 = Dense(
    1, activation="linear", name="total_amp3", kernel_initializer="he_normal"
)(total_amp1_act)

total_peak_number1 = Dense(
    100, name="total_peak_number1", kernel_initializer="he_normal"
)(x)
peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(
    peak_number_Batchnormalization
)
total_peak_number3 = Dense(
    1,
    activation="linear",
    name="total_peak_number3",
    kernel_initializer="he_normal",
)(total_peak_number1_act)


model_sparse_densenet = Model(
    inputs=input_data,
    outputs=[total_center3, total_width3, total_amp3, total_peak_number3],
)
print(model_sparse_densenet.summary())


# In[7]:
# sparse net with mae

model_sparse_densenet.compile(
    optimizer="adam",
    loss={
        "total_center3": "mae",
        "total_width3": "mae",
        "total_amp3": "mae",
        "total_peak_number3": "mae",
    },
    loss_weights={
        "total_center3": 1,
        "total_width3": 10,
        "total_amp3": 20,
        "total_peak_number3": 2,
    },
    metrics=["mae"],
)


# In[8]:


#  콜백설정 - "Setting up callbacks"
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

print(os.getcwd())

model_checkpoint = ModelCheckpoint(
    "best_model_sparse_densenet_mae.h5", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)


# In[9]:


models_sparse_densenet = model_sparse_densenet.fit(
    train_peak,
    [
        np.array(train_center),
        np.array(train_width),
        np.array(train_amp),
        np.array(train_peak_number),
    ],
    epochs=5,
    batch_size=512,
    validation_data=(
        val_peak,
        [
            np.array(val_center),
            np.array(val_width),
            np.array(val_amp),
            np.array(val_peak_number),
        ],
    ),
    callbacks=[model_checkpoint, reduce_lr],
    shuffle=True,
)


# In[117]:


from tensorflow.keras.models import load_model

best_model_sparse_densenet = load_model(
    "best_model_sparse_densenet_mae.h5"  # All the load models are not available - fix save path !
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지 - "# Loaded the model with the best performance."
best_model_sparse_densenet.summary()


# In[118]:


prediction_sparse_densenet = best_model_sparse_densenet.predict(test_peak)
print(len(prediction_sparse_densenet))
print(np.array(prediction_sparse_densenet).shape)


# In[119]:


loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_sparse_densenet[0][i][0]))
    loss_width += abs((test_width[i] - prediction_sparse_densenet[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_sparse_densenet[2][i][0]))
    loss_peak_number += abs(
        (test_peak_number[i] - prediction_sparse_densenet[3][i][0])
    )

sparse_densenet_loss_center_mae = loss_center / len(test_center)
sparse_densenet_loss_width_mae = loss_width / len(test_center)
sparse_densenet_loss_amp_mae = loss_amp / len(test_center)
sparse_densenet_loss_peak_number_mae = loss_peak_number / len(test_center)


# In[120]:


print(sparse_densenet_loss_center_mae)
print(sparse_densenet_loss_width_mae)
print(sparse_densenet_loss_amp_mae)
print(sparse_densenet_loss_peak_number_mae)

# mse
# 0.10023051886639821
# 0.022580907377600164
# 0.016069925147315128
# 0.03569239818655941

# mae
# 0.08271695761466934
# 0.01806041833750177
# 0.01273618048286805
# 0.023920056910235743

# %%
# sparse dense net with mse

model_sparse_densenet.compile(
    optimizer="adam",
    loss={
        "total_center3": "mse",
        "total_width3": "mse",
        "total_amp3": "mse",
        "total_peak_number3": "mse",
    },
    loss_weights={
        "total_center3": 1,
        "total_width3": 10,
        "total_amp3": 20,
        "total_peak_number3": 2,
    },
    metrics=["mse"],
)


# In[8]:


#  콜백설정 - "Setting up callbacks"
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

print(os.getcwd())

model_checkpoint = ModelCheckpoint(
    "best_model_sparse_densenet_mse.h5", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)


# In[9]:


models_sparse_densenet = model_sparse_densenet.fit(
    train_peak,
    [
        np.array(train_center),
        np.array(train_width),
        np.array(train_amp),
        np.array(train_peak_number),
    ],
    epochs=5,
    batch_size=512,
    validation_data=(
        val_peak,
        [
            np.array(val_center),
            np.array(val_width),
            np.array(val_amp),
            np.array(val_peak_number),
        ],
    ),
    callbacks=[model_checkpoint, reduce_lr],
    shuffle=True,
)


# In[117]:


from tensorflow.keras.models import load_model

best_model_sparse_densenet = load_model(
    "best_model_sparse_densenet_mse.h5"  # All the load models are not available - fix save path !
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지 - "# Loaded the model with the best performance."
best_model_sparse_densenet.summary()


# In[118]:


prediction_sparse_densenet = best_model_sparse_densenet.predict(test_peak)
print(len(prediction_sparse_densenet))
print(np.array(prediction_sparse_densenet).shape)


# In[119]:


loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += np.square(
        (test_center[i] - prediction_sparse_densenet[0][i][0])
    )
    loss_width += np.square(
        (test_width[i] - prediction_sparse_densenet[1][i][0])
    )
    loss_amp += np.square((test_amp[i] - prediction_sparse_densenet[2][i][0]))
    loss_peak_number += np.square(
        (test_peak_number[i] - prediction_sparse_densenet[3][i][0])
    )

sparse_densenet_loss_center_mse = np.sqrt(loss_center / len(test_center))
sparse_densenet_loss_width_mse = np.sqrt(loss_width / len(test_center))
sparse_densenet_loss_amp_mse = np.sqrt(loss_amp / len(test_center))
sparse_densenet_loss_peak_number_mse = np.sqrt(
    loss_peak_number / len(test_center)
)


# In[120]:


print(sparse_densenet_loss_center_mse)
print(sparse_densenet_loss_width_mse)
print(sparse_densenet_loss_amp_mse)
print(sparse_densenet_loss_peak_number_mse)

# %%


# ### SE-Resnet
#
# #### SE-block
# - concept
# - - i) 압축, 펌핑을 통한 channel의 재보정
# - - ii) 10%이내의 적은 parameter투자를 통해 성능향상
# - - iii) flexible 하여 모든 model에 연결 가능
#
# - How
# - - i) GlobalAveragePooling1D를 통해 압축
# - - ii) r (= Reduction ratio) = 16 (논문에서 추천한 값) 값을 이용하여 점차적으로 펌핑
# - - iii) 모든 residual block에 사용 (identity connection, projection connection)
#

# In[150]:


# origin Se-resnet


# In[151]:


x = np.linspace(0, 15, 401)

input_data = Input(shape=(len(x), 1))
r = 16
shortcut = 0
se = 0

# resnet 1차
x = layers.Conv1D(
    32, 4, strides=2, padding="same", kernel_initializer="he_normal"
)(input_data)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    32, 4, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    32, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.MaxPooling1D(3, strides=2)(x)  # 나누기 2
# 443
# --------------------------------------

shortcut = x
shortcut = layers.Conv1D(
    64, 1, strides=1, padding="valid", kernel_initializer="he_normal"
)(shortcut)


x = layers.Conv1D(
    64, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    64, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)  # global pooling
se = Dense(64 // r, kernel_initializer="he_normal")(se)  # FC
se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
se = Dense(64, kernel_initializer="he_normal")(se)  # FC
se = Activation("sigmoid")(se)  # Sigmoid
# se= Reshape([1,64])(se)

x = Multiply()([x, se])  # Scale

x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)


shortcut = x
x = layers.Conv1D(
    64, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    64, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(64 // r, kernel_initializer="he_normal")(se)
se = layers.LeakyReLU(alpha=0.01)(se)
se = Dense(64, kernel_initializer="he_normal")(se)
se = Activation("sigmoid")(se)  # Sigmoid
# se= Reshape([1,64])(se)

x = Multiply()([x, se])


x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)


# --------------------------------------

shortcut = x
shortcut = layers.Conv1D(
    128, 1, strides=2, padding="valid", kernel_initializer="he_normal"
)(shortcut)

x = layers.Conv1D(
    128, 3, strides=2, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    128, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)  # global pooling
se = Dense(128 // r, kernel_initializer="he_normal")(se)  # FC
se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
se = Dense(128, kernel_initializer="he_normal")(se)  # FC
se = Activation("sigmoid")(se)  # Sigmoid
# se= Reshape([1,128])(se)

x = Multiply()([x, se])  # Scale

x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)

shortcut = x
x = layers.Conv1D(
    128, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # identity shortcut
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    128, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(128 // r, kernel_initializer="he_normal")(se)
se = layers.LeakyReLU(alpha=0.01)(se)
se = Dense(128, kernel_initializer="he_normal")(se)
se = Activation("sigmoid")(se)  # Sigmoid
# se= Reshape([1,128])(se)

x = Multiply()([x, se])

x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)


# --------------------------------------

shortcut = x
shortcut = layers.Conv1D(
    256, 1, strides=2, padding="valid", kernel_initializer="he_normal"
)(shortcut)

x = layers.Conv1D(
    256, 3, strides=2, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    256, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)  # global pooling
se = Dense(256 // r, kernel_initializer="he_normal")(se)  # FC
se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
se = Dense(256, kernel_initializer="he_normal")(se)  # FC
se = Activation("sigmoid")(se)  # Sigmoid
# se= Reshape([1,256])(se)

x = Multiply()([x, se])  # Scale

x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)

shortcut = x
x = layers.Conv1D(
    256, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # identity shortcut
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    256, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(256 // r, kernel_initializer="he_normal")(se)
se = layers.LeakyReLU(alpha=0.01)(se)
se = Dense(256, kernel_initializer="he_normal")(se)
se = Activation("sigmoid")(se)  # Sigmoid
# se= Reshape([1,256])(se)

x = Multiply()([x, se])

x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)

# --------------------------------------

shortcut = x
shortcut = layers.Conv1D(
    512, 1, strides=2, padding="valid", kernel_initializer="he_normal"
)(shortcut)

x = layers.Conv1D(
    512, 3, strides=2, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    512, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)  # global pooling
se = Dense(512 // r, kernel_initializer="he_normal")(se)  # FC
se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
se = Dense(512, kernel_initializer="he_normal")(se)  # FC
se = Activation("sigmoid")(se)  # Sigmoid
# se= Reshape([1,512])(se)

x = Multiply()([x, se])  # Scale

x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)

shortcut = x
x = layers.Conv1D(
    512, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # identity shortcut
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    512, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)

se = layers.GlobalAveragePooling1D()(x)
se = Dense(512 // r, kernel_initializer="he_normal")(se)
se = layers.LeakyReLU(alpha=0.01)(se)
se = Dense(512, kernel_initializer="he_normal")(se)
se = Activation("sigmoid")(se)  # Sigmoid
# se= Reshape([1,512])(se)

x = Multiply()([x, se])

x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)


# --------------------------------------

x = layers.GlobalAveragePooling1D()(x)

# and BN을 확인해보자


total_center1 = Dense(
    100, name="total_center1", kernel_initializer="he_normal"
)(x)
center_Batchnormalization = BatchNormalization()(total_center1)
total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
total_center3 = Dense(
    1,
    activation="linear",
    name="total_center3",
    kernel_initializer="he_normal",
)(total_center1_act)

total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(
    x
)
width_Batchnormalization = BatchNormalization()(total_width1)
total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
total_width3 = Dense(
    1, activation="linear", name="total_width3", kernel_initializer="he_normal"
)(total_width1_act)

total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
amp_Batchnormalization = BatchNormalization()(total_amp1)
total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
total_amp3 = Dense(
    1, activation="linear", name="total_amp3", kernel_initializer="he_normal"
)(total_amp1_act)

total_peak_number1 = Dense(
    100, name="total_peak_number1", kernel_initializer="he_normal"
)(x)
peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(
    peak_number_Batchnormalization
)
total_peak_number3 = Dense(
    1,
    activation="linear",
    name="total_peak_number3",
    kernel_initializer="he_normal",
)(total_peak_number1_act)


model_se_resnet = Model(
    inputs=input_data,
    outputs=[total_center3, total_width3, total_amp3, total_peak_number3],
)
print(model_se_resnet.summary())


# In[152]:


plot_model(model_se_resnet, show_shapes=True)


# In[153]:

# se resnet with mae

model_se_resnet.compile(
    optimizer="adam",
    loss={
        "total_center3": "mae",
        "total_width3": "mae",
        "total_amp3": "mae",
        "total_peak_number3": "mae",
    },
    loss_weights={
        "total_center3": 1,
        "total_width3": 10,
        "total_amp3": 20,
        "total_peak_number3": 2,
    },
    metrics=["mae"],
)


# In[154]:


#  콜백설정 -  "# Callback settings."
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_model_se_resnet_mae.h5", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)


# In[155]:


models_se_resnet = model_se_resnet.fit(
    train_peak,
    [
        np.array(train_center),
        np.array(train_width),
        np.array(train_amp),
        np.array(train_peak_number),
    ],
    epochs=5,
    batch_size=5,
    validation_data=(
        val_peak,
        [
            np.array(val_center),
            np.array(val_width),
            np.array(val_amp),
            np.array(val_peak_number),
        ],
    ),
    callbacks=[model_checkpoint, reduce_lr],
    shuffle=True,
)


# In[121]:


# se-resnet

from tensorflow.keras.models import load_model

best_model_se_resnet = load_model(
    "best_model_se_resnet_mae.h5"
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_se_resnet.summary()


# In[122]:


prediction_se_resnet = best_model_se_resnet.predict(test_peak)
print(len(prediction_se_resnet))
print(np.array(prediction_se_resnet).shape)


# In[124]:


loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_se_resnet[0][i][0]))
    loss_width += abs((test_width[i] - prediction_se_resnet[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_se_resnet[2][i][0]))
    loss_peak_number += abs(
        (test_peak_number[i] - prediction_se_resnet[3][i][0])
    )

se_resnet_loss_center_mae = loss_center / len(test_center)
se_resnet_loss_width_mae = loss_width / len(test_center)
se_resnet_loss_amp_mae = loss_amp / len(test_center)
se_resnet_loss_peak_number_mae = loss_peak_number / len(test_center)


# In[153]:


print(se_resnet_loss_center_mae)
print(se_resnet_loss_width_mae)
print(se_resnet_loss_amp_mae)
print(se_resnet_loss_peak_number_mae)


# mse
# 0.11536822213987753
# 0.02587420437073307
# 0.021748055343352845
# 0.050200699981981844


# mae
# 0.08127153942264843
# 0.017654453027160018
# 0.01247379819761519
# 0.02911586313333888

# 0.08095316096541305
# 0.017308590367836604
# 0.012174876559596785
# 0.028306338731499976

# %%
model_se_resnet.compile(
    optimizer="adam",
    loss={
        "total_center3": "mse",
        "total_width3": "mse",
        "total_amp3": "mse",
        "total_peak_number3": "mse",
    },
    loss_weights={
        "total_center3": 1,
        "total_width3": 10,
        "total_amp3": 20,
        "total_peak_number3": 2,
    },
    metrics=["mse"],
)


# In[154]:


#  콜백설정 -  "# Callback settings."
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_model_se_resnet_mse.h5", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)


# In[155]:


models_se_resnet = model_se_resnet.fit(
    train_peak,
    [
        np.array(train_center),
        np.array(train_width),
        np.array(train_amp),
        np.array(train_peak_number),
    ],
    epochs=5,
    batch_size=5,
    validation_data=(
        val_peak,
        [
            np.array(val_center),
            np.array(val_width),
            np.array(val_amp),
            np.array(val_peak_number),
        ],
    ),
    callbacks=[model_checkpoint, reduce_lr],
    shuffle=True,
)


# In[121]:


# se-resnet

from tensorflow.keras.models import load_model

best_model_se_resnet = load_model(
    "best_model_se_resnet_mse.h5"
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_se_resnet.summary()


# In[122]:


prediction_se_resnet = best_model_se_resnet.predict(test_peak)
print(len(prediction_se_resnet))
print(np.array(prediction_se_resnet).shape)


# In[124]:


loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += np.square((test_center[i] - prediction_se_resnet[0][i][0]))
    loss_width += np.square((test_width[i] - prediction_se_resnet[1][i][0]))
    loss_amp += np.square((test_amp[i] - prediction_se_resnet[2][i][0]))
    loss_peak_number += np.square(
        (test_peak_number[i] - prediction_se_resnet[3][i][0])
    )

se_resnet_loss_center_mse = np.sqrt(loss_center / len(test_center))
se_resnet_loss_width_mse = np.sqrt(loss_width / len(test_center))
se_resnet_loss_amp_mse = np.sqrt(loss_amp / len(test_center))
se_resnet_loss_peak_number_mse = np.sqrt(loss_peak_number / len(test_center))


# In[153]:


print(se_resnet_loss_center_mse)
print(se_resnet_loss_width_mse)
print(se_resnet_loss_amp_mse)
print(se_resnet_loss_peak_number_mse)

# %%

# ### Resnet
# - concept
# - -  i) residual connection을 통해 function 재설정
# - How
# - - i) pre activation( 논문에서 추천한 sumsampling순서)
# - - ii) 쌓을수록 resnet의 장점이 두드러지지만 vggnet과 비교를 위해 4개의 block으로 8개의 layer을 쌓음
# - review
# - - (64x2-128x2-256x2-512x2 총 8개의 convolution layers 사용)
# - - resnet의 장점인 depth의 극대화를 하지 않고 vggnet과 같이 8layer밖에 되지 않아 성능이 조금밖에 차이가 없음
#
#
# The translation of the Korean text into English is:
# ##### Resnet
#
# - Concept:
#     -- i) Resetting function through residual connections.
# - How:
#     -- i) Pre-activation (recommended subsampling order in the paper).
#     -- ii) Stacked with 4 blocks and 8 layers to compare with VGGNet and highlight the advantages of ResNet.
# - Review:
#     Used a total of 8 convolution layers (64x2-128x2-256x2-512x2).
#     Performance difference is not significant compared to VGGNet, as ResNet does not maximize the depth like VGGNet and is limited to only 8 layers.

# In[160]:


############################# resnet ###########################3


# In[161]:


# resnet not pre activation
# original


# In[162]:


# resnet not pre activation

x = np.linspace(0, 15, 401)

input_data = Input(shape=(len(x), 1))
r = 16
# /gpu:0
# resnet 1차
x = layers.Conv1D(
    32, 4, strides=2, padding="same", kernel_initializer="he_normal"
)(input_data)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    32, 4, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.Conv1D(
    32, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)
x = layers.MaxPooling1D(3, strides=2)(x)  # 나누기 2

# --------------------------------------

shortcut = x
shortcut = layers.Conv1D(
    64, 1, strides=1, padding="valid", kernel_initializer="he_normal"
)(shortcut)


x = layers.Conv1D(
    64, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    64, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)


x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)


shortcut = x  # identity shortcut
x = layers.Conv1D(
    64, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    64, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)


x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)


# --------------------------------------

shortcut = x
shortcut = layers.Conv1D(
    128, 1, strides=2, padding="valid", kernel_initializer="he_normal"
)(shortcut)

x = layers.Conv1D(
    128, 3, strides=2, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    128, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)

x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)


shortcut = x  # identity shortcut
x = layers.Conv1D(
    128, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # identity shortcut
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    128, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)


x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)

# --------------------------------------

shortcut = x
shortcut = layers.Conv1D(
    256, 1, strides=2, padding="valid", kernel_initializer="he_normal"
)(shortcut)

x = layers.Conv1D(
    256, 3, strides=2, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    256, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)

x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)


shortcut = x  # identity shortcut
x = layers.Conv1D(
    256, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # identity shortcut
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    256, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)


x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)

# --------------------------------------

shortcut = x
shortcut = layers.Conv1D(
    512, 1, strides=2, padding="valid", kernel_initializer="he_normal"
)(shortcut)

x = layers.Conv1D(
    512, 3, strides=2, padding="same", kernel_initializer="he_normal"
)(
    x
)  # 나누기 2
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    512, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)

x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)


shortcut = x  # identity shortcut
x = layers.Conv1D(
    512, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(
    x
)  # identity shortcut
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.01)(x)

x = layers.Conv1D(
    512, 3, strides=1, padding="same", kernel_initializer="he_normal"
)(x)
x = layers.BatchNormalization()(x)


x = layers.Add()([x, shortcut])
x = layers.LeakyReLU(alpha=0.01)(x)

# --------------------------------------

x = layers.GlobalAveragePooling1D()(x)

# and BN을 확인해보자


total_center1 = Dense(
    100, name="total_center1", kernel_initializer="he_normal"
)(x)
center_Batchnormalization = BatchNormalization()(total_center1)
total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
total_center3 = Dense(
    1,
    activation="linear",
    name="total_center3",
    kernel_initializer="he_normal",
)(total_center1_act)

total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(
    x
)
width_Batchnormalization = BatchNormalization()(total_width1)
total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
total_width3 = Dense(
    1, activation="linear", name="total_width3", kernel_initializer="he_normal"
)(total_width1_act)

total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
amp_Batchnormalization = BatchNormalization()(total_amp1)
total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
total_amp3 = Dense(
    1, activation="linear", name="total_amp3", kernel_initializer="he_normal"
)(total_amp1_act)

total_peak_number1 = Dense(
    100, name="total_peak_number1", kernel_initializer="he_normal"
)(x)
peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(
    peak_number_Batchnormalization
)
total_peak_number3 = Dense(
    1,
    activation="linear",
    name="total_peak_number3",
    kernel_initializer="he_normal",
)(total_peak_number1_act)

model_resnet = Model(
    inputs=input_data,
    outputs=[total_center3, total_width3, total_amp3, total_peak_number3],
)
print(model_resnet.summary())


# In[163]:


plot_model(model_resnet, show_shapes=True)


# In[164]:

# Resnet with mae

model_resnet.compile(
    optimizer="adam",
    loss={
        "total_center3": "mae",
        "total_width3": "mae",
        "total_amp3": "mae",
        "total_peak_number3": "mae",
    },
    loss_weights={
        "total_center3": 1,
        "total_width3": 10,
        "total_amp3": 20,
        "total_peak_number3": 2,
    },
    metrics=["mae"],
)


# In[165]:


# 콜백설정
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_model_resnet_mae.h5", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)
# 4 7


# In[166]:


models_resnet = model_resnet.fit(
    train_peak,
    [
        np.array(train_center),
        np.array(train_width),
        np.array(train_amp),
        np.array(train_peak_number),
    ],
    epochs=5,
    batch_size=512,
    validation_data=(
        val_peak,
        [
            np.array(val_center),
            np.array(val_width),
            np.array(val_amp),
            np.array(val_peak_number),
        ],
    ),
    callbacks=[model_checkpoint, reduce_lr],
    shuffle=True,
)


# In[167]:


for key in models_resnet.history.keys():
    print(key)


# In[168]:


plt.figure(figsize=(25, 15))

plt.subplot(231)
plt.plot(models_resnet.history["loss"], "b-", label="Resnet - training")
plt.plot(models_resnet.history["val_loss"], "r:", label="Resnet - validation")
plt.grid(True)
plt.title("Total Loss", size=32)
plt.legend()


plt.subplot(232)
plt.plot(
    models_resnet.history["total_center3_loss"],
    "b-",
    label="Resnet - training",
)
plt.plot(
    models_resnet.history["val_total_center3_loss"],
    "r:",
    label="Resnet - validation",
)
plt.grid(True)
plt.title("center Loss", size=32)
plt.legend()

plt.subplot(234)
plt.plot(
    models_resnet.history["total_width3_loss"], "b-", label="Resnet - training"
)
plt.plot(
    models_resnet.history["val_total_width3_loss"],
    "r:",
    label="Resnet - validation",
)
plt.grid(True)
plt.title("width Loss", size=32)
plt.legend()

plt.subplot(235)
plt.plot(
    models_resnet.history["total_amp3_loss"], "b-", label="Resnet - training"
)
plt.plot(
    models_resnet.history["val_total_amp3_loss"],
    "r:",
    label="Resnet - validation",
)
plt.grid(True)
plt.title("amp Loss", size=32)
plt.legend()


# In[169]:


plt.plot(
    models_resnet.history["total_peak_number3_loss"],
    "b-",
    label="SE-Resnet - training",
)
plt.plot(
    models_resnet.history["val_total_peak_number3_loss"],
    "r:",
    label="SE-Resnet - validation",
)
plt.grid(True)
plt.ylim(0, 0.05)
plt.title("peak number Loss", size=32)
plt.legend()


# In[126]:


# resnet

from tensorflow.keras.models import load_model

best_model_resnet = load_model(
    "best_model_resnet_mae.h5"
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_resnet.summary()


# In[127]:


# resnet

prediction_resnet = best_model_resnet.predict(test_peak)
print(len(prediction_resnet))
print(np.array(prediction_resnet).shape)


# In[172]:


# resnet

x = np.linspace(0, 15, 401)

for i in range(0, 15):  # 50 :100
    plt.figure(figsize=(10, 5))
    plt.plot(x, test_peak[i], c="black")
    plt.plot(
        x,
        y(
            prediction_resnet[0][i][0],
            prediction_resnet[1][i][0],
            prediction_resnet[2][i][0],
            x,
        ),
        label="predict",
        c="red",
    )

    plt.plot(
        x,
        y(test_center[i], test_width[i], test_amp[i], x),
        c="blue",
        label="real",
    )
    plt.legend()
    plt.title(i)


# In[128]:


# resnet


loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_resnet[0][i][0]))
    loss_width += abs((test_width[i] - prediction_resnet[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_resnet[2][i][0]))
    loss_peak_number += abs((test_peak_number[i] - prediction_resnet[3][i][0]))

resnet_loss_center_mae = loss_center / len(test_center)
resnet_loss_width_mae = loss_width / len(test_center)
resnet_loss_amp_mae = loss_amp / len(test_center)
resnet_loss_peak_number_mae = loss_peak_number / len(test_center)


# In[154]:


# print(resnet_loss_center_mse)
# print(resnet_loss_width_mse)
# print(resnet_loss_amp_mse)
# print(resnet_loss_peak_number_mse)

# mse
# 0.13585617930802474
# 0.026985588911410834
# 0.018131230965164946
# 0.053011787017337386

# mae
# 0.09873772096446064
# 0.01909559831345548
# 0.012985318065110189
# 0.03740239148552307

0.11626761483127424
0.0259811554032087
0.02165633446751664
0.049792706323664726


# %% Resnet with mse

model_resnet.compile(
    optimizer="adam",
    loss={
        "total_center3": "mse",
        "total_width3": "mse",
        "total_amp3": "mse",
        "total_peak_number3": "mse",
    },
    loss_weights={
        "total_center3": 1,
        "total_width3": 10,
        "total_amp3": 20,
        "total_peak_number3": 2,
    },
    metrics=["mse"],
)


# In[165]:


# 콜백설정
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_model_resnet_mse.h5", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)
# 4 7


# In[166]:


models_resnet = model_resnet.fit(
    train_peak,
    [
        np.array(train_center),
        np.array(train_width),
        np.array(train_amp),
        np.array(train_peak_number),
    ],
    epochs=5,
    batch_size=512,
    validation_data=(
        val_peak,
        [
            np.array(val_center),
            np.array(val_width),
            np.array(val_amp),
            np.array(val_peak_number),
        ],
    ),
    callbacks=[model_checkpoint, reduce_lr],
    shuffle=True,
)


# In[167]:


for key in models_resnet.history.keys():
    print(key)


# In[168]:


plt.figure(figsize=(25, 15))

plt.subplot(231)
plt.plot(models_resnet.history["loss"], "b-", label="Resnet - training")
plt.plot(models_resnet.history["val_loss"], "r:", label="Resnet - validation")
plt.grid(True)
plt.title("Total Loss", size=32)
plt.legend()


plt.subplot(232)
plt.plot(
    models_resnet.history["total_center3_loss"],
    "b-",
    label="Resnet - training",
)
plt.plot(
    models_resnet.history["val_total_center3_loss"],
    "r:",
    label="Resnet - validation",
)
plt.grid(True)
plt.title("center Loss", size=32)
plt.legend()

plt.subplot(234)
plt.plot(
    models_resnet.history["total_width3_loss"], "b-", label="Resnet - training"
)
plt.plot(
    models_resnet.history["val_total_width3_loss"],
    "r:",
    label="Resnet - validation",
)
plt.grid(True)
plt.title("width Loss", size=32)
plt.legend()

plt.subplot(235)
plt.plot(
    models_resnet.history["total_amp3_loss"], "b-", label="Resnet - training"
)
plt.plot(
    models_resnet.history["val_total_amp3_loss"],
    "r:",
    label="Resnet - validation",
)
plt.grid(True)
plt.title("amp Loss", size=32)
plt.legend()


# In[169]:


plt.plot(
    models_resnet.history["total_peak_number3_loss"],
    "b-",
    label="SE-Resnet - training",
)
plt.plot(
    models_resnet.history["val_total_peak_number3_loss"],
    "r:",
    label="SE-Resnet - validation",
)
plt.grid(True)
plt.ylim(0, 0.05)
plt.title("peak number Loss", size=32)
plt.legend()


# In[126]:


# resnet

from tensorflow.keras.models import load_model

best_model_resnet = load_model(
    "best_model_resnet_mse.h5"
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_resnet.summary()


# In[127]:


# resnet

prediction_resnet = best_model_resnet.predict(test_peak)
print(len(prediction_resnet))
print(np.array(prediction_resnet).shape)


# In[172]:


# resnet

x = np.linspace(0, 15, 401)

for i in range(0, 15):  # 50 :100
    plt.figure(figsize=(10, 5))
    plt.plot(x, test_peak[i], c="black")
    plt.plot(
        x,
        y(
            prediction_resnet[0][i][0],
            prediction_resnet[1][i][0],
            prediction_resnet[2][i][0],
            x,
        ),
        label="predict",
        c="red",
    )

    plt.plot(
        x,
        y(test_center[i], test_width[i], test_amp[i], x),
        c="blue",
        label="real",
    )
    plt.legend()
    plt.title(i)


# In[128]:


# resnet


loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += np.square((test_center[i] - prediction_resnet[0][i][0]))
    loss_width += np.square((test_width[i] - prediction_resnet[1][i][0]))
    loss_amp += np.square((test_amp[i] - prediction_resnet[2][i][0]))
    loss_peak_number += np.square(
        (test_peak_number[i] - prediction_resnet[3][i][0])
    )

resnet_loss_center_mse = np.sqrt(loss_center / len(test_center))
resnet_loss_width_mse = np.sqrt(loss_width / len(test_center))
resnet_loss_amp_mse = np.sqrt(loss_amp / len(test_center))
resnet_loss_peak_number_mse = np.sqrt(loss_peak_number / len(test_center))


# In[154]:


print(resnet_loss_center_mse)
print(resnet_loss_width_mse)
print(resnet_loss_amp_mse)
print(resnet_loss_peak_number_mse)
# %%


# ### Vggnet
# - concent : 인수분해된 filter size로 반복 극대화
# - how :
# - -  i) filter fize=4,3의 convolution을 3,2개를 한꺼번에 쌓은후 subsampling
# - - ii) 32x3-64x2-128x2-256x2-512x2

# In[64]:


########################### vggnet ########################


# In[175]:


# vggnet

x = np.linspace(0, 15, 401)

input_data = Input(shape=(len(x), 1))

x = layers.Conv1D(32, 4, strides=2, activation="relu", padding="same")(
    input_data
)
x = layers.Conv1D(32, 4, strides=1, activation="relu", padding="same")(x)
x = layers.Conv1D(32, 3, strides=1, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2, strides=2)(x)

x = layers.Conv1D(64, 3, strides=1, activation="relu", padding="same")(x)
x = layers.Conv1D(64, 3, strides=1, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2, strides=2)(x)

x = layers.Conv1D(128, 3, strides=1, activation="relu", padding="same")(x)
x = layers.Conv1D(128, 3, strides=1, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2, strides=2)(x)

x = layers.Conv1D(256, 3, strides=1, activation="relu", padding="same")(x)
x = layers.Conv1D(256, 3, strides=1, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2, strides=2)(x)

x = layers.Conv1D(512, 3, strides=1, activation="relu", padding="same")(x)
x = layers.Conv1D(512, 3, strides=1, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2, strides=2)(x)

x = layers.GlobalMaxPooling1D()(x)


total_center1 = Dense(
    100, name="total_center1", kernel_initializer="he_normal"
)(x)
center_Batchnormalization = BatchNormalization()(total_center1)
total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
total_center3 = Dense(
    1,
    activation="linear",
    name="total_center3",
    kernel_initializer="he_normal",
)(total_center1_act)

total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(
    x
)
width_Batchnormalization = BatchNormalization()(total_width1)
total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
total_width3 = Dense(
    1, activation="linear", name="total_width3", kernel_initializer="he_normal"
)(total_width1_act)

total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
amp_Batchnormalization = BatchNormalization()(total_amp1)
total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
total_amp3 = Dense(
    1, activation="linear", name="total_amp3", kernel_initializer="he_normal"
)(total_amp1_act)

total_peak_number1 = Dense(
    100, name="total_peak_number1", kernel_initializer="he_normal"
)(x)
peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(
    peak_number_Batchnormalization
)
total_peak_number3 = Dense(
    1,
    activation="linear",
    name="total_peak_number3",
    kernel_initializer="he_normal",
)(total_peak_number1_act)

model_vggnet = Model(
    inputs=input_data,
    outputs=[total_center3, total_width3, total_amp3, total_peak_number3],
)
print(model_vggnet.summary())


# In[176]:


# vggnet
plot_model(model_vggnet, show_shapes=True)


# In[177]:


# vggnet
# best_model3.h5

model_vggnet.compile(
    optimizer="adam",
    loss={
        "total_center3": "mae",
        "total_width3": "mae",
        "total_amp3": "mae",
        "total_peak_number3": "mae",
    },
    loss_weights={
        "total_center3": 1,
        "total_width3": 10,
        "total_amp3": 20,
        "total_peak_number3": 2,
    },
    metrics=["mae"],
)


# In[178]:


# vggnet

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_model_vggnet_mae.h5", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)
# 4 7


# In[179]:


# vggnet

models_vggnet = model_vggnet.fit(
    train_peak,
    [
        np.array(train_center),
        np.array(train_width),
        np.array(train_amp),
        np.array(train_peak_number),
    ],
    epochs=5,
    batch_size=512,
    validation_data=(
        val_peak,
        [
            np.array(val_center),
            np.array(val_width),
            np.array(val_amp),
            np.array(val_peak_number),
        ],
    ),
    callbacks=[model_checkpoint, reduce_lr],
    shuffle=True,
)


# In[180]:


# vggnet

for key in models_vggnet.history.keys():
    print(key)


# In[181]:


# vggnet
plt.figure(figsize=(25, 15))

plt.subplot(231)
plt.plot(models_vggnet.history["loss"], "b-", label="VGGnet - training")
plt.plot(models_vggnet.history["val_loss"], "r:", label="VGGnet - validation")
plt.grid(True)
plt.title("Total Loss", size=32)
plt.legend()


plt.subplot(232)
plt.plot(
    models_vggnet.history["total_center3_loss"],
    "b-",
    label="VGGnet - training",
)
plt.plot(
    models_vggnet.history["val_total_center3_loss"],
    "r:",
    label="VGGnet - validation",
)
plt.grid(True)
plt.title("center Loss", size=32)
plt.legend()

plt.subplot(234)
plt.plot(
    models_vggnet.history["total_width3_loss"], "b-", label="VGGnet - training"
)
plt.plot(
    models_vggnet.history["val_total_width3_loss"],
    "r:",
    label="VGGnet - validation",
)
plt.grid(True)
plt.title("width Loss", size=32)
plt.legend()

plt.subplot(235)
plt.plot(
    models_vggnet.history["total_amp3_loss"], "b-", label="VGGnet - training"
)
plt.plot(
    models_vggnet.history["val_total_amp3_loss"],
    "r:",
    label="VGGnet - validation",
)
plt.grid(True)
plt.title("amp Loss", size=32)
plt.legend()


# In[182]:


plt.plot(
    models_vggnet.history["total_peak_number3_loss"],
    "b-",
    label="SE-Resnet - training",
)
plt.plot(
    models_vggnet.history["val_total_peak_number3_loss"],
    "r:",
    label="SE-Resnet - validation",
)
plt.grid(True)
plt.ylim(0, 0.03)
plt.title("peak number Loss", size=32)
plt.legend()


# In[130]:


# vggnet

from tensorflow.keras.models import load_model

best_model_vggnet = load_model(
    "best_model_vggnet_mae.h5"
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_vggnet.summary()


# In[131]:


# vggnet

prediction_vggnet = best_model_vggnet.predict(test_peak)
print(len(prediction_vggnet))
print(np.array(prediction_vggnet).shape)


# In[185]:


# vggnet

x = np.linspace(0, 15, 401)

for i in range(0, 15):
    plt.figure(figsize=(10, 5))
    plt.plot(x, test_peak[i], c="black")
    #     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(
        x,
        y(
            prediction_vggnet[0][i][0],
            prediction_vggnet[1][i][0],
            prediction_vggnet[2][i][0],
            x,
        ),
        label="predict",
        c="red",
    )

    plt.plot(
        x,
        y(test_center[i], test_width[i], test_amp[i], x),
        c="blue",
        label="real",
    )
    #     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[132]:


# vggnet

loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_vggnet[0][i][0]))
    loss_width += abs((test_width[i] - prediction_vggnet[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_vggnet[2][i][0]))
    loss_peak_number += abs((test_peak_number[i] - prediction_vggnet[3][i][0]))

vggnet_loss_center_mae = loss_center / len(test_center)
vggnet_loss_width_mae = loss_width / len(test_center)
vggnet_loss_amp_mae = loss_amp / len(test_center)
vggnet_loss_peak_number_mae = loss_peak_number / len(test_center)


# In[133]:


print(vggnet_loss_center_mae)
print(vggnet_loss_width_mae)
print(vggnet_loss_amp_mae)
print(vggnet_loss_peak_number_mae)

# mse
# 0.16759907352027545
# 0.02866257626182038
# 0.019091169889994618
# 0.067914283398507

# mae
# 0.1410774833079569
# 0.022938466563435428
# 0.01608092693611724
# 0.04853178667237555

# %%
# vggnet with mse

model_vggnet.compile(
    optimizer="adam",
    loss={
        "total_center3": "mse",
        "total_width3": "mse",
        "total_amp3": "mse",
        "total_peak_number3": "mse",
    },
    loss_weights={
        "total_center3": 1,
        "total_width3": 10,
        "total_amp3": 20,
        "total_peak_number3": 2,
    },
    metrics=["mse"],
)


# In[178]:


# vggnet

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_model_vggnet_mse.h5", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)
# 4 7


# In[179]:


# vggnet

models_vggnet = model_vggnet.fit(
    train_peak,
    [
        np.array(train_center),
        np.array(train_width),
        np.array(train_amp),
        np.array(train_peak_number),
    ],
    epochs=5,
    batch_size=512,
    validation_data=(
        val_peak,
        [
            np.array(val_center),
            np.array(val_width),
            np.array(val_amp),
            np.array(val_peak_number),
        ],
    ),
    callbacks=[model_checkpoint, reduce_lr],
    shuffle=True,
)
# %% Alexnet #############################################################
# -concent :
# - - i) input data에 맞는 다양한 filter size 사용
# - - ii) conv-pooling의 단순한 subsampling 반복 탈피
# - - iii) overlapped pooling
# - How
# - - i) 중간 conv1d 2개의 layer는subsampling 안함
# - - ii) 96x1 -256x1 - 384x3 의 channel

# %%  alexnet+zfnet
x = np.linspace(0, 15, 401)

input_data = Input(shape=(len(x), 1))


x = layers.Conv1D(96, 20, strides=2, activation="relu", padding="same")(
    input_data
)
x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
x = layers.Conv1D(256, 9, strides=2, activation="relu", padding="same")(x)
x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
x = layers.Conv1D(384, 4, activation="relu", padding="same")(x)
x = layers.Conv1D(384, 4, activation="relu", padding="same")(x)
x = layers.Conv1D(256, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
x = layers.GlobalMaxPooling1D()(x)


total_center1 = Dense(
    100, name="total_center1", kernel_initializer="he_normal"
)(x)
center_Batchnormalization = BatchNormalization()(total_center1)
total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
total_center3 = Dense(
    1,
    activation="linear",
    name="total_center3",
    kernel_initializer="he_normal",
)(total_center1_act)

total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(
    x
)
width_Batchnormalization = BatchNormalization()(total_width1)
total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
total_width3 = Dense(
    1, activation="linear", name="total_width3", kernel_initializer="he_normal"
)(total_width1_act)

total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
amp_Batchnormalization = BatchNormalization()(total_amp1)
total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
total_amp3 = Dense(
    1, activation="linear", name="total_amp3", kernel_initializer="he_normal"
)(total_amp1_act)

total_peak_number1 = Dense(
    200, name="total_peak_number1", kernel_initializer="he_normal"
)(x)
peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(
    peak_number_Batchnormalization
)
total_peak_number3 = Dense(
    1,
    activation="linear",
    name="total_peak_number3",
    kernel_initializer="he_normal",
)(total_peak_number1_act)


model_alex_zfnet = Model(
    inputs=input_data,
    outputs=[total_center3, total_width3, total_amp3, total_peak_number3],
)
print(model_alex_zfnet.summary())

# %%
# alexnet+zfnet
plot_model(model_alex_zfnet, show_shapes=True)

# %%
model_alex_zfnet.compile(
    optimizer="adam",
    loss={
        "total_center3": "mae",
        "total_width3": "mae",
        "total_amp3": "mae",
        "total_peak_number3": "mae",
    },
    loss_weights={
        "total_center3": 1,
        "total_width3": 10,
        "total_amp3": 20,
        "total_peak_number3": 2,
    },
    metrics=["mae"],
)


# %%


# alexnet+zfnet

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_model_alex_zfnet_mae.h5", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)
# 4 7


# In[192]:


# alexnet+zfnet

models_alex_zfnet = model_alex_zfnet.fit(
    train_peak,
    [
        np.array(train_center),
        np.array(train_width),
        np.array(train_amp),
        np.array(train_peak_number),
    ],
    epochs=5,
    batch_size=512,
    validation_data=(
        val_peak,
        [
            np.array(val_center),
            np.array(val_width),
            np.array(val_amp),
            np.array(val_peak_number),
        ],
    ),
    callbacks=[model_checkpoint, reduce_lr],
    shuffle=True,
)


# In[193]:


# alexnet+zfnet

for key in models_alex_zfnet.history.keys():
    print(key)


# In[194]:


# alexnet+zfnet
plt.figure(figsize=(25, 15))

plt.subplot(231)
plt.plot(
    models_alex_zfnet.history["loss"], "b-", label="Alex_ZFnet - training"
)
plt.plot(
    models_alex_zfnet.history["val_loss"],
    "r:",
    label="Alex_ZFnet - validation",
)
plt.grid(True)
plt.title("Total Loss", size=32)
plt.legend()


plt.subplot(232)
plt.plot(
    models_alex_zfnet.history["total_center3_loss"],
    "b-",
    label="Alex_ZFnet - training",
)
plt.plot(
    models_alex_zfnet.history["val_total_center3_loss"],
    "r:",
    label="Alex_ZFnet - validation",
)
plt.grid(True)
plt.title("center Loss", size=32)
plt.legend()

plt.subplot(234)
plt.plot(
    models_alex_zfnet.history["total_width3_loss"],
    "b-",
    label="Alex_ZFnet - training",
)
plt.plot(
    models_alex_zfnet.history["val_total_width3_loss"],
    "r:",
    label="Alex_ZFnet - validation",
)
plt.grid(True)
plt.title("width Loss", size=32)
plt.legend()

plt.subplot(235)
plt.plot(
    models_alex_zfnet.history["total_amp3_loss"],
    "b-",
    label="Alex_ZFnet - training",
)
plt.plot(
    models_alex_zfnet.history["val_total_amp3_loss"],
    "r:",
    label="Alex_ZFnet - validation",
)
plt.grid(True)
plt.title("amp Loss", size=32)
plt.legend()


# In[195]:


plt.plot(
    models_alex_zfnet.history["total_peak_number3_loss"],
    "b-",
    label="SE-Resnet - training",
)
plt.plot(
    models_alex_zfnet.history["val_total_peak_number3_loss"],
    "r:",
    label="SE-Resnet - validation",
)
plt.grid(True)
plt.ylim(0, 0.03)
plt.title("peak number Loss", size=32)
plt.legend()


# In[134]:


# alexnet+zfnet

from tensorflow.keras.models import load_model

best_model_alex_zfnet = load_model(
    "best_model_alex_zfnet_mae.h5"
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_alex_zfnet.summary()


# In[135]:


# alexnet+zfnet

prediction_alex_zfnet = best_model_alex_zfnet.predict(test_peak)
print(len(prediction_alex_zfnet))
print(np.array(prediction_alex_zfnet).shape)


# In[198]:


# alexnet+zfnet

x = np.linspace(0, 15, 401)

for i in range(0, 15):
    plt.figure(figsize=(10, 5))
    plt.plot(x, test_peak[i], c="black")
    #     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(
        x,
        y(
            prediction_alex_zfnet[0][i][0],
            prediction_alex_zfnet[1][i][0],
            prediction_alex_zfnet[2][i][0],
            x,
        ),
        label="predict",
        c="red",
    )

    plt.plot(
        x,
        y(test_center[i], test_width[i], test_amp[i], x),
        c="blue",
        label="real",
    )
    #     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[136]:


# alexnet+zfnet

loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_alex_zfnet[0][i][0]))
    loss_width += abs((test_width[i] - prediction_alex_zfnet[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_alex_zfnet[2][i][0]))
    loss_peak_number += abs(
        (test_peak_number[i] - prediction_alex_zfnet[3][i][0])
    )

alex_zfnet_loss_center_mae = loss_center / len(test_center)
alex_zfnet_loss_width_mae = loss_width / len(test_center)
alex_zfnet_loss_amp_mae = loss_amp / len(test_center)
alex_zfnet_loss_peak_number_mae = loss_peak_number / len(test_center)


# In[137]:


# alexnet+zfnet

print(alex_zfnet_loss_center_mae)
print(alex_zfnet_loss_width_mae)
print(alex_zfnet_loss_amp_mae)
print(alex_zfnet_loss_peak_number_mae)


# In[180]:


# vggnet

for key in models_vggnet.history.keys():
    print(key)


# In[181]:


# vggnet
plt.figure(figsize=(25, 15))

plt.subplot(231)
plt.plot(models_vggnet.history["loss"], "b-", label="VGGnet - training")
plt.plot(models_vggnet.history["val_loss"], "r:", label="VGGnet - validation")
plt.grid(True)
plt.title("Total Loss", size=32)
plt.legend()


plt.subplot(232)
plt.plot(
    models_vggnet.history["total_center3_loss"],
    "b-",
    label="VGGnet - training",
)
plt.plot(
    models_vggnet.history["val_total_center3_loss"],
    "r:",
    label="VGGnet - validation",
)
plt.grid(True)
plt.title("center Loss", size=32)
plt.legend()

plt.subplot(234)
plt.plot(
    models_vggnet.history["total_width3_loss"], "b-", label="VGGnet - training"
)
plt.plot(
    models_vggnet.history["val_total_width3_loss"],
    "r:",
    label="VGGnet - validation",
)
plt.grid(True)
plt.title("width Loss", size=32)
plt.legend()

plt.subplot(235)
plt.plot(
    models_vggnet.history["total_amp3_loss"], "b-", label="VGGnet - training"
)
plt.plot(
    models_vggnet.history["val_total_amp3_loss"],
    "r:",
    label="VGGnet - validation",
)
plt.grid(True)
plt.title("amp Loss", size=32)
plt.legend()


# In[182]:


plt.plot(
    models_vggnet.history["total_peak_number3_loss"],
    "b-",
    label="SE-Resnet - training",
)
plt.plot(
    models_vggnet.history["val_total_peak_number3_loss"],
    "r:",
    label="SE-Resnet - validation",
)
plt.grid(True)
plt.ylim(0, 0.03)
plt.title("peak number Loss", size=32)
plt.legend()


# In[130]:


# vggnet

from tensorflow.keras.models import load_model

best_model_vggnet = load_model(
    "best_model_vggnet_mse.h5"
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_vggnet.summary()


# In[131]:


# vggnet

prediction_vggnet = best_model_vggnet.predict(test_peak)
print(len(prediction_vggnet))
print(np.array(prediction_vggnet).shape)


# In[185]:


# vggnet

x = np.linspace(0, 15, 401)

for i in range(0, 15):
    plt.figure(figsize=(10, 5))
    plt.plot(x, test_peak[i], c="black")
    #     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(
        x,
        y(
            prediction_vggnet[0][i][0],
            prediction_vggnet[1][i][0],
            prediction_vggnet[2][i][0],
            x,
        ),
        label="predict",
        c="red",
    )

    plt.plot(
        x,
        y(test_center[i], test_width[i], test_amp[i], x),
        c="blue",
        label="real",
    )
    #     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[132]:


# vggnet

loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += np.square((test_center[i] - prediction_vggnet[0][i][0]))
    loss_width += np.square((test_width[i] - prediction_vggnet[1][i][0]))
    loss_amp += np.square((test_amp[i] - prediction_vggnet[2][i][0]))
    loss_peak_number += np.square(
        (test_peak_number[i] - prediction_vggnet[3][i][0])
    )

vggnet_loss_center_mse = np.sqrt(loss_center / len(test_center))
vggnet_loss_width_mse = np.sqrt(loss_width / len(test_center))
vggnet_loss_amp_mse = np.sqrt(loss_amp / len(test_center))
vggnet_loss_peak_number_mse = np.sqrt(loss_peak_number / len(test_center))


# In[133]:


print(vggnet_loss_center_mse)
print(vggnet_loss_width_mse)
print(vggnet_loss_amp_mse)
print(vggnet_loss_peak_number_mse)

# %%


# ### Alexnet
# -concent :
# - - i) input data에 맞는 다양한 filter size 사용
# - - ii) conv-pooling의 단순한 subsampling 반복 탈피
# - - iii) overlapped pooling
# - How
# - - i) 중간 conv1d 2개의 layer는subsampling 안함
# - - ii) 96x1 -256x1 - 384x3 의 channel
#
#

# In[78]:


####################### Alex+ZFnet ###################


# In[188]:


# alexnet+zfnet
x = np.linspace(0, 15, 401)

input_data = Input(shape=(len(x), 1))


x = layers.Conv1D(96, 20, strides=2, activation="relu", padding="same")(
    input_data
)
x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
x = layers.Conv1D(256, 9, strides=2, activation="relu", padding="same")(x)
x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
x = layers.Conv1D(384, 4, activation="relu", padding="same")(x)
x = layers.Conv1D(384, 4, activation="relu", padding="same")(x)
x = layers.Conv1D(256, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
x = layers.GlobalMaxPooling1D()(x)


total_center1 = Dense(
    100, name="total_center1", kernel_initializer="he_normal"
)(x)
center_Batchnormalization = BatchNormalization()(total_center1)
total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
total_center3 = Dense(
    1,
    activation="linear",
    name="total_center3",
    kernel_initializer="he_normal",
)(total_center1_act)

total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(
    x
)
width_Batchnormalization = BatchNormalization()(total_width1)
total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
total_width3 = Dense(
    1, activation="linear", name="total_width3", kernel_initializer="he_normal"
)(total_width1_act)

total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
amp_Batchnormalization = BatchNormalization()(total_amp1)
total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
total_amp3 = Dense(
    1, activation="linear", name="total_amp3", kernel_initializer="he_normal"
)(total_amp1_act)

total_peak_number1 = Dense(
    200, name="total_peak_number1", kernel_initializer="he_normal"
)(x)
peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(
    peak_number_Batchnormalization
)
total_peak_number3 = Dense(
    1,
    activation="linear",
    name="total_peak_number3",
    kernel_initializer="he_normal",
)(total_peak_number1_act)


model_alex_zfnet = Model(
    inputs=input_data,
    outputs=[total_center3, total_width3, total_amp3, total_peak_number3],
)
print(model_alex_zfnet.summary())


# In[189]:


# alexnet+zfnet
plot_model(model_alex_zfnet, show_shapes=True)


# In[190]:


# alexnet+zfnet with mae

model_alex_zfnet.compile(
    optimizer="adam",
    loss={
        "total_center3": "mae",
        "total_width3": "mae",
        "total_amp3": "mae",
        "total_peak_number3": "mae",
    },
    loss_weights={
        "total_center3": 1,
        "total_width3": 10,
        "total_amp3": 20,
        "total_peak_number3": 2,
    },
    metrics=["mae"],
)


# In[191]:


# alexnet+zfnet

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_model_alex_zfnet_mae.h5", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)
# 4 7


# In[192]:


# alexnet+zfnet

models_alex_zfnet = model_alex_zfnet.fit(
    train_peak,
    [
        np.array(train_center),
        np.array(train_width),
        np.array(train_amp),
        np.array(train_peak_number),
    ],
    epochs=5,
    batch_size=512,
    validation_data=(
        val_peak,
        [
            np.array(val_center),
            np.array(val_width),
            np.array(val_amp),
            np.array(val_peak_number),
        ],
    ),
    callbacks=[model_checkpoint, reduce_lr],
    shuffle=True,
)


# In[193]:


# alexnet+zfnet

for key in models_alex_zfnet.history.keys():
    print(key)


# In[194]:


# alexnet+zfnet
plt.figure(figsize=(25, 15))

plt.subplot(231)
plt.plot(
    models_alex_zfnet.history["loss"], "b-", label="Alex_ZFnet - training"
)
plt.plot(
    models_alex_zfnet.history["val_loss"],
    "r:",
    label="Alex_ZFnet - validation",
)
plt.grid(True)
plt.title("Total Loss", size=32)
plt.legend()


plt.subplot(232)
plt.plot(
    models_alex_zfnet.history["total_center3_loss"],
    "b-",
    label="Alex_ZFnet - training",
)
plt.plot(
    models_alex_zfnet.history["val_total_center3_loss"],
    "r:",
    label="Alex_ZFnet - validation",
)
plt.grid(True)
plt.title("center Loss", size=32)
plt.legend()

plt.subplot(234)
plt.plot(
    models_alex_zfnet.history["total_width3_loss"],
    "b-",
    label="Alex_ZFnet - training",
)
plt.plot(
    models_alex_zfnet.history["val_total_width3_loss"],
    "r:",
    label="Alex_ZFnet - validation",
)
plt.grid(True)
plt.title("width Loss", size=32)
plt.legend()

plt.subplot(235)
plt.plot(
    models_alex_zfnet.history["total_amp3_loss"],
    "b-",
    label="Alex_ZFnet - training",
)
plt.plot(
    models_alex_zfnet.history["val_total_amp3_loss"],
    "r:",
    label="Alex_ZFnet - validation",
)
plt.grid(True)
plt.title("amp Loss", size=32)
plt.legend()


# In[195]:


plt.plot(
    models_alex_zfnet.history["total_peak_number3_loss"],
    "b-",
    label="SE-Resnet - training",
)
plt.plot(
    models_alex_zfnet.history["val_total_peak_number3_loss"],
    "r:",
    label="SE-Resnet - validation",
)
plt.grid(True)
plt.ylim(0, 0.03)
plt.title("peak number Loss", size=32)
plt.legend()


# In[134]:


# alexnet+zfnet

from tensorflow.keras.models import load_model

best_model_alex_zfnet = load_model(
    "best_model_alex_zfnet_mae.h5"
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_alex_zfnet.summary()


# In[135]:


# alexnet+zfnet

prediction_alex_zfnet = best_model_alex_zfnet.predict(test_peak)
print(len(prediction_alex_zfnet))
print(np.array(prediction_alex_zfnet).shape)


# In[198]:


# alexnet+zfnet

x = np.linspace(0, 15, 401)

for i in range(0, 15):
    plt.figure(figsize=(10, 5))
    plt.plot(x, test_peak[i], c="black")
    #     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(
        x,
        y(
            prediction_alex_zfnet[0][i][0],
            prediction_alex_zfnet[1][i][0],
            prediction_alex_zfnet[2][i][0],
            x,
        ),
        label="predict",
        c="red",
    )

    plt.plot(
        x,
        y(test_center[i], test_width[i], test_amp[i], x),
        c="blue",
        label="real",
    )
    #     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[136]:


# alexnet+zfnet

loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_alex_zfnet[0][i][0]))
    loss_width += abs((test_width[i] - prediction_alex_zfnet[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_alex_zfnet[2][i][0]))
    loss_peak_number += abs(
        (test_peak_number[i] - prediction_alex_zfnet[3][i][0])
    )

alex_zfnet_loss_center_mae = loss_center / len(test_center)
alex_zfnet_loss_width_mae = loss_width / len(test_center)
alex_zfnet_loss_amp_mae = loss_amp / len(test_center)
alex_zfnet_loss_peak_number_mae = loss_peak_number / len(test_center)


# In[137]:


# alexnet+zfnet

print(alex_zfnet_loss_center_mae)
print(alex_zfnet_loss_width_mae)
print(alex_zfnet_loss_amp_mae)
print(alex_zfnet_loss_peak_number_mae)

# mse
# 0.19464697865481032
# 0.029954838888122934
# 0.020524412014218365
# 0.06146805145220747

# mae
# 0.16305827815352805
# 0.022958871326448534
# 0.015632465159365342
# 0.04992606308040581


# %%

# alex&zfnet with mse

model_alex_zfnet.compile(
    optimizer="adam",
    loss={
        "total_center3": "mse",
        "total_width3": "mse",
        "total_amp3": "mse",
        "total_peak_number3": "mse",
    },
    loss_weights={
        "total_center3": 1,
        "total_width3": 10,
        "total_amp3": 20,
        "total_peak_number3": 2,
    },
    metrics=["mse"],
)


# In[191]:


# alexnet+zfnet

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_model_alex_zfnet_mse.h5", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)
# 4 7


# In[192]:


# alexnet+zfnet

models_alex_zfnet = model_alex_zfnet.fit(
    train_peak,
    [
        np.array(train_center),
        np.array(train_width),
        np.array(train_amp),
        np.array(train_peak_number),
    ],
    epochs=5,
    batch_size=512,
    validation_data=(
        val_peak,
        [
            np.array(val_center),
            np.array(val_width),
            np.array(val_amp),
            np.array(val_peak_number),
        ],
    ),
    callbacks=[model_checkpoint, reduce_lr],
    shuffle=True,
)


# In[193]:


# alexnet+zfnet

for key in models_alex_zfnet.history.keys():
    print(key)


# In[194]:


# alexnet+zfnet
plt.figure(figsize=(25, 15))

plt.subplot(231)
plt.plot(
    models_alex_zfnet.history["loss"], "b-", label="Alex_ZFnet - training"
)
plt.plot(
    models_alex_zfnet.history["val_loss"],
    "r:",
    label="Alex_ZFnet - validation",
)
plt.grid(True)
plt.title("Total Loss", size=32)
plt.legend()


plt.subplot(232)
plt.plot(
    models_alex_zfnet.history["total_center3_loss"],
    "b-",
    label="Alex_ZFnet - training",
)
plt.plot(
    models_alex_zfnet.history["val_total_center3_loss"],
    "r:",
    label="Alex_ZFnet - validation",
)
plt.grid(True)
plt.title("center Loss", size=32)
plt.legend()

plt.subplot(234)
plt.plot(
    models_alex_zfnet.history["total_width3_loss"],
    "b-",
    label="Alex_ZFnet - training",
)
plt.plot(
    models_alex_zfnet.history["val_total_width3_loss"],
    "r:",
    label="Alex_ZFnet - validation",
)
plt.grid(True)
plt.title("width Loss", size=32)
plt.legend()

plt.subplot(235)
plt.plot(
    models_alex_zfnet.history["total_amp3_loss"],
    "b-",
    label="Alex_ZFnet - training",
)
plt.plot(
    models_alex_zfnet.history["val_total_amp3_loss"],
    "r:",
    label="Alex_ZFnet - validation",
)
plt.grid(True)
plt.title("amp Loss", size=32)
plt.legend()


# In[195]:


plt.plot(
    models_alex_zfnet.history["total_peak_number3_loss"],
    "b-",
    label="SE-Resnet - training",
)
plt.plot(
    models_alex_zfnet.history["val_total_peak_number3_loss"],
    "r:",
    label="SE-Resnet - validation",
)
plt.grid(True)
plt.ylim(0, 0.03)
plt.title("peak number Loss", size=32)
plt.legend()


# In[134]:


# alexnet+zfnet

from tensorflow.keras.models import load_model

best_model_alex_zfnet = load_model(
    "best_model_alex_zfnet_mse.h5"
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지
best_model_alex_zfnet.summary()


# In[135]:


# alexnet+zfnet

prediction_alex_zfnet = best_model_alex_zfnet.predict(test_peak)
print(len(prediction_alex_zfnet))
print(np.array(prediction_alex_zfnet).shape)


# In[198]:


# alexnet+zfnet

x = np.linspace(0, 15, 401)

for i in range(0, 15):
    plt.figure(figsize=(10, 5))
    plt.plot(x, test_peak[i], c="black")
    #     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(
        x,
        y(
            prediction_alex_zfnet[0][i][0],
            prediction_alex_zfnet[1][i][0],
            prediction_alex_zfnet[2][i][0],
            x,
        ),
        label="predict",
        c="red",
    )

    plt.plot(
        x,
        y(test_center[i], test_width[i], test_amp[i], x),
        c="blue",
        label="real",
    )
    #     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[136]:


# alexnet+zfnet

loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += np.square((test_center[i] - prediction_alex_zfnet[0][i][0]))
    loss_width += np.square((test_width[i] - prediction_alex_zfnet[1][i][0]))
    loss_amp += np.square((test_amp[i] - prediction_alex_zfnet[2][i][0]))
    loss_peak_number += np.square(
        (test_peak_number[i] - prediction_alex_zfnet[3][i][0])
    )

alex_zfnet_loss_center_mse = np.sqrt(loss_center / len(test_center))
alex_zfnet_loss_width_mse = np.sqrt(loss_width / len(test_center))
alex_zfnet_loss_amp_mse = np.sqrt(loss_amp / len(test_center))
alex_zfnet_loss_peak_number_mse = np.sqrt(loss_peak_number / len(test_center))


# In[137]:


# alexnet+zfnet

print(alex_zfnet_loss_center_mse)
print(alex_zfnet_loss_width_mse)
print(alex_zfnet_loss_amp_mse)
print(alex_zfnet_loss_peak_number_mse)


# %%


# ### Lenet
# - concept:
# - - i) convolution layer의 첫 사용
# - - ii) 단순한 conv-pooling의 반복단계
# - - iii) 이전 peak fitting 논문의 cnn model
# - How
# - - i) conv-subsampling을 한개의 block으로 총  4 개의 block 쌓음
# - - ii) channel 32x1-64x1-128x1-256x1

# In[92]:


########################### lenet#################################
###########################before peak_fitting2 project model#######


# In[201]:


x = np.linspace(0, 15, 401)

input_data = Input(shape=(len(x), 1))

x = layers.Conv1D(32, 100, strides=3, activation="relu")(input_data)
x = layers.MaxPooling1D(2)(x)

x = layers.Conv1D(64, 10, strides=2, activation="relu")(x)
x = layers.MaxPooling1D(2)(x)

x = layers.Conv1D(128, 4, activation="relu")(x)
x = layers.MaxPooling1D(2)(x)

x = layers.Conv1D(256, 2, activation="relu")(x)
x = layers.MaxPooling1D(2)(x)

x = layers.GlobalMaxPooling1D()(x)


total_center1 = Dense(
    100, name="total_center1", kernel_initializer="he_normal"
)(x)
center_Batchnormalization = BatchNormalization()(total_center1)
total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
total_center3 = Dense(
    1,
    activation="linear",
    name="total_center3",
    kernel_initializer="he_normal",
)(total_center1_act)

total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(
    x
)
width_Batchnormalization = BatchNormalization()(total_width1)
total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
total_width3 = Dense(
    1, activation="linear", name="total_width3", kernel_initializer="he_normal"
)(total_width1_act)

total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
amp_Batchnormalization = BatchNormalization()(total_amp1)
total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
total_amp3 = Dense(
    1, activation="linear", name="total_amp3", kernel_initializer="he_normal"
)(total_amp1_act)


total_peak_number1 = Dense(
    100, name="total_peak_number1", kernel_initializer="he_normal"
)(x)
peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(
    peak_number_Batchnormalization
)
total_peak_number3 = Dense(
    1,
    activation="linear",
    name="total_peak_number3",
    kernel_initializer="he_normal",
)(total_peak_number1_act)

model_lenet = Model(
    inputs=input_data,
    outputs=[total_center3, total_width3, total_amp3, total_peak_number3],
)
print(model_lenet.summary())


# In[202]:


plot_model(model_lenet, show_shapes=True)


# In[203]:

# Lenet with mae

model_lenet.compile(
    optimizer="adam",
    loss={
        "total_center3": "mae",
        "total_width3": "mae",
        "total_amp3": "mae",
        "total_peak_number3": "mae",
    },
    loss_weights={
        "total_center3": 1,
        "total_width3": 10,
        "total_amp3": 20,
        "total_peak_number3": 2,
    },
    metrics=["mae"],
)


# In[204]:


# lenet
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_model_lenet_mae.h5", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)
# 4 7


# In[205]:


# lenet
models_lenet = model_lenet.fit(
    train_peak,
    [
        np.array(train_center),
        np.array(train_width),
        np.array(train_amp),
        np.array(train_peak_number),
    ],
    epochs=5,
    batch_size=512,
    validation_data=(
        val_peak,
        [
            np.array(val_center),
            np.array(val_width),
            np.array(val_amp),
            np.array(val_peak_number),
        ],
    ),
    callbacks=[model_checkpoint, reduce_lr],
    shuffle=True,
)


# In[206]:


for key in models_lenet.history.keys():
    print(key)


# In[207]:


plt.figure(figsize=(25, 15))

plt.subplot(231)
plt.plot(models_lenet.history["loss"], "b-", label="Lenet - training")
plt.plot(models_lenet.history["val_loss"], "r:", label="Lenet - validation")
plt.grid(True)
plt.title("Total Loss", size=32)
plt.legend()


plt.subplot(232)
plt.plot(
    models_lenet.history["total_center3_loss"], "b-", label="Lenet - training"
)
plt.plot(
    models_lenet.history["val_total_center3_loss"],
    "r:",
    label="Lenet - validation",
)
plt.grid(True)
plt.title("center Loss", size=32)
plt.legend()

plt.subplot(234)
plt.plot(
    models_lenet.history["total_width3_loss"], "b-", label="Lenet - training"
)
plt.plot(
    models_lenet.history["val_total_width3_loss"],
    "r:",
    label="Lenet - validation",
)
plt.grid(True)
plt.title("width Loss", size=32)
plt.legend()

plt.subplot(235)
plt.plot(
    models_lenet.history["total_amp3_loss"], "b-", label="Lenet - training"
)
plt.plot(
    models_lenet.history["val_total_amp3_loss"],
    "r:",
    label="Lenet - validation",
)
plt.grid(True)
plt.title("amp Loss", size=32)
plt.legend()


# In[208]:


plt.plot(
    models_lenet.history["total_peak_number3_loss"],
    "b-",
    label="SE-Resnet - training",
)
plt.plot(
    models_lenet.history["val_total_peak_number3_loss"],
    "r:",
    label="SE-Resnet - validation",
)
plt.grid(True)
plt.ylim(0, 0.03)
plt.title("peak number Loss", size=32)
plt.legend()


# In[138]:


from tensorflow.keras.models import load_model

best_model_lenet = load_model(
    "best_model_lenet_mae.h5"
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지 - "Loading the model with the best performance"
best_model_lenet.summary()


# In[139]:


prediction_lenet = best_model_lenet.predict(test_peak)
print(len(prediction_lenet))
print(np.array(prediction_lenet).shape)


# In[338]:


x = np.linspace(0, 15, 401)

for i in range(0, 15):
    plt.figure(figsize=(10, 5))
    plt.plot(x, test_peak[i], c="black")
    #     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(
        x,
        y(
            prediction_lenet[0][i][0],
            prediction_lenet[1][i][0],
            prediction_lenet[2][i][0],
            x,
        ),
        label="predict",
        c="red",
    )

    plt.plot(
        x,
        y(test_center[i], test_width[i], test_amp[i], x),
        c="blue",
        label="real",
    )
    #     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[140]:


loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += abs((test_center[i] - prediction_lenet[0][i][0]))
    loss_width += abs((test_width[i] - prediction_lenet[1][i][0]))
    loss_amp += abs((test_amp[i] - prediction_lenet[2][i][0]))
    loss_peak_number += abs((test_peak_number[i] - prediction_lenet[3][i][0]))

lenet_loss_center_mae = loss_center / len(test_center)
lenet_loss_width_mae = loss_width / len(test_center)
lenet_loss_amp_mae = loss_amp / len(test_center)
lenet_loss_peak_number_mae = loss_peak_number / len(test_center)


# In[141]:


print(lenet_loss_center_mae)
print(lenet_loss_width_mae)
print(lenet_loss_amp_mae)
print(lenet_loss_peak_number_mae)

# mse
# 0.3048221954039839
# 0.0599072047609455
# 0.03834416206222157
# 0.16315269697927653

# mae
# 0.23661864752383063
# 0.045261228087972255
# 0.02944405885147966
# 0.10858076005911464


# In[106]:

# lenet with mse

model_lenet.compile(
    optimizer="adam",
    loss={
        "total_center3": "mse",
        "total_width3": "mse",
        "total_amp3": "mse",
        "total_peak_number3": "mse",
    },
    loss_weights={
        "total_center3": 1,
        "total_width3": 10,
        "total_amp3": 20,
        "total_peak_number3": 2,
    },
    metrics=["mse"],
)


# In[204]:


# lenet
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_model_lenet_mse.h5", save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)
# 4 7


# In[205]:


# lenet
models_lenet = model_lenet.fit(
    train_peak,
    [
        np.array(train_center),
        np.array(train_width),
        np.array(train_amp),
        np.array(train_peak_number),
    ],
    epochs=5,
    batch_size=512,
    validation_data=(
        val_peak,
        [
            np.array(val_center),
            np.array(val_width),
            np.array(val_amp),
            np.array(val_peak_number),
        ],
    ),
    callbacks=[model_checkpoint, reduce_lr],
    shuffle=True,
)


# In[206]:


for key in models_lenet.history.keys():
    print(key)


# In[207]:


plt.figure(figsize=(25, 15))

plt.subplot(231)
plt.plot(models_lenet.history["loss"], "b-", label="Lenet - training")
plt.plot(models_lenet.history["val_loss"], "r:", label="Lenet - validation")
plt.grid(True)
plt.title("Total Loss", size=32)
plt.legend()


plt.subplot(232)
plt.plot(
    models_lenet.history["total_center3_loss"], "b-", label="Lenet - training"
)
plt.plot(
    models_lenet.history["val_total_center3_loss"],
    "r:",
    label="Lenet - validation",
)
plt.grid(True)
plt.title("center Loss", size=32)
plt.legend()

plt.subplot(234)
plt.plot(
    models_lenet.history["total_width3_loss"], "b-", label="Lenet - training"
)
plt.plot(
    models_lenet.history["val_total_width3_loss"],
    "r:",
    label="Lenet - validation",
)
plt.grid(True)
plt.title("width Loss", size=32)
plt.legend()

plt.subplot(235)
plt.plot(
    models_lenet.history["total_amp3_loss"], "b-", label="Lenet - training"
)
plt.plot(
    models_lenet.history["val_total_amp3_loss"],
    "r:",
    label="Lenet - validation",
)
plt.grid(True)
plt.title("amp Loss", size=32)
plt.legend()


# In[208]:


plt.plot(
    models_lenet.history["total_peak_number3_loss"],
    "b-",
    label="SE-Resnet - training",
)
plt.plot(
    models_lenet.history["val_total_peak_number3_loss"],
    "r:",
    label="SE-Resnet - validation",
)
plt.grid(True)
plt.ylim(0, 0.03)
plt.title("peak number Loss", size=32)
plt.legend()


# In[138]:


from tensorflow.keras.models import load_model

best_model_lenet = load_model(
    "best_model_lenet_mse.h5"
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지 - "Loading the model with the best performance"
best_model_lenet.summary()


# In[139]:


prediction_lenet = best_model_lenet.predict(test_peak)
print(len(prediction_lenet))
print(np.array(prediction_lenet).shape)


# In[338]:


x = np.linspace(0, 15, 401)

for i in range(0, 15):
    plt.figure(figsize=(10, 5))
    plt.plot(x, test_peak[i], c="black")
    #     plt.plot(x,y(prediction[1][i][0],prediction[2][i][0],prediction[3][i][0],x),c = 'red')
    plt.plot(
        x,
        y(
            prediction_lenet[0][i][0],
            prediction_lenet[1][i][0],
            prediction_lenet[2][i][0],
            x,
        ),
        label="predict",
        c="red",
    )

    plt.plot(
        x,
        y(test_center[i], test_width[i], test_amp[i], x),
        c="blue",
        label="real",
    )
    #     plt.plot(x,y(prediction[1][i][1],prediction[2][i][1],prediction[3][i][1],x),c = 'blue')
    plt.legend()
    plt.title(i)


# In[140]:


loss_center = 0
loss_width = 0
loss_amp = 0
loss_peak_number = 0

for i in range(len(test_center)):
    loss_center += np.square((test_center[i] - prediction_lenet[0][i][0]))
    loss_width += np.square((test_width[i] - prediction_lenet[1][i][0]))
    loss_amp += np.square((test_amp[i] - prediction_lenet[2][i][0]))
    loss_peak_number += np.square(
        (test_peak_number[i] - prediction_lenet[3][i][0])
    )

lenet_loss_center_mse = np.sqrt(loss_center / len(test_center))
lenet_loss_width_mse = np.sqrt(loss_width / len(test_center))
lenet_loss_amp_mse = np.sqrt(loss_amp / len(test_center))
lenet_loss_peak_number_mse = np.sqrt(loss_peak_number / len(test_center))


# In[141]:


print(lenet_loss_center_mse)
print(lenet_loss_width_mse)
print(lenet_loss_amp_mse)
print(lenet_loss_peak_number_mse)


# In[214]:


plt.figure(figsize=(25, 15))

# total_train_loss
plt.subplot(231)
plt.plot(
    models_sparse_densenet.history["loss"],
    "b-",
    label="Sparse DenseNet - train",
)
plt.plot(
    models_se_resnet.history["loss"], "b-", label="SE-ResNet - train", c="m"
)
plt.plot(models_resnet.history["loss"], "b-", label="ResNet - train", c="red")
plt.plot(
    models_vggnet.history["loss"], "b-", label="VGGNet - train", c="firebrick"
)
plt.plot(
    models_alex_zfnet.history["loss"],
    "b-",
    label="Alex-ZFnet - train",
    c="sandybrown",
)
plt.plot(models_lenet.history["loss"], "b-", label="LeNet - train", c="gold")
plt.grid(True)
plt.ylim(0, 3)
plt.xlim(0, 5)
plt.xlabel("Epoch", size=20)
plt.ylabel("Training error", size=20)
plt.tick_params(axis="y", labelsize=15)
plt.tick_params(axis="x", labelsize=15)
plt.title("Total training Loss", size=32)
plt.legend(fontsize="x-large")


# total_val_loss
plt.subplot(232)
plt.plot(
    models_sparse_densenet.history["val_loss"],
    "b-",
    label="Sparse DenseNet - val",
)
plt.plot(
    models_se_resnet.history["val_loss"], "b-", label="SE-ResNet - val", c="m"
)
plt.plot(
    models_resnet.history["val_loss"], "b-", label="ResNet - val", c="red"
)
plt.plot(
    models_vggnet.history["val_loss"],
    "b-",
    label="VGGNet - val",
    c="firebrick",
)
plt.plot(
    models_alex_zfnet.history["val_loss"],
    "b-",
    label="Alex-ZFNet - val",
    c="sandybrown",
)
plt.plot(models_lenet.history["val_loss"], "b-", label="LeNet - val", c="gold")
plt.grid(True)
plt.ylim(0, 3)
plt.xlim(0, 5)
plt.xlabel("Epoch", size=20)
plt.ylabel("Training error", size=20)
plt.tick_params(axis="y", labelsize=15)
plt.tick_params(axis="x", labelsize=15)
plt.title("Total validation Loss", size=32)
plt.legend(fontsize="x-large")


# center_train_loss
plt.subplot(234)
plt.plot(
    models_sparse_densenet.history["total_center3_loss"],
    "b-",
    label="Sparse DenseNet - train",
)
plt.plot(
    models_se_resnet.history["total_center3_loss"],
    "b-",
    label="SE-ResNet - train",
    c="m",
)
plt.plot(
    models_resnet.history["total_center3_loss"],
    "b-",
    label="ResNet - train",
    c="red",
)
plt.plot(
    models_vggnet.history["total_center3_loss"],
    "b-",
    label="VGGNet - train",
    c="firebrick",
)
plt.plot(
    models_alex_zfnet.history["total_center3_loss"],
    "b-",
    label="Alex-ZFnet - train",
    c="sandybrown",
)
plt.plot(
    models_lenet.history["total_center3_loss"],
    "b-",
    label="LeNet - train",
    c="gold",
)
plt.grid(True)
plt.ylim(0, 3)
plt.xlim(0, 5)
plt.ylabel("Validation error", size=20)
plt.xlabel("Epoch", size=20)
plt.tick_params(axis="y", labelsize=15)
plt.tick_params(axis="x", labelsize=15)
plt.title("Training center Loss", size=32)
plt.legend(fontsize="x-large")

# center_val_loss
plt.subplot(235)
plt.subplots_adjust(hspace=0.25)
plt.plot(
    models_sparse_densenet.history["val_total_center3_loss"],
    "b-",
    label="Sparse DenseNet - val",
)
plt.plot(
    models_se_resnet.history["val_total_center3_loss"],
    "b-",
    label="SE-ResNet - val",
    c="m",
)
plt.plot(
    models_resnet.history["val_total_center3_loss"],
    "b-",
    label="ResNet - val",
    c="red",
)
plt.plot(
    models_vggnet.history["val_total_center3_loss"],
    "b-",
    label="VGGNet - val",
    c="firebrick",
)
plt.plot(
    models_alex_zfnet.history["val_total_center3_loss"],
    "b-",
    label="Alex-ZFNet - val",
    c="sandybrown",
)
plt.plot(
    models_lenet.history["val_total_center3_loss"],
    "b-",
    label="LeNet - val",
    c="gold",
)
plt.grid(True)
plt.ylim(0, 3)
plt.xlim(0, 5)
plt.ylabel("Validation error", size=20)
plt.xlabel("Epoch", size=20)
plt.tick_params(axis="y", labelsize=15)
plt.tick_params(axis="x", labelsize=15)
plt.title("Validation center Loss", size=32)
plt.legend(loc="upper right", fontsize="x-large")


# In[130]:


plt.figure(figsize=(25, 25))

plt.subplot(211)
plt.plot(
    models_sparse_densenet.history["total_peak_number3_loss"],
    "b-",
    label="SE-Res-Densenet - training",
)
plt.plot(
    models_se_resnet.history["total_peak_number3_loss"],
    "b-",
    label="SE-Resnet - training",
    c="m",
)
plt.plot(
    models_resnet.history["total_peak_number3_loss"],
    "b-",
    label="Resnet - training",
    c="red",
)
plt.plot(
    models_vggnet.history["total_peak_number3_loss"],
    "b-",
    label="VGGnet - training",
    c="firebrick",
)
plt.plot(
    models_alex_zfnet.history["total_peak_number3_loss"],
    "b-",
    label="Alex-ZFnet - training",
    c="sandybrown",
)
plt.plot(
    models_lenet.history["total_peak_number3_loss"],
    "b-",
    label="Lenet - training",
    c="gold",
)
plt.grid(True)
plt.title("Train peak detection Loss", size=32)
plt.legend()
plt.ylim(0, 0.2)

plt.subplot(212)
plt.plot(
    models_sparse_densenet.history["val_total_peak_number3_loss"],
    "b-",
    label="SE-Res-Densenet - training",
)
plt.plot(
    models_se_resnet.history["val_total_peak_number3_loss"],
    "b-",
    label="SE-Resnet - training",
    c="m",
)
plt.plot(
    models_resnet.history["val_total_peak_number3_loss"],
    "b-",
    label="Resnet - training",
    c="red",
)
plt.plot(
    models_vggnet.history["val_total_peak_number3_loss"],
    "b-",
    label="VGGnet - training",
    c="firebrick",
)
plt.plot(
    models_alex_zfnet.history["val_total_peak_number3_loss"],
    "b-",
    label="Alex-ZFnet - training",
    c="sandybrown",
)
plt.plot(
    models_lenet.history["val_total_peak_number3_loss"],
    "b-",
    label="Lenet - training",
    c="gold",
)
plt.grid(True)
plt.title("Validation peak detection Loss", size=32)
plt.legend()
plt.ylim(0, 0.2)


# In[147]:


import pandas as pd

compare_error_mse = np.zeros((6, 4))
compare_error_mse[0] = np.array(
    (
        lenet_loss_center_mse,
        lenet_loss_width_mse,
        lenet_loss_amp_mse,
        lenet_loss_peak_number_mse,
    )
)
compare_error_mse[1] = np.array(
    (
        alex_zfnet_loss_center_mse,
        alex_zfnet_loss_width_mse,
        alex_zfnet_loss_amp_mse,
        alex_zfnet_loss_peak_number_mse,
    )
)
compare_error_mse[2] = np.array(
    (
        vggnet_loss_center_mse,
        vggnet_loss_width_mse,
        vggnet_loss_amp_mse,
        vggnet_loss_peak_number_mse,
    )
)
compare_error_mse[3] = np.array(
    (
        resnet_loss_center_mse,
        resnet_loss_width_mse,
        resnet_loss_amp_mse,
        resnet_loss_peak_number_mse,
    )
)
compare_error_mse[4] = np.array(
    (
        se_resnet_loss_center_mse,
        se_resnet_loss_width_mse,
        se_resnet_loss_amp_mse,
        se_resnet_loss_peak_number_mse,
    )
)
compare_error_mse[5] = np.array(
    (
        sparse_densenet_loss_center_mse,
        sparse_densenet_loss_width_mse,
        sparse_densenet_loss_amp_mse,
        sparse_densenet_loss_peak_number_mse,
    )
)
architectures_mse = pd.DataFrame(
    compare_error_mse,
    index=[
        "LeNet",
        "Alex-ZFNet",
        "VGGNet",
        "ResNet",
        "SE-ResNet",
        "Sparse DenseNet",
    ],
    columns=["center", "width", "amp", "peak_number"],
)
architectures_mse


# In[148]:


compare_error_mae = np.zeros((6, 4))

compare_error_mae[0] = np.array(
    (
        lenet_loss_center_mae,
        lenet_loss_width_mae,
        lenet_loss_amp_mae,
        lenet_loss_peak_number_mae,
    )
)
compare_error_mae[1] = np.array(
    (
        alex_zfnet_loss_center_mae,
        alex_zfnet_loss_width_mae,
        alex_zfnet_loss_amp_mae,
        alex_zfnet_loss_peak_number_mae,
    )
)
compare_error_mae[2] = np.array(
    (
        vggnet_loss_center_mae,
        vggnet_loss_width_mae,
        vggnet_loss_amp_mae,
        vggnet_loss_peak_number_mae,
    )
)
compare_error_mae[3] = np.array(
    (
        resnet_loss_center_mae,
        resnet_loss_width_mae,
        resnet_loss_amp_mae,
        resnet_loss_peak_number_mae,
    )
)
compare_error_mae[4] = np.array(
    (
        se_resnet_loss_center_mae,
        se_resnet_loss_width_mae,
        se_resnet_loss_amp_mae,
        se_resnet_loss_peak_number_mae,
    )
)
compare_error_mae[5] = np.array(
    (
        sparse_densenet_loss_center_mae,
        sparse_densenet_loss_width_mae,
        sparse_densenet_loss_amp_mae,
        sparse_densenet_loss_peak_number_mae,
    )
)
architectures_mae = pd.DataFrame(
    compare_error_mae,
    index=[
        "LeNet",
        "Alex-ZFNet",
        "VGGNet",
        "ResNet",
        "SE-ResNet",
        "Sparse DenseNet",
    ],
    columns=["center", "width", "amp", "peak_number"],
)
architectures_mae


# In[158]:


# architectures['MSE']
# architectures.index

plt.figure(figsize=(25, 15))

bar_width = 0.35
indexs = np.arange(6)
p1 = plt.bar(indexs, architectures_mse["center"], bar_width, label="MSE")
p2 = plt.bar(
    indexs + bar_width, architectures_mae["center"], bar_width, label="MAE"
)

plt.xticks(indexs + 0.17, architectures_mae.index, fontsize=30)
plt.yticks(fontsize=35)
plt.ylabel("Center Loss", fontsize=50)
plt.xlabel("CNN architectures", fontsize=50)
plt.legend(fontsize=45, frameon=True, shadow=True, framealpha=0.95)
plt.ylim(0, 0.32)


# In[165]:


plt.figure(figsize=(25, 15))

bar_width = 0.35
indexs = np.arange(6)
p1 = plt.bar(indexs, architectures_mse["width"], bar_width, label="MSE")
p2 = plt.bar(
    indexs + bar_width, architectures_mae["width"], bar_width, label="MAE"
)

plt.xticks(indexs + 0.17, architectures_mae.index, fontsize=30)
plt.yticks(fontsize=35)
plt.ylabel("Width Loss", fontsize=50)
plt.xlabel("CNN architectures", fontsize=50)
plt.legend(fontsize=45, frameon=True, shadow=True, framealpha=0.95)
plt.ylim(0, 0.1)


# In[166]:


plt.figure(figsize=(25, 15))

bar_width = 0.35
indexs = np.arange(6)
p1 = plt.bar(indexs, architectures_mse["amp"], bar_width, label="MSE")
p2 = plt.bar(
    indexs + bar_width, architectures_mae["amp"], bar_width, label="MAE"
)

plt.xticks(indexs + 0.17, architectures_mae.index, fontsize=30)
plt.yticks(fontsize=35)
plt.ylabel("Amplitude Loss", fontsize=50)
plt.xlabel("CNN architectures", fontsize=50)
plt.legend(fontsize=45, frameon=True, shadow=True, framealpha=0.95)
plt.ylim(0, 0.1)


# In[164]:


plt.figure(figsize=(25, 15))

p1 = plt.bar(indexs, architectures_mse["peak_number"], bar_width, label="MSE")
p2 = plt.bar(
    indexs + bar_width,
    architectures_mae["peak_number"],
    bar_width,
    label="MAE",
)

plt.xticks(indexs + 0.17, architectures_mae.index, fontsize=30)
plt.yticks(fontsize=35)
plt.ylabel("Peak number Loss", fontsize=50)
plt.xlabel("CNN architectures", fontsize=50)
plt.legend(fontsize=45, frameon=True, shadow=True, framealpha=0.95)
plt.ylim(0, 0.32)


# In[1056]:

plt.figure(figsize=(50, 5))

bar_width = 0.6

p1 = plt.bar(indexs, architectures_mse["center"], bar_width, label="MSE")


plt.xticks(indexs, architectures_mae.index, fontsize=8, rotation=0)
plt.ylabel("Center Loss", fontsize=50)
plt.xlabel("CNN architectures", fontsize=50)
plt.legend(fontsize=12)


# In[332]:


# lenet 으로 안잡힌 data 조사

t = 0
lenet_not_catch = []
for i in range(len(test_peak_number)):
    if round(prediction_lenet[3][i][0], 0) - test_peak_number[i] != 0:
        lenet_not_catch.append(t)
    t += 1

len(lenet_not_catch)


# In[ ]:


150000 - 13723


# In[333]:


print(len(lenet_not_catch) / len(test_peak_number) * 100, "%의 오차율")
print(
    (len(test_peak_number) - len(lenet_not_catch))
    / len(test_peak_number)
    * 100,
    "%의 정확도",
)

# mae
# 8.043940467753366 %의 오차율
# 91.95605953224664 %의 정확도

# mse
# 9.175213618068277 %의 오차율
# 90.82478638193172 %의 정확도


# In[334]:


# SE-Dens-Resnet으로 안잡힌 data 조사

t = 0
sparse_densenet_not_catch = []
for i in range(len(test_peak_number)):
    if (
        round(prediction_sparse_densenet[3][i][0], 0) - test_peak_number[i]
        != 0
    ):
        sparse_densenet_not_catch.append(t)
    t += 1

len(sparse_densenet_not_catch)


# In[342]:


150000 - 3181


# In[335]:


print(len(sparse_densenet_not_catch) / len(test_peak_number) * 100, "%의 오차율")
print(
    (len(test_peak_number) - len(sparse_densenet_not_catch))
    / len(test_peak_number)
    * 100,
    "%의 정확도",
)

# mae
# 2.215075618790367 %의 오차율
# 97.78492438120963 %의 정확도

# mse
# 2.12682026663814 %의 오차율
# 97.87317973336187 %의 정확도


# In[23]:


from tensorflow.keras.models import load_model

best_model_sparse_densenet = load_model(
    "best_model_sparse_densenet_mae.h5"
)  # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지 - "Loading the model with the best performance"
# best_model_sparse_densenet= load_model('best_model_se_res_densenet_finish1_1000.h5')

best_model_sparse_densenet.summary()


# In[1091]:


# for i in range(300,400):
i = 6

plt.figure(figsize=(5, 5))
for k in range(len(test_peak_param[i])):
    plt.plot(x, test_peak[i], c="black")
    #     plt.plot(x,y(test_peak_param[i][k][0],test_peak_param[i][k][1],test_peak_param[i][k][2],x),c = 'blue', linewidth = 3)
    plt.ylim(0, 1)
    plt.title("Input", fontsize=32)

#       6 123 141 209


# In[1092]:


i = 6

plt.figure(figsize=(5, 5))
for k in range(len(test_peak_param[i])):
    #     plt.plot(x,test_peak[i],c = 'black')
    plt.plot(
        x,
        y(
            test_peak_param[i][k][0],
            test_peak_param[i][k][1],
            test_peak_param[i][k][2],
            x,
        ),
        c="blue",
        linewidth=3,
    )
    plt.ylim(0, 1)
    plt.title("Prediction", fontsize=32)


# In[1009]:


# x = np.linspace(0,15,401)
for i in range(0, 15):  # range(700,800) originally
    plt.figure(figsize=(10, 5))
    plt.plot(x, test_peak[i])
    plt.plot(
        x,
        y(
            prediction_sparse_densenet[0][i][0],
            prediction_sparse_densenet[1][i][0],
            prediction_sparse_densenet[2][i][0],
            x,
        ),
        label="predict",
        c="red",
    )
    plt.plot(
        x,
        y(test_center[i], test_width[i], test_amp[i], x),
        c="blue",
        label="real",
    )
    plt.ylim(-0.1, 1.1)
    plt.title(i)
#     26
#     57
#     83
#     128
#     144
#     170
#     309, 322, 338, 373
#     409, 455, 466, 477
#     506
#     608, 628, 667


# In[1010]:


# 빼고
for i in range(0, 15):  # range(700,800) originally
    plt.figure(figsize=(10, 5))
    first_correction = test_peak[i].reshape(
        401,
    ) - y(
        prediction_sparse_densenet[0][i][0],
        prediction_sparse_densenet[1][i][0],
        prediction_sparse_densenet[2][i][0],
        x,
    )
    plt.plot(x, first_correction)
    plt.ylim(-0.1, 1.1)
    plt.title(i)
    plt.axhline(y=0, color="r")


# In[1011]:


# 빼고
for i in range(0, 15):  #
    plt.figure(figsize=(10, 5))
    first_correction = test_peak[i].reshape(
        401,
    ) - y(
        prediction_sparse_densenet[0][i][0],
        prediction_sparse_densenet[1][i][0],
        prediction_sparse_densenet[2][i][0],
        x,
    )
    for k in range(first_correction.shape[0]):  # 401번 반복
        if first_correction[k] < 0.0:
            first_correction[k] = 0
    plt.plot(x, first_correction)
    plt.ylim(-0.1, 1.1)
    plt.title(i)


# In[1030]:


#     409, 455, 466, 477
#     506
#     608, 628, 667
i = 4
print(round(prediction_sparse_densenet[3][i][0], 0))
print(test_peak_number[i])
# 409 506


# In[1031]:


# first 예측
x = np.linspace(0, 15, 401)

# plt.figure(figsize = (10,5))
plt.plot(x, test_peak[i], c="black")
plt.plot(
    x,
    y(
        prediction_sparse_densenet[0][i][0],
        prediction_sparse_densenet[1][i][0],
        prediction_sparse_densenet[2][i][0],
        x,
    ),
    label="predict",
    c="red",
)
plt.plot(
    x,
    y(test_center[i], test_width[i], test_amp[i], x),
    c="blue",
    label="target",
)
plt.legend(fontsize="large")
plt.ylim(-0.1, 1.1)
plt.xlim(0, 15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Predict", fontsize=20)
plt.grid()
plt.title("Predict", fontsize=32)


save_param = []
save_param.append(prediction_sparse_densenet[0][i][0])
save_param.append(prediction_sparse_densenet[1][i][0])
save_param.append(prediction_sparse_densenet[2][i][0])
save_param


# In[1032]:


# first빼고

first_correction = test_peak[i].reshape(
    401,
) - y(
    prediction_sparse_densenet[0][i][0],
    prediction_sparse_densenet[1][i][0],
    prediction_sparse_densenet[2][i][0],
    x,
)
plt.plot(x, first_correction, c="black")
plt.ylim(-0.1, 1.1)
plt.xlim(0, 15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Truncate", fontsize=32)
plt.grid()


# In[1033]:


# first 보정

for k in range(first_correction.shape[0]):  # 401번 반복
    if first_correction[k] < 0.0:
        first_correction[k] = 0
plt.plot(x, first_correction, c="black")
plt.ylim(-0.1, 1.1)
plt.xlim(0, 15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.title("Correction", fontsize=32)


# In[1034]:


# second 예측

second_process = best_model_sparse_densenet.predict(
    first_correction.reshape(1, 401, 1)
)
plt.plot(x, first_correction, c="black")
plt.plot(
    x,
    y(second_process[0][0], second_process[1][0], second_process[2][0], x),
    label="predict",
    c="red",
)
i = 2
plt.plot(
    x,
    y(
        test_peak_param[i][2][0],
        test_peak_param[i][2][1],
        test_peak_param[i][2][2],
        x,
    ),
    c="blue",
    label="target",
)

plt.ylim(-0.1, 1.1)
plt.xlim(0, 15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize="large", loc="upper left")
plt.grid()

save_param.append(second_process[0][0][0])
save_param.append(second_process[1][0][0])
save_param.append(second_process[2][0][0])
save_param


# In[1035]:


# second 빼고
second_correction = first_correction - y(
    second_process[0][0], second_process[1][0], second_process[2][0], x
)
plt.plot(x, second_correction, c="black")
plt.ylim(-0.1, 1.1)
plt.xlim(0, 15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()


# In[1036]:


# second 보정

for k in range(second_correction.shape[0]):  # 401번 반복
    if second_correction[k] < 0.0:
        second_correction[k] = 0
plt.plot(x, second_correction, c="black")
plt.ylim(-0.1, 1.1)
plt.xlim(0, 15)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# In[1037]:


# third 예측

third_process = best_model_sparse_densenet.predict(
    second_correction.reshape(1, 401, 1)
)
plt.plot(x, second_correction)
plt.plot(
    x, y(third_process[0][0], third_process[1][0], third_process[2][0], x)
)
plt.ylim(-0.1, 1.1)

save_param.append(third_process[0][0][0])
save_param.append(third_process[1][0][0])
save_param.append(third_process[2][0][0])
save_param


# In[1038]:


# third 빼고
third_correction = second_correction - y(
    third_process[0][0], third_process[1][0], third_process[2][0], x
)
plt.plot(x, third_correction)
plt.ylim(-0.1, 1.1)

# third 보정

for k in range(third_correction.shape[0]):  # 401번 반복
    if third_correction[k] < 0.0:
        third_correction[k] = 0
plt.plot(x, third_correction)
plt.ylim(-0.1, 1.1)


# In[1039]:


# four 예측

four_process = best_model_sparse_densenet.predict(
    third_correction.reshape(1, 401, 1)
)
plt.plot(x, third_correction)
plt.plot(x, y(four_process[0][0], four_process[1][0], four_process[2][0], x))
plt.ylim(-0.1, 1.1)

save_param.append(four_process[0][0][0])
save_param.append(four_process[1][0][0])
save_param.append(four_process[2][0][0])
save_param


# In[1040]:


# four 빼고
four_correction = third_correction - y(
    four_process[0][0], four_process[1][0], four_process[2][0], x
)
plt.plot(x, four_correction)
plt.ylim(-0.5, 1.1)

# four 보정

for k in range(four_correction.shape[0]):  # 401번 반복
    if four_correction[k] < 0.0:
        four_correction[k] = 0
plt.plot(x, four_correction)
plt.ylim(-0.1, 1.1)


# In[1041]:


# five 예측

five_process = best_model_sparse_densenet.predict(
    four_correction.reshape(1, 401, 1)
)
plt.plot(x, four_correction)
plt.plot(x, y(five_process[0][0], five_process[1][0], five_process[2][0], x))
plt.ylim(-0.1, 1.1)

save_param.append(five_process[0][0][0])
save_param.append(five_process[1][0][0])
save_param.append(five_process[2][0][0])
save_param


# In[1042]:


# five 빼고
five_correction = four_correction - y(
    five_process[0][0], five_process[1][0], five_process[2][0], x
)
plt.plot(x, five_correction)
plt.ylim(-0.2, 1.1)

# five 보정

for k in range(five_correction.shape[0]):  # 401번 반복
    if five_correction[k] < 0.0:
        five_correction[k] = 0
plt.plot(x, five_correction)
plt.ylim(-0.1, 1.1)


# In[1048]:


# len(save_param)
print(save_param)
predict_area = (
    y(save_param[0], save_param[1], save_param[2], x)
    + y(save_param[3], save_param[4], save_param[5], x)
    + y(save_param[6], save_param[7], save_param[8], x)
    + y(save_param[9], save_param[10], save_param[11], x)
    + y(save_param[12], save_param[13], save_param[14], x)
)

plt.plot(x, predict_area, label="predict area", c="green")
plt.plot(x, test_peak[i], label="target area", c="royalblue")

plt.legend()

print(
    "test peak area = ",
    sum(
        test_peak[i].reshape(
            401,
        )
    ),
)
print("predict peak area = ", sum(predict_area))

# residual_area = (test_peak[i].reshape(401,)-predict_area)**2
residual_area = abs(
    test_peak[i].reshape(
        401,
    )
    - predict_area
)

for k in range(401):
    if residual_area[k] < 0.01:  # 너무 noise한 부분도 고려하니까 없애주자
        residual_area[k] = 0

print(residual_area.shape)
# print(residual_area)
print("residual area = ", sum(residual_area))


plt.plot(x, residual_area, label="residual area", c="red")
plt.legend(fontsize="large")
plt.grid()
plt.xlim(0, 15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# plt.axvline(x = save_param[0], c = 'black', linewidth = 1, linestyle = '--')
# plt.axvline(x = save_param[3], c = 'black', linewidth = 1, linestyle = '--')
# plt.axvline(x = save_param[6], c = 'black', linewidth = 1, linestyle = '--')
# plt.axvline(x = save_param[9], c = 'black', linewidth = 1, linestyle = '--')
# plt.axvline(x = save_param[12], c = 'black', linewidth = 1, linestyle = '--')


# In[1047]:

plt.plot(
    x,
    y(save_param[0], save_param[1], save_param[2], x),
    c="green",
    label="predict",
)
plt.plot(x, y(save_param[3], save_param[4], save_param[5], x), c="green")
plt.plot(x, y(save_param[6], save_param[7], save_param[8], x), c="green")
plt.plot(x, y(save_param[9], save_param[10], save_param[11], x), c="green")
plt.plot(x, y(save_param[12], save_param[13], save_param[14], x), c="green")

plt.plot(
    x,
    y(
        test_peak_param[i][0][0],
        test_peak_param[i][0][1],
        test_peak_param[i][0][2],
        x,
    ),
    c="royalblue",
    label="target",
)
plt.plot(
    x,
    y(
        test_peak_param[i][1][0],
        test_peak_param[i][1][1],
        test_peak_param[i][1][2],
        x,
    ),
    c="royalblue",
)
plt.plot(
    x,
    y(
        test_peak_param[i][2][0],
        test_peak_param[i][2][1],
        test_peak_param[i][2][2],
        x,
    ),
    c="royalblue",
)


# plt.plot(
#     x,
#     y(
#         test_peak_param[i][3][0],
#         test_peak_param[i][3][1],
#         test_peak_param[i][3][2],
#         x,
#     ),
#     c="royalblue",
# )
# plt.plot(
#     x,
#     y(
#         test_peak_param[i][4][0],
#         test_peak_param[i][4][1],
#         test_peak_param[i][4][2],
#         x,
#     ),
#     c="royalblue",
# )
plt.grid()
plt.legend(fontsize="large")
plt.xlim(0, 15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# In[1045]:


import numpy as np
import scipy.optimize as sco

i = 2

# test_peak_area = sum(test_peak[i])
test_peak_area = test_peak[i].reshape(
    401,
)
beta = 5.09791537e-01
gamma = 4.41140472e-01
x = np.linspace(0, 15, 401)


def y2(p):
    (a_1,b_1,c_1,a_2,b_2,c_2,a_3,b_3,c_3,a_4,b_4,c_4,a_5,b_5,c_5) = p

    total_y = sum(abs((c_1*((0.7*np.exp(-np.log(2) * (x - a_1) ** 2 / (beta * b_1) ** 2))
                    +(0.3 / (1 + (x - a_1) ** 2 / (gamma * b_1) ** 2))))
            +(c_2*((0.7*np.exp(-np.log(2) * (x - a_2) ** 2 / (beta * b_2) ** 2))
                    +(0.3 / (1 + (x - a_2) ** 2 / (gamma * b_2) ** 2))))
            +(c_3*((0.7*np.exp(-np.log(2) * (x - a_3) ** 2 / (beta * b_3) ** 2))
                    +(0.3 / (1 + (x - a_3) ** 2 / (gamma * b_3) ** 2))) )
            +(c_4*((0.7*np.exp(-np.log(2) * (x - a_4) ** 2 / (beta * b_4) ** 2))
                    +(0.3 / (1 + (x - a_4) ** 2 / (gamma * b_4) ** 2))))
            +(c_5*((0.7*np.exp(-np.log(2) * (x - a_5) ** 2 / (beta * b_5) ** 2))
                    +(0.3 / (1 + (x - a_5) ** 2 / (gamma * b_5) ** 2))))
            - test_peak_area))

    return total_y


optima = sco.basinhopping(y2,(save_param[0],save_param[1],save_param[2],save_param[3],save_param[4],
                              save_param[5],save_param[6],save_param[7],save_param[8],save_param[9],
                              save_param[10],save_param[11],save_param[12],save_param[13],save_param[14]),
                          niter=100,T=1,stepsize=0.5)
optima


# In[1049]:


plt.plot(
    x,
    y(save_param[0], save_param[1], save_param[2], x),
    c="green",
    label="predict",
)
plt.plot(x, y(save_param[3], save_param[4], save_param[5], x), c="green")
plt.plot(x, y(save_param[6], save_param[7], save_param[8], x), c="green")
plt.plot(x, y(save_param[9], save_param[10], save_param[11], x), c="green")
plt.plot(x, y(save_param[12], save_param[13], save_param[14], x), c="green")

plt.plot(
    x,
    y(
        test_peak_param[i][0][0],
        test_peak_param[i][0][1],
        test_peak_param[i][0][2],
        x,
    ),
    c="blue",
    linewidth=2,
    label="target",
)
plt.plot(
    x,
    y(
        test_peak_param[i][1][0],
        test_peak_param[i][1][1],
        test_peak_param[i][1][2],
        x,
    ),
    c="blue",
    linewidth=2,
)
plt.plot(
    x,
    y(
        test_peak_param[i][2][0],
        test_peak_param[i][2][1],
        test_peak_param[i][2][2],
        x,
    ),
    c="blue",
    linewidth=2,
)
# plt.plot(
#     x,
#     y(
#         test_peak_param[i][3][0],
#         test_peak_param[i][3][1],
#         test_peak_param[i][3][2],
#         x,
#     ),
#     c="blue",
#     linewidth=2,
# )
# plt.plot(
#     x,
#     y(
#         test_peak_param[i][4][0],
#         test_peak_param[i][4][1],
#         test_peak_param[i][4][2],
#         x,
#     ),
#     c="blue",
#     linewidth=2,
# )

plt.plot(
    x,
    y(optima.x[0], optima.x[1], optima.x[2], x),
    c="red",
    linewidth=1.5,
    linestyle="--",
    label="after correction",
)
plt.plot(
    x,
    y(optima.x[3], optima.x[4], optima.x[5], x),
    c="red",
    linewidth=1.5,
    linestyle="--",
)
plt.plot(
    x,
    y(optima.x[6], optima.x[7], optima.x[8], x),
    c="red",
    linewidth=1.5,
    linestyle="--",
)
plt.plot(
    x,
    y(optima.x[9], optima.x[10], optima.x[11], x),
    c="red",
    linewidth=1.5,
    linestyle="--",
)
plt.plot(
    x,
    y(optima.x[12], optima.x[13], optima.x[14], x),
    c="red",
    linewidth=1.5,
    linestyle="--",
)

plt.xlim(0, 15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.grid()
plt.legend(fontsize="large")


# In[24]:


# only three peaks


bg = np.loadtxt("2021Park_ML-peak-fitting/ITO_O1s_bg.txt")
exp = np.loadtxt("2021Park_ML-peak-fitting/ITO_O1s_exp.txt")
fitting = np.loadtxt("2021Park_ML-peak-fitting/ITO_O1s_fitting.txt")
peak1 = np.loadtxt("2021Park_ML-peak-fitting/ITO_O1s_p1.txt")
peak2 = np.loadtxt("2021Park_ML-peak-fitting/ITO_O1s_p2.txt")
peak3 = np.loadtxt("2021Park_ML-peak-fitting/ITO_O1s_p3.txt")

# 테스트하고자 하는 실제 XPS 데이터의 parameter 범위가 너무 크므로
# 네트워크 자체는 작은 규모의 파라미터 범위에서 학습시키고,
# 테스트할 경우, 범위를 줄인 xps 데이터를 불러와서 테스트해본다

plt.figure(figsize=(10, 5))
# exp data에서 background를 제거하고, peak 높이를 1로 normalize한다.
plt.plot(
    exp[:, 0],
    (exp[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    label="exp bg removed",
)

# plt.plot(fitting[:, 0], fitting[:, 1] - bg[:, 1], label = "fitting", linewidth = 2)

# 마찬가지로 개별 peak도 크기를 줄인다.
plt.plot(
    peak1[:, 0],
    (peak1[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    label="peak1",
    linewidth=2,
)
plt.plot(
    peak2[:, 0],
    (peak2[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    label="peak2",
    linewidth=2,
)
plt.plot(
    peak3[:, 0],
    (peak3[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    label="peak3",
    linewidth=2,
)

plt.grid(True)
plt.title("P3HT Fitting and experiment", size=24)
plt.xlabel("Energy range", size=24)
plt.ylabel("Intensity", size=24)
plt.legend()
plt.show()


# In[25]:


test_result = (
    (exp[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max())
).reshape((1, 401, 1))
print(test_result.shape)
# plt.plot( (exp[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()), label = "exp bg removed")
# plt.plot(exp[:, 0], (exp[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()), label = "exp bg removed")

plt.plot(test_result[0], c="black")
plt.title("total p3ht")


# In[26]:
# Commented by me , due to missing generation of the h5 file

# best_model_sparse_densenet= load_model('best_model_sparse_densenet_mse.h5')   # 가장 좋게 성능을 내는 모델을 저장한걸 불러왔지 - "Loading the model with the best performance"
# best_model_sparse_densenet = load_model(
#     "best_model_se_res_densenet_finish1_1000.h5"
# )
# # best_model_sparse_densenet= load_model('best_model_lenet_mae.h5')

# best_model_sparse_densenet.summary()


# In[27]:


predict = best_model_sparse_densenet.predict(test_result)
print(predict)

new_predict = []

for element in predict:
    new_predict.append(element.reshape((element.shape[1])))


# In[28]:


x = np.linspace(0, 15, 401)
# x = np.linspace(0,7,401)
t = np.linspace(0, 401, 401)


# In[29]:


# plt.plot(test_result[0])
plt.plot(test_result[0], c="black", label="ito")
plt.plot(
    t,
    (peak1[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    label="real peak",
    linewidth=2,
    color="purple",
)
plt.plot(
    t,
    (peak2[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    linewidth=2,
    color="purple",
)
plt.plot(
    t,
    (peak3[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    linewidth=2,
    color="purple",
)

plt.plot(
    t,
    y(new_predict[0][0], new_predict[1][0], new_predict[2][0], x),
    label="predict first peak",
    c="blue",
)
plt.title("first peak")
plt.legend()


# In[30]:


plt.plot(
    test_result[0].reshape(
        401,
    )
    - y(new_predict[0][0], new_predict[1][0], new_predict[2][0], x),
    c="black",
)
plt.title("total peak without first peak")
plt.ylim(0, 1)


# In[31]:


test_result2 = test_result[0].reshape(
    401,
) - y(new_predict[0][0], new_predict[1][0], new_predict[2][0], x)

a = y(new_predict[0][0], new_predict[1][0], new_predict[2][0], x)
print(list(a).index(max(a)))

# for j in range(list(a).index(max(a)), len(a), 1):
# #     test_result2[j] = np.random.rand() * noise_level - noise_level * 0.5
#     test_result2[j] = 0

for i in range(test_result2.shape[0]):
    if test_result2[i] < 0.0:
        #         test_result2[i] = np.random.rand() * noise_level - noise_level * 0.5
        test_result2[i] = 0

plt.plot(test_result2)
plt.ylim(0, 1)


# In[32]:


test_result2 = test_result2.reshape(1, 401, 1)
test_result2.shape

predict2 = best_model_sparse_densenet.predict(test_result2)
# print(predict2)

new_predict2 = []

for element in predict2:
    new_predict2.append(element.reshape((element.shape[1])))

# plt.plot(test_result2.reshape(401,),c = 'green')
# plt.plot(test_result[0].reshape(401,)- y(new_predict[0][0],new_predict[1][0],new_predict[2][0],x),c = 'black')
# plt.plot(t, (peak2[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()),color = 'purple', label = "peak2", linewidth = 2)
plt.plot(test_result[0], c="black", label="ito")
plt.plot(
    t,
    (peak1[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    label="real peak",
    linewidth=2,
    color="purple",
)
plt.plot(
    t,
    (peak2[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    linewidth=2,
    color="purple",
)
plt.plot(
    t,
    (peak3[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    linewidth=2,
    color="purple",
)


# plt.plot(test_result2.reshape(401,), c = 'dodgerblue')
plt.plot(
    t,
    y(new_predict2[0][0], new_predict2[1][0], new_predict2[2][0], x),
    label="predict second peak",
    c="blue",
)
plt.title("second peak")
plt.legend()


# In[33]:


plt.plot(
    test_result2[0].reshape(
        401,
    )
    - y(new_predict2[0][0], new_predict2[1][0], new_predict2[2][0], x),
    c="black",
)
plt.title("total peak without second  peak")
plt.ylim(0, 1)


# In[34]:


test_result3 = test_result2[0].reshape(
    401,
) - y(new_predict2[0][0], new_predict2[1][0], new_predict2[2][0], x)

a = y(new_predict2[0][0], new_predict2[1][0], new_predict2[2][0], x)
print(list(a).index(max(a)))

# for j in range(list(a).index(max(a)), len(a), 1):
#     test_result3[j] = 0


for i in range(test_result3.shape[0]):
    if test_result3[i] < 0.0:
        test_result3[i] = 0

plt.plot(test_result3)
plt.ylim(0, 1)


# In[35]:


test_result3 = test_result3.reshape(1, 401, 1)
test_result3.shape

predict3 = best_model_sparse_densenet.predict(test_result3)
# print(predict3)

new_predict3 = []

for element in predict3:
    new_predict3.append(element.reshape((element.shape[1])))

plt.plot(test_result[0], c="black", label="ito")
plt.plot(
    t,
    (peak1[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    label="real peak",
    linewidth=2,
    color="purple",
)
plt.plot(
    t,
    (peak2[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    linewidth=2,
    color="purple",
)
plt.plot(
    t,
    (peak3[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    linewidth=2,
    color="purple",
)

# plt.plot(test_result3.reshape(401,), c = 'dodgerblue')
# plt.plot(test_result2[0].reshape(401,)- y(new_predict2[0][0],new_predict2[1][0],new_predict2[2][0],x),c = 'black')
# plt.plot(t, (peak3[:, 1] - bg[:, 1])/ ((exp[:, 1] - bg[:, 1]).max()), label = 'peak3',color = 'purple', linewidth = 2)
plt.plot(
    t,
    y(new_predict3[0][0], new_predict3[1][0], new_predict3[2][0], x),
    label="predict thrid peak",
    c="blue",
)
plt.title("third peak")
plt.legend()


# In[36]:


plt.plot(
    test_result3[0].reshape(
        401,
    )
    - y(new_predict3[0][0], new_predict3[1][0], new_predict3[2][0], x),
    c="black",
)
plt.title("total peak without thrid right peak")
plt.ylim(0, 1)


# In[37]:


plt.xlim(0, 400)
plt.ylim(-0.05, 1.1)
plt.plot(
    t, y(new_predict[0][0], new_predict[1][0], new_predict[2][0], x), c="blue"
)
plt.plot(
    t,
    y(new_predict2[0][0], new_predict2[1][0], new_predict2[2][0], x),
    c="blue",
)
plt.plot(
    t,
    y(new_predict3[0][0], new_predict3[1][0], new_predict3[2][0], x),
    label="predict",
    c="blue",
)


plt.plot(
    t,
    (peak1[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    color="purple",
    linewidth=2,
)
plt.plot(
    t,
    (peak2[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    color="purple",
    linewidth=2,
)
plt.plot(
    t,
    (peak3[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    label="real",
    color="purple",
    linewidth=2,
)
plt.title("ito with real, predict peaks", size=20)

# plt.plot(t,y(optima.x[0],optima.x[1],optima.x[2],x), c ='red', label = 'after correction')
# plt.plot(t,y(optima.x[3],optima.x[4],optima.x[5],x), c = 'red')
# plt.plot(t,y(optima.x[6],optima.x[7],optima.x[8],x), c = 'red')

plt.plot(test_result[0], c="black", label="p3ht")
plt.legend()


# In[38]:


import numpy as np
import scipy.optimize as sco


# In[39]:


predict_area = (
    y(new_predict[0][0], new_predict[1][0], new_predict[2][0], x)
    + y(new_predict2[0][0], new_predict2[1][0], new_predict2[2][0], x)
    + y(new_predict3[0][0], new_predict3[1][0], new_predict3[2][0], x)
)

plt.plot(x, test_result[0], label="real")
plt.plot(x, predict_area, label="predict")
plt.legend()

print(
    "test peak area = ",
    sum(
        test_result[0].reshape(
            401,
        )
    ),
)
print("predict peak area = ", sum(predict_area))

# residual_area = (test_result[0].reshape(401,)-predict_area)**2
residual_area = abs(
    test_result[0].reshape(
        401,
    )
    - predict_area
)


# for k in range(401):
#     if residual_area[k] < 0.01: # 너무 noise한 부분도 고려하니까 없애주자
#         residual_area[k] = 0

print(residual_area.shape)
# print(residual_area)
print("residual area = ", sum(residual_area))


plt.plot(x, residual_area, label="residual area")
plt.legend()

# plt.axvline(x = new_predict[0][0], c = 'black', linewidth = 1, linestyle = '--')
# plt.axvline(x = new_predict2[0][0], c = 'black', linewidth = 1, linestyle = '--')
# plt.axvline(x = new_predict3[0][0], c = 'black', linewidth = 1, linestyle = '--')


# In[56]:


test_peak_area = test_result.reshape(
    401,
)
beta = 5.09791537e-01
gamma = 4.41140472e-01
# beta = 0
# gamma = 0
x = np.linspace(0, 15, 401)


def y2(p):
    a_1, b_1, c_1, a_2, b_2, c_2, a_3, b_3, c_3 = p

    total_y = sum(
        abs(
            (
                c_1
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_1) ** 2 / (beta * b_1) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_1) ** 2 / (gamma * b_1) ** 2))
                )
            )
            + (
                c_2
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_2) ** 2 / (beta * b_2) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_2) ** 2 / (gamma * b_2) ** 2))
                )
            )
            + (
                c_3
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_3) ** 2 / (beta * b_3) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_3) ** 2 / (gamma * b_3) ** 2))
                )
            )
            - test_peak_area
        )
    )

    #     total_y  = sum(((c_1 * ((0.7*np.exp(-np.log(2) * (x - a_1)**2 / (beta * b_1)**2)) + (0.3 / (1 + (x -a_1)**2 / (gamma * b_1)**2))))
    #         +(c_2 * ((0.7*np.exp(-np.log(2) * (x - a_2)**2 / (beta * b_2)**2)) + (0.3 / (1 + (x -a_2)**2 / (gamma * b_2)**2))))
    #         +(c_3 * ((0.7*np.exp(-np.log(2) * (x - a_3)**2 / (beta * b_3)**2)) + (0.3 / (1 + (x -a_3)**2 / (gamma * b_3)**2))))
    #         -test_peak_area )**2)

    return total_y


optima = sco.basinhopping(
    y2,
    (
        new_predict[0][0],
        new_predict[1][0],
        new_predict[2][0],
        new_predict2[0][0],
        new_predict2[1][0],
        new_predict2[2][0],
        new_predict3[0][0],
        new_predict3[1][0],
        new_predict3[2][0],
    ),
    niter=100,
    T=1,
    stepsize=0.5,
)
optima


# In[57]:


plt.plot(
    t,
    y(optima.x[0], optima.x[1], optima.x[2], x),
    c="red",
    linewidth=2,
    label="after correction",
)
plt.plot(t, y(optima.x[3], optima.x[4], optima.x[5], x), c="red", linewidth=2)
plt.plot(t, y(optima.x[6], optima.x[7], optima.x[8], x), c="red", linewidth=2)

plt.plot(
    t,
    y(new_predict[0][0], new_predict[1][0], new_predict[2][0], x),
    c="green",
    label="predict",
)
plt.plot(
    t,
    y(new_predict2[0][0], new_predict2[1][0], new_predict2[2][0], x),
    c="green",
)
plt.plot(
    t,
    y(new_predict3[0][0], new_predict3[1][0], new_predict3[2][0], x),
    c="green",
)

# plt.plot(x,y(test_peak_param[i][0][0],test_peak_param[i][0][1],test_peak_param[i][0][2],x), c = 'blue', label = 'real')
# plt.plot(x,y(test_peak_param[i][1][0],test_peak_param[i][1][1],test_peak_param[i][1][2],x), c = 'blue')
# plt.plot(x,y(test_peak_param[i][2][0],test_peak_param[i][2][1],test_peak_param[i][2][2],x), c = 'blue')

plt.plot(
    t,
    (peak1[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    color="purple",
    linewidth=2,
)
plt.plot(
    t,
    (peak2[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    color="purple",
    linewidth=2,
)
plt.plot(
    t,
    (peak3[:, 1] - bg[:, 1]) / ((exp[:, 1] - bg[:, 1]).max()),
    label="real",
    color="purple",
    linewidth=2,
)

plt.xlim(0, 400)
plt.ylim(-0.05, 1.1)
plt.legend()


# In[ ]:


# mae vs mse bar chart


# In[ ]:


# resnet
# 0.10816401457979095
# 0.030619180051451405
# 0.020286300825483432

# se-resnet
# 0.08335333617545047
# 0.030164464011837484
# 0.020591724598516287

# 0.08119587027995552
# 0.02853893844762173
# 0.019406292207859128


# In[ ]:


# 0.1980241036026108
# 0.0434090221804756
# 0.03321981394717362

# lenet
# noise = 0.01 ,batch_size = 200, epoch = 100, lr = 0.5
# 기준 area

# 0.2876684894293567
# 0.033030751748916974
# 0.023151721246088306

# zfnet
# noise = 0.01 ,batch_size = 200, epoch = 100, lr = 0.5
# 기준 area

# 0.11264473291117566
# 0.021803617173060594
# 0.01740119580528536

# ---------------------------------------- p3ht그래프 시각화 수정
# zfnet
# noise = 0.01 ,batch_size = 500, epoch = 50, lr = 0.1
# 0.11684046318704204
# 0.02365474199326388
# 0.017884804965016435
# 기준 area

# zfnet
# noise = 0.05 ,batch_size = 500, epoch = 50, lr = 0.1
# 기준 area

# 0.14462273796028452
# 0.031952290796258394
# 0.02227899902482363

# zfnet
# noise = 0.05 ,batch_size = 500, epoch = 50, lr = 0.1
# 기준 center

# 0.0403674395524352
# 0.02615626201945091
# 0.02677103315473494
# 파라미더 1,580,000

# resnet
# noise = 0.05 ,batch_size = 500, epoch = 50, lr = 0.1
# 기준 area

# 0.11857149868883526
# 0.03263180739217448
# 0.022952831307554454

# -----------------------------------------new data
# vggnet
# noise = 0.05 ,batch_size = 500, epoch = 50, patience = 5, lr = 0.1 feature 32
# 기준 area
# 0.11484013720310739
# 0.03064794644125652
# 0.02168992991763753

# vggnet
# noise = 0.05 ,batch_size = 500, epoch = 50, patience = 5, lr = 0.1 feature 64
# 기준 area
# 0.17572729285820568
# 0.029032754705661946
# 0.020763448977790314

# vggnet
# noise = 0.05 ,batch_size = 500, epoch = 50, patience = 10, lr = 0.1 feature 64
# 기준 area
# 0.17314537565227558
# 0.028766554175357072
# 0.021212849061068805

# resnet
# noise = 0.05 ,batch_size = 500, epoch = 50, patience = 5, lr = 0.1
# ,projection shortcut 1
# 기준 area

# 0.219934999018171
# 0.041640542136089184
# 0.028254922584841143

# resnet 18layer
# noise = 0.05 ,batch_size = 500, epoch = 50, patience = 5, lr = 0.1

# 0.11579423334714921
# 0.03162808413659801
# 0.021646715531842685

# resnet 18layer
# leakyrelu(0.01), kernel_init = he_normal,384
# noise = 0.05 ,batch_size = 500, epoch = 50, patience = 5, lr = 0.1
# not pre-activation

# 0.11236779294782767
# 0.030167100940907545
# 0.02108751209296305

# noise =0.05, batch_size = 512, epoch = 50, patience = 10, lr= 0.1, feature map = 32
# vgg(plain)net
# 0.11908134603087908
# 0.029694876976764893
# 0.02220266332971771

# plainet + projection shortcut 1layer
# noise =0.05, batch_size = 512, epoch = 50, patience = 10, lr= 0.1, feature map = 32
# 0.14454205265157546
# 0.035656453286206834
# 0.02518174399651547

# plainet+ projection shortcut 2layer
# noise =0.05, batch_size = 512, epoch = 50, patience = 10, lr= 0.1, feature map = 32
# 실수의 maxpooling and strides 2
# 최종 output kernelsize 7

# 0.13743766259137466
# 0.034960850331634034
# 0.024586934629632767

# resnet original
# noise =0.05, batch_size = 512, epoch = 50, patience = 10, lr= 0.1, feature map = 32
# layers 19
# pre activation

# 0.10910127349335477
# 0.029502006677230425
# 0.02118154023550757


# In[1377]:


def truncate_peak1(prediction, best):
    # predict
    save_param = []
    #     save_param.append(prediction[0][i][0])
    #     save_param.append(prediction[1][i][0])
    #     save_param.append(prediction[2][i][0])

    save_param.append(
        [prediction[0][i][0], prediction[1][i][0], prediction[2][i][0]]
    )

    return save_param


def truncate_peak2(prediction, best):
    # predict
    save_param = []
    #     save_param.append(prediction[0][i][0])
    #     save_param.append(prediction[1][i][0])
    #     save_param.append(prediction[2][i][0])

    save_param.append(
        [prediction[0][i][0], prediction[1][i][0], prediction[2][i][0]]
    )

    # truncate
    first_correction = test_peak[i].reshape(
        401,
    ) - y(
        prediction[0][i][
            0
        ],  # former prediction_se_res_densenet_finish1, changed due to missing value
        prediction[1][i][0],
        prediction[2][i][0],
        x,
    )

    # correction
    for k in range(first_correction.shape[0]):  # 401번 반복
        if first_correction[k] < 0.0:
            first_correction[k] = 0

    # predict
    second_process = best.predict(first_correction.reshape(1, 401, 1))
    #     save_param.append(second_process[0][0][0])
    #     save_param.append(second_process[1][0][0])
    #     save_param.append(second_process[2][0][0])

    save_param.append(
        [
            second_process[0][0][0],
            second_process[1][0][0],
            second_process[2][0][0],
        ]
    )

    # truncate
    second_correction = first_correction - y(
        second_process[0][0], second_process[1][0], second_process[2][0], x
    )

    # correction
    for k in range(second_correction.shape[0]):  # 401번 반복
        if second_correction[k] < 0.0:
            second_correction[k] = 0

    return save_param


def truncate_peak3(prediction, best):
    # predict
    save_param = []
    #     save_param.append(prediction[0][i][0])
    #     save_param.append(prediction[1][i][0])
    #     save_param.append(prediction[2][i][0])

    save_param.append(
        [prediction[0][i][0], prediction[1][i][0], prediction[2][i][0]]
    )

    # truncate
    first_correction = test_peak[i].reshape(
        401,
    ) - y(prediction[0][i][0], prediction[1][i][0], prediction[2][i][0], x)

    # correction
    for k in range(first_correction.shape[0]):  # 401번 반복
        if first_correction[k] < 0.0:
            first_correction[k] = 0

    # predict
    second_process = best.predict(first_correction.reshape(1, 401, 1))
    #     save_param.append(second_process[0][0][0])
    #     save_param.append(second_process[1][0][0])
    #     save_param.append(second_process[2][0][0])

    save_param.append(
        [
            second_process[0][0][0],
            second_process[1][0][0],
            second_process[2][0][0],
        ]
    )

    # truncate
    second_correction = first_correction - y(
        second_process[0][0], second_process[1][0], second_process[2][0], x
    )

    # correction
    for k in range(second_correction.shape[0]):  # 401번 반복
        if second_correction[k] < 0.0:
            second_correction[k] = 0

    # predict
    third_process = best.predict(second_correction.reshape(1, 401, 1))
    #     save_param.append(third_process[0][0][0])
    #     save_param.append(third_process[1][0][0])
    #     save_param.append(third_process[2][0][0])

    save_param.append(
        [
            third_process[0][0][0],
            third_process[1][0][0],
            third_process[2][0][0],
        ]
    )

    # truncate
    third_correction = second_correction - y(
        third_process[0][0], third_process[1][0], third_process[2][0], x
    )

    # correction
    for k in range(third_correction.shape[0]):  # 401번 반복
        if third_correction[k] < 0.0:
            third_correction[k] = 0

    return save_param


def truncate_peak4(prediction, best):
    # predict
    save_param = []
    #     save_param.append(prediction[0][i][0])
    #     save_param.append(prediction[1][i][0])
    #     save_param.append(prediction[2][i][0])
    save_param.append(
        [prediction[0][i][0], prediction[1][i][0], prediction[2][i][0]]
    )

    # truncate
    first_correction = test_peak[i].reshape(
        401,
    ) - y(prediction[0][i][0], prediction[1][i][0], prediction[2][i][0], x)

    # correction
    for k in range(first_correction.shape[0]):  # 401번 반복
        if first_correction[k] < 0.0:
            first_correction[k] = 0

    # predict
    second_process = best.predict(first_correction.reshape(1, 401, 1))
    #     save_param.append(second_process[0][0][0])
    #     save_param.append(second_process[1][0][0])
    #     save_param.append(second_process[2][0][0])

    save_param.append(
        [
            second_process[0][0][0],
            second_process[1][0][0],
            second_process[2][0][0],
        ]
    )

    # truncate
    second_correction = first_correction - y(
        second_process[0][0], second_process[1][0], second_process[2][0], x
    )

    # correction
    for k in range(second_correction.shape[0]):  # 401번 반복
        if second_correction[k] < 0.0:
            second_correction[k] = 0

    # predict
    third_process = best.predict(second_correction.reshape(1, 401, 1))
    #     save_param.append(third_process[0][0][0])
    #     save_param.append(third_process[1][0][0])
    #     save_param.append(third_process[2][0][0])

    save_param.append(
        [
            third_process[0][0][0],
            third_process[1][0][0],
            third_process[2][0][0],
        ]
    )

    # truncate
    third_correction = second_correction - y(
        third_process[0][0], third_process[1][0], third_process[2][0], x
    )

    # correction
    for k in range(third_correction.shape[0]):
        if third_correction[k] < 0.0:
            third_correction[k] = 0

    # predict
    four_process = best.predict(third_correction.reshape(1, 401, 1))
    #     save_param.append(four_process[0][0][0])
    #     save_param.append(four_process[1][0][0])
    #     save_param.append(four_process[2][0][0])

    save_param.append(
        [four_process[0][0][0], four_process[1][0][0], four_process[2][0][0]]
    )

    # truncate
    four_correction = third_correction - y(
        four_process[0][0], four_process[1][0], four_process[2][0], x
    )

    # correction
    for k in range(four_correction.shape[0]):
        if four_correction[k] < 0.0:
            four_correction[k] = 0

    return save_param


def truncate_peak5(prediction, best):
    # predict
    save_param = []
    #     save_param.append(prediction[0][i][0])
    #     save_param.append(prediction[1][i][0])
    #     save_param.append(prediction[2][i][0])

    save_param.append(
        [prediction[0][i][0], prediction[1][i][0], prediction[2][i][0]]
    )

    # truncate
    first_correction = test_peak[i].reshape(
        401,
    ) - y(prediction[0][i][0], prediction[1][i][0], prediction[2][i][0], x)

    # correction
    for k in range(first_correction.shape[0]):  # 401번 반복
        if first_correction[k] < 0.0:
            first_correction[k] = 0

    # predict
    second_process = best.predict(first_correction.reshape(1, 401, 1))
    #     save_param.append(second_process[0][0][0])
    #     save_param.append(second_process[1][0][0])
    #     save_param.append(second_process[2][0][0])

    save_param.append(
        [
            second_process[0][0][0],
            second_process[1][0][0],
            second_process[2][0][0],
        ]
    )

    # truncate
    second_correction = first_correction - y(
        second_process[0][0], second_process[1][0], second_process[2][0], x
    )

    # correction
    for k in range(second_correction.shape[0]):  # 401번 반복
        if second_correction[k] < 0.0:
            second_correction[k] = 0

    # predict
    third_process = best.predict(second_correction.reshape(1, 401, 1))
    #     save_param.append(third_process[0][0][0])
    #     save_param.append(third_process[1][0][0])
    #     save_param.append(third_process[2][0][0])

    save_param.append(
        [
            third_process[0][0][0],
            third_process[1][0][0],
            third_process[2][0][0],
        ]
    )

    # truncate
    third_correction = second_correction - y(
        third_process[0][0], third_process[1][0], third_process[2][0], x
    )

    # correction
    for k in range(third_correction.shape[0]):
        if third_correction[k] < 0.0:
            third_correction[k] = 0

    # predict
    four_process = best.predict(third_correction.reshape(1, 401, 1))
    #     save_param.append(four_process[0][0][0])
    #     save_param.append(four_process[1][0][0])
    #     save_param.append(four_process[2][0][0])

    save_param.append(
        [four_process[0][0][0], four_process[1][0][0], four_process[2][0][0]]
    )

    # truncate
    four_correction = third_correction - y(
        four_process[0][0], four_process[1][0], four_process[2][0], x
    )

    # correction
    for k in range(four_correction.shape[0]):
        if four_correction[k] < 0.0:
            four_correction[k] = 0

    # predict
    five_process = best.predict(four_correction.reshape(1, 401, 1))
    #     save_param.append(five_process[0][0][0])
    #     save_param.append(five_process[1][0][0])
    #     save_param.append(five_process[2][0][0])

    save_param.append(
        [five_process[0][0][0], five_process[1][0][0], five_process[2][0][0]]
    )

    # truncate
    five_correction = four_correction - y(
        five_process[0][0], five_process[1][0], five_process[2][0], x
    )

    # correction
    for k in range(five_correction.shape[0]):
        if five_correction[k] < 0.0:
            five_correction[k] = 0

    return save_param


# In[ ]:


########################## before basinhopping my model#######################


# In[2240]:
# Uncomment the whole section , due to missing understanding of predictio_se_res_densenet

predict_machinelearning_mymodel = []

for i in range(15):  # range(700)
    print(i)
    if round(test_peak_number[i]) == 1:
        save_param = truncate_peak1(
            prediction_se_resnet,  # _densenet_finish1,
            best_model_se_resnet,  # _densenet_finish1,
        )
        predict_machinelearning_mymodel.append(save_param)
        print("truncate_peak1 done")

    elif round(test_peak_number[i]) == 2:
        save_param = truncate_peak2(
            prediction_se_resnet,  # _densenet_finish1,
            best_model_se_resnet,  # _densenet_finish1,
        )
        predict_machinelearning_mymodel.append(save_param)
        print("truncate_peak2 done")

    elif round(test_peak_number[i]) == 3:
        save_param = truncate_peak3(
            prediction_se_resnet,  # _densenet_finish1,
            best_model_se_resnet,  # _densenet_finish1,
        )
        predict_machinelearning_mymodel.append(save_param)
        print("truncate_peak3 done")

    elif round(test_peak_number[i]) == 4:
        save_param = truncate_peak4(
            prediction_se_resnet,  # _densenet_finish1,
            best_model_se_resnet,  # _densenet_finish1,
        )
        predict_machinelearning_mymodel.append(save_param)
        print("truncate_peak4 done")

    elif round(test_peak_number[i]) == 5:
        save_param = truncate_peak5(
            prediction_se_resnet,  # _densenet_finish1,
            best_model_se_resnet,  # _densenet_finish1,
        )
        predict_machinelearning_mymodel.append(save_param)
        print("truncate_peak5 done")


# In[2190]:


test_peak_param2 = test_peak_param.copy()


# In[2191]:


# test data 몇개를 할건지 - "How many test data to do"
# peak number 빼주기 -  "Subtract peak number"

for i in range(15):  # range(10000)
    for k in range(len(test_peak_param2[i])):
        if len(test_peak_param2[i][k]) != 3:  # peak1
            test_peak_param2[i][k].pop()

        # test_data 10000개까지 삭제시킴 - "Deleted up to 10,000 test_data."


# In[2231]:


# test_peak_param2[9995:10005]
aa = np.array(predict_machinelearning_mymodel[0]).sort()
predict_machinelearning_mymodel[0]


# In[2236]:


# predict_machinelearning_lenet[0] not defined variable
# predict_machinelearning_mymodel[690]


# In[2241]:
### Error : ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

# center 값 기준으로 sort하기

# for i in range(15): # range(700)
#     #     test_peak_param2[i].sort()
#     predict_machinelearning_mymodel[i].sort()


# In[2242]:


machinelearning_mymodel_center_error = []
machinelearning_mymodel_width_error = []
machinelearning_mymodel_amp_error = []

for i in range(15):  # range(700)
    for k in range(len(predict_machinelearning_mymodel[i])):
        predict_machinelearning_mymodel[i][k] = np.array(
            predict_machinelearning_mymodel[i][k]
        )
        test_peak_param2[i][k] = np.array(test_peak_param2[i][k])
    error = abs(
        np.array(test_peak_param2[i])
        - np.array(predict_machinelearning_mymodel[i])
    )
    #     print(error)
    for j in range(len(predict_machinelearning_mymodel[i])):
        machinelearning_mymodel_center_error.append(error[j][0])
        machinelearning_mymodel_width_error.append(error[j][1])
        machinelearning_mymodel_amp_error.append(error[j][2])


# In[2243]:


print(np.array(machinelearning_mymodel_center_error).sum() / 10000)
print(np.array(machinelearning_mymodel_width_error).sum() / 10000)
print(np.array(machinelearning_mymodel_amp_error).sum() / 10000)


# 0.27962325792417414
# 0.144630430967448
# 0.1156989668347496


# In[ ]:


############################ before basinhopping lenet ###########################3


# In[2232]:


predict_machinelearning_lenet = []

for i in range(15):  # range(700)
    print(i)
    if round(test_peak_number[i]) == 1:
        save_param = truncate_peak1(prediction_lenet, best_model_lenet)
        predict_machinelearning_lenet.append(save_param)
        print("truncate_peak1 done")

    elif round(test_peak_number[i]) == 2:
        save_param = truncate_peak2(prediction_lenet, best_model_lenet)
        predict_machinelearning_lenet.append(save_param)
        print("truncate_peak2 done")

    elif round(test_peak_number[i]) == 3:
        save_param = truncate_peak3(prediction_lenet, best_model_lenet)
        predict_machinelearning_lenet.append(save_param)
        print("truncate_peak3 done")

    elif round(test_peak_number[i]) == 4:
        save_param = truncate_peak4(prediction_lenet, best_model_lenet)
        predict_machinelearning_lenet.append(save_param)
        print("truncate_peak4 done")

    elif round(test_peak_number[i]) == 5:
        save_param = truncate_peak5(prediction_lenet, best_model_lenet)
        predict_machinelearning_lenet.append(save_param)
        print("truncate_peak5 done")


# In[2233]:


# center 값 기준으로 sort하기

for i in range(15):  # range(700)
    predict_machinelearning_lenet[i].sort()


# In[2238]:


machinelearning_lenet_center_error = []
machinelearning_lenet_width_error = []
machinelearning_lenet_amp_error = []

for i in range(15):  # range(700)
    for k in range(len(predict_machinelearning_lenet[i])):
        predict_machinelearning_lenet[i][k] = np.array(
            predict_machinelearning_lenet[i][k]
        )
        test_peak_param2[i][k] = np.array(test_peak_param2[i][k])
    error = abs(
        np.array(test_peak_param2[i])
        - np.array(predict_machinelearning_lenet[i])
    )
    #     print(error)
    for j in range(len(predict_machinelearning_lenet[i])):
        machinelearning_lenet_center_error.append(error[j][0])
        machinelearning_lenet_width_error.append(error[j][1])
        machinelearning_lenet_amp_error.append(error[j][2])


# In[2239]:


print(np.array(machinelearning_lenet_center_error).sum() / 700)
print(np.array(machinelearning_lenet_width_error).sum() / 700)
print(np.array(machinelearning_lenet_amp_error).sum() / 700)


# In[1733]:


for i in range(10):
    plt.figure(figsize=(5, 5))
    for k in range(len(test_peak_param[i])):
        plt.plot(
            x,
            y(
                test_peak_param[i][k][0],
                test_peak_param[i][k][1],
                test_peak_param[i][k][2],
                x,
            ),
            c="black",
            linewidth=3,
        )
        plt.plot(
            x,
            y(
                predict_machinelearning_mymodel[i][k][0],
                predict_machinelearning_mymodel[i][k][1],
                predict_machinelearning_mymodel[i][k][2],
                x,
            ),
            linewidth=2,
            c="royalblue",
        )
        plt.plot(
            x,
            y(
                predict_machinelearning_lenet[i][k][0],
                predict_machinelearning_lenet[i][k][1],
                predict_machinelearning_lenet[i][k][2],
                x,
            ),
            c="orange",
        )


# In[ ]:


################# after basinhopping my model#####################3


# In[1734]:


import numpy as np
import scipy.optimize as sco


# In[1752]:


def y2_peak1(p):
    a_1, b_1, c_1 = p

    total_y = sum(
        abs(
            (
                c_1
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_1) ** 2 / (beta * b_1) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_1) ** 2 / (gamma * b_1) ** 2))
                )
            )
            - test_peak_area
        )
    )

    return total_y


def y2_peak2(p):
    a_1, b_1, c_1, a_2, b_2, c_2 = p

    total_y = sum(
        abs(
            (
                c_1
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_1) ** 2 / (beta * b_1) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_1) ** 2 / (gamma * b_1) ** 2))
                )
            )
            + (
                c_2
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_2) ** 2 / (beta * b_2) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_2) ** 2 / (gamma * b_2) ** 2))
                )
            )
            - test_peak_area
        )
    )

    return total_y


def y2_peak3(p):
    a_1, b_1, c_1, a_2, b_2, c_2, a_3, b_3, c_3 = p

    total_y = sum(
        abs(
            (
                c_1
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_1) ** 2 / (beta * b_1) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_1) ** 2 / (gamma * b_1) ** 2))
                )
            )
            + (
                c_2
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_2) ** 2 / (beta * b_2) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_2) ** 2 / (gamma * b_2) ** 2))
                )
            )
            + (
                c_3
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_3) ** 2 / (beta * b_3) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_3) ** 2 / (gamma * b_3) ** 2))
                )
            )
            - test_peak_area
        )
    )

    return total_y


def y2_peak4(p):
    a_1, b_1, c_1, a_2, b_2, c_2, a_3, b_3, c_3, a_4, b_4, c_4 = p

    total_y = sum(
        abs(
            (
                c_1
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_1) ** 2 / (beta * b_1) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_1) ** 2 / (gamma * b_1) ** 2))
                )
            )
            + (
                c_2
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_2) ** 2 / (beta * b_2) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_2) ** 2 / (gamma * b_2) ** 2))
                )
            )
            + (
                c_3
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_3) ** 2 / (beta * b_3) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_3) ** 2 / (gamma * b_3) ** 2))
                )
            )
            + (
                c_4
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_4) ** 2 / (beta * b_4) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_4) ** 2 / (gamma * b_4) ** 2))
                )
            )
            - test_peak_area
        )
    )

    return total_y


def y2_peak5(p):
    (
        a_1,
        b_1,
        c_1,
        a_2,
        b_2,
        c_2,
        a_3,
        b_3,
        c_3,
        a_4,
        b_4,
        c_4,
        a_5,
        b_5,
        c_5,
    ) = p

    total_y = sum(
        abs(
            (
                c_1
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_1) ** 2 / (beta * b_1) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_1) ** 2 / (gamma * b_1) ** 2))
                )
            )
            + (
                c_2
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_2) ** 2 / (beta * b_2) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_2) ** 2 / (gamma * b_2) ** 2))
                )
            )
            + (
                c_3
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_3) ** 2 / (beta * b_3) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_3) ** 2 / (gamma * b_3) ** 2))
                )
            )
            + (
                c_4
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_4) ** 2 / (beta * b_4) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_4) ** 2 / (gamma * b_4) ** 2))
                )
            )
            + (
                c_5
                * (
                    (
                        0.7
                        * np.exp(
                            -np.log(2) * (x - a_5) ** 2 / (beta * b_5) ** 2
                        )
                    )
                    + (0.3 / (1 + (x - a_5) ** 2 / (gamma * b_5) ** 2))
                )
            )
            - test_peak_area
        )
    )

    return total_y


# In[2076]:


check_number = 15  # formerly 700


# In[2077]:


not_arrangement_after_basin_mymodel_predict = []

for i in range(check_number):
    test_peak_area = test_peak[i].reshape(
        401,
    )
    beta = 5.09791537e-01
    gamma = 4.41140472e-01
    x = np.linspace(0, 15, 401)
    print(i)

    if len(predict_machinelearning_mymodel[i]) == 1:
        optima = sco.basinhopping(
            y2_peak1,
            (
                predict_machinelearning_mymodel[i][0][0],
                predict_machinelearning_mymodel[i][0][1],
                predict_machinelearning_mymodel[i][0][2],
            ),
            niter=100,
            T=1,
            stepsize=0.5,
        )
        not_arrangement_after_basin_mymodel_predict.append(abs(optima.x))

    elif len(predict_machinelearning_mymodel[i]) == 2:
        optima = sco.basinhopping(
            y2_peak2,
            (
                predict_machinelearning_mymodel[i][0][0],
                predict_machinelearning_mymodel[i][0][1],
                predict_machinelearning_mymodel[i][0][2],
                predict_machinelearning_mymodel[i][1][0],
                predict_machinelearning_mymodel[i][1][1],
                predict_machinelearning_mymodel[i][1][2],
            ),
            niter=100,
            T=1,
            stepsize=0.5,
        )
        not_arrangement_after_basin_mymodel_predict.append(abs(optima.x))

    elif len(predict_machinelearning_mymodel[i]) == 3:
        optima = sco.basinhopping(
            y2_peak3,
            (
                predict_machinelearning_mymodel[i][0][0],
                predict_machinelearning_mymodel[i][0][1],
                predict_machinelearning_mymodel[i][0][2],
                predict_machinelearning_mymodel[i][1][0],
                predict_machinelearning_mymodel[i][1][1],
                predict_machinelearning_mymodel[i][1][2],
                predict_machinelearning_mymodel[i][2][0],
                predict_machinelearning_mymodel[i][2][1],
                predict_machinelearning_mymodel[i][2][2],
            ),
            niter=100,
            T=1,
            stepsize=0.5,
        )
        not_arrangement_after_basin_mymodel_predict.append(abs(optima.x))

    elif len(predict_machinelearning_mymodel[i]) == 4:
        optima = sco.basinhopping(
            y2_peak4,
            (
                predict_machinelearning_mymodel[i][0][0],
                predict_machinelearning_mymodel[i][0][1],
                predict_machinelearning_mymodel[i][0][2],
                predict_machinelearning_mymodel[i][1][0],
                predict_machinelearning_mymodel[i][1][1],
                predict_machinelearning_mymodel[i][1][2],
                predict_machinelearning_mymodel[i][2][0],
                predict_machinelearning_mymodel[i][2][1],
                predict_machinelearning_mymodel[i][2][2],
                predict_machinelearning_mymodel[i][3][0],
                predict_machinelearning_mymodel[i][3][1],
                predict_machinelearning_mymodel[i][3][2],
            ),
            niter=100,
            T=1,
            stepsize=0.5,
        )
        not_arrangement_after_basin_mymodel_predict.append(abs(optima.x))

    elif len(predict_machinelearning_mymodel[i]) == 5:
        optima = sco.basinhopping(
            y2_peak5,
            (
                predict_machinelearning_mymodel[i][0][0],
                predict_machinelearning_mymodel[i][0][1],
                predict_machinelearning_mymodel[i][0][2],
                predict_machinelearning_mymodel[i][1][0],
                predict_machinelearning_mymodel[i][1][1],
                predict_machinelearning_mymodel[i][1][2],
                predict_machinelearning_mymodel[i][2][0],
                predict_machinelearning_mymodel[i][2][1],
                predict_machinelearning_mymodel[i][2][2],
                predict_machinelearning_mymodel[i][3][0],
                predict_machinelearning_mymodel[i][3][1],
                predict_machinelearning_mymodel[i][3][2],
                predict_machinelearning_mymodel[i][4][0],
                predict_machinelearning_mymodel[i][4][1],
                predict_machinelearning_mymodel[i][4][2],
            ),
            niter=100,
            T=1,
            stepsize=0.5,
        )
        not_arrangement_after_basin_mymodel_predict.append(abs(optima.x))


# In[2129]:


# center width amp 서로 뺄수있게 shape 맞춰주기

after_basin_mymodel_predict = []

for i in range(check_number):
    print(i)
    append_list = []

    if len(not_arrangement_after_basin_mymodel_predict[i]) == 3:
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][:3])
        )
        after_basin_mymodel_predict.append(append_list)

    elif len(not_arrangement_after_basin_mymodel_predict[i]) == 6:
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][:3])
        )
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][3:6])
        )
        after_basin_mymodel_predict.append(append_list)

    elif len(not_arrangement_after_basin_mymodel_predict[i]) == 9:
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][:3])
        )
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][3:6])
        )
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][6:9])
        )
        after_basin_mymodel_predict.append(append_list)

    elif len(not_arrangement_after_basin_mymodel_predict[i]) == 12:
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][:3])
        )
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][3:6])
        )
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][6:9])
        )
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][9:12])
        )
        after_basin_mymodel_predict.append(append_list)

    elif len(not_arrangement_after_basin_mymodel_predict[i]) == 15:
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][:3])
        )
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][3:6])
        )
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][6:9])
        )
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][9:12])
        )
        append_list.append(
            list(not_arrangement_after_basin_mymodel_predict[i][12:15])
        )
        after_basin_mymodel_predict.append(append_list)


# In[2130]:


# 재배열 -"Rearranging"

for i in range(check_number):
    after_basin_mymodel_predict[i].sort()


# In[2131]:


# for tt in range(10,12):
tt = 10
test_peak_param2[tt]
after_basin_mymodel_predict[tt]


# In[2134]:


basin_mymodel_center_error = []
basin_mymodel_width_error = []
basin_mymodel_amp_error = []

for i in range(check_number):
    for k in range(len(after_basin_mymodel_predict[i])):
        after_basin_mymodel_predict[i][k] = np.array(
            after_basin_mymodel_predict[i][k]
        )
        test_peak_param2[i][k] = np.array(test_peak_param2[i][k])
    error = abs(
        np.array(test_peak_param2[i])
        - np.array(after_basin_mymodel_predict[i])
    )

    for j in range(len(after_basin_mymodel_predict[i])):
        basin_mymodel_center_error.append(error[j][0])
        basin_mymodel_width_error.append(error[j][1])
        basin_mymodel_amp_error.append(error[j][2])


# In[2135]:


print(np.array(basin_mymodel_center_error).sum() / check_number)
print(np.array(basin_mymodel_width_error).sum() / check_number)
print(np.array(basin_mymodel_amp_error).sum() / check_number)


# In[2184]:


after_basin_mymodel_predict[0]
# test_peak_param2[63]


# In[2247]:


for i in range(0, 15):  # range(100, 200):
    plt.figure(figsize=(10, 5))

    for k in range(len(after_basin_mymodel_predict[i])):
        plt.plot(
            x,
            y(
                test_peak_param2[i][k][0],
                test_peak_param2[i][k][1],
                test_peak_param2[i][k][2],
                x,
            ),
            c="black",
            linewidth=2,
        )
        plt.plot(
            x,
            y(
                after_basin_mymodel_predict[i][k][0],
                after_basin_mymodel_predict[i][k][1],
                after_basin_mymodel_predict[i][k][2],
                x,
            ),
            c="red",
            marker="^",
            linewidth=1,
        )
        plt.plot(
            x,
            y(
                predict_machinelearning_mymodel[i][k][0],
                predict_machinelearning_mymodel[i][k][1],
                predict_machinelearning_mymodel[i][k][2],
                x,
            ),
            c="blue",
            linewidth=2,
        )

        plt.title(i)


# In[ ]:


# In[ ]:


################### after basinhopping lenet ######################


# In[2139]:


not_arrangement_after_basin_lenet_predict = []

for i in range(check_number):
    test_peak_area = test_peak[i].reshape(
        401,
    )
    beta = 5.09791537e-01
    gamma = 4.41140472e-01
    x = np.linspace(0, 15, 401)
    print(i)

    if len(predict_machinelearning_lenet[i]) == 1:
        optima = sco.basinhopping(
            y2_peak1,
            (
                predict_machinelearning_lenet[i][0][0],
                predict_machinelearning_lenet[i][0][1],
                predict_machinelearning_lenet[i][0][2],
            ),
            niter=100,
            T=1,
            stepsize=0.5,
        )
        not_arrangement_after_basin_lenet_predict.append(abs(optima.x))

    elif len(predict_machinelearning_lenet[i]) == 2:
        optima = sco.basinhopping(
            y2_peak2,
            (
                predict_machinelearning_lenet[i][0][0],
                predict_machinelearning_lenet[i][0][1],
                predict_machinelearning_lenet[i][0][2],
                predict_machinelearning_lenet[i][1][0],
                predict_machinelearning_lenet[i][1][1],
                predict_machinelearning_lenet[i][1][2],
            ),
            niter=100,
            T=1,
            stepsize=0.5,
        )
        not_arrangement_after_basin_lenet_predict.append(abs(optima.x))

    elif len(predict_machinelearning_lenet[i]) == 3:
        optima = sco.basinhopping(
            y2_peak3,
            (
                predict_machinelearning_lenet[i][0][0],
                predict_machinelearning_lenet[i][0][1],
                predict_machinelearning_lenet[i][0][2],
                predict_machinelearning_lenet[i][1][0],
                predict_machinelearning_lenet[i][1][1],
                predict_machinelearning_lenet[i][1][2],
                predict_machinelearning_lenet[i][2][0],
                predict_machinelearning_lenet[i][2][1],
                predict_machinelearning_lenet[i][2][2],
            ),
            niter=100,
            T=1,
            stepsize=0.5,
        )
        not_arrangement_after_basin_lenet_predict.append(abs(optima.x))

    elif len(predict_machinelearning_lenet[i]) == 4:
        optima = sco.basinhopping(
            y2_peak4,
            (
                predict_machinelearning_lenet[i][0][0],
                predict_machinelearning_lenet[i][0][1],
                predict_machinelearning_lenet[i][0][2],
                predict_machinelearning_lenet[i][1][0],
                predict_machinelearning_lenet[i][1][1],
                predict_machinelearning_lenet[i][1][2],
                predict_machinelearning_lenet[i][2][0],
                predict_machinelearning_lenet[i][2][1],
                predict_machinelearning_lenet[i][2][2],
                predict_machinelearning_lenet[i][3][0],
                predict_machinelearning_lenet[i][3][1],
                predict_machinelearning_lenet[i][3][2],
            ),
            niter=100,
            T=1,
            stepsize=0.5,
        )
        not_arrangement_after_basin_lenet_predict.append(abs(optima.x))

    elif len(predict_machinelearning_lenet[i]) == 5:
        optima = sco.basinhopping(
            y2_peak5,
            (
                predict_machinelearning_lenet[i][0][0],
                predict_machinelearning_lenet[i][0][1],
                predict_machinelearning_lenet[i][0][2],
                predict_machinelearning_lenet[i][1][0],
                predict_machinelearning_lenet[i][1][1],
                predict_machinelearning_lenet[i][1][2],
                predict_machinelearning_lenet[i][2][0],
                predict_machinelearning_lenet[i][2][1],
                predict_machinelearning_lenet[i][2][2],
                predict_machinelearning_lenet[i][3][0],
                predict_machinelearning_lenet[i][3][1],
                predict_machinelearning_lenet[i][3][2],
                predict_machinelearning_lenet[i][4][0],
                predict_machinelearning_lenet[i][4][1],
                predict_machinelearning_lenet[i][4][2],
            ),
            niter=100,
            T=1,
            stepsize=0.5,
        )
        not_arrangement_after_basin_lenet_predict.append(abs(optima.x))


# In[2146]:


# center width amp 서로 뺄수있게 shape 맞춰주기 - "Aligning shapes of center, width, and amplitude for subtraction"

after_basin_lenet_predict = []

for i in range(check_number):
    print(i)
    append_list = []

    if len(not_arrangement_after_basin_lenet_predict[i]) == 3:
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][:3])
        )
        after_basin_lenet_predict.append(append_list)

    elif len(not_arrangement_after_basin_lenet_predict[i]) == 6:
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][:3])
        )
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][3:6])
        )
        after_basin_lenet_predict.append(append_list)

    elif len(not_arrangement_after_basin_lenet_predict[i]) == 9:
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][:3])
        )
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][3:6])
        )
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][6:9])
        )
        after_basin_lenet_predict.append(append_list)

    elif len(not_arrangement_after_basin_lenet_predict[i]) == 12:
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][:3])
        )
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][3:6])
        )
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][6:9])
        )
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][9:12])
        )
        after_basin_lenet_predict.append(append_list)

    elif len(not_arrangement_after_basin_lenet_predict[i]) == 15:
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][:3])
        )
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][3:6])
        )
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][6:9])
        )
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][9:12])
        )
        append_list.append(
            list(not_arrangement_after_basin_lenet_predict[i][12:15])
        )
        after_basin_lenet_predict.append(append_list)


# In[2147]:


# 재배열

for i in range(check_number):
    after_basin_lenet_predict[i].sort()


# In[2148]:


basin_lenet_center_error = []
basin_lenet_width_error = []
basin_lenet_amp_error = []

for i in range(check_number):
    for k in range(len(after_basin_lenet_predict[i])):
        after_basin_lenet_predict[i][k] = np.array(
            after_basin_lenet_predict[i][k]
        )
        test_peak_param2[i][k] = np.array(test_peak_param2[i][k])
    error = abs(
        np.array(test_peak_param2[i]) - np.array(after_basin_lenet_predict[i])
    )

    for j in range(len(after_basin_lenet_predict[i])):
        basin_lenet_center_error.append(error[j][0])
        basin_lenet_width_error.append(error[j][1])
        basin_lenet_amp_error.append(error[j][2])


# In[2149]:


print(np.array(basin_lenet_center_error).sum() / check_number)
print(np.array(basin_lenet_width_error).sum() / check_number)
print(np.array(basin_lenet_amp_error).sum() / check_number)


# In[2159]:


after_basin_lenet_predict[0]
# after_baisn_lenet[i]


# In[2162]:


for i in range(15):  # range(100):
    plt.figure(figsize=(10, 5))

    for k in range(len(after_basin_lenet_predict[i])):
        plt.plot(
            x,
            y(
                test_peak_param2[i][k][0],
                test_peak_param2[i][k][1],
                test_peak_param2[i][k][2],
                x,
            ),
            c="black",
            linewidth=5,
        )
        plt.plot(
            x,
            y(
                after_basin_lenet_predict[i][k][0],
                after_basin_lenet_predict[i][k][1],
                after_basin_lenet_predict[i][k][2],
                x,
            ),
            c="red",
            marker="^",
        )
        plt.plot(
            x,
            y(
                predict_machinelearning_lenet[i][k][0],
                predict_machinelearning_lenet[i][k][1],
                predict_machinelearning_lenet[i][k][2],
                x,
            ),
            c="blue",
            linewidth=2,
        )

        plt.title(i)


#         black = real
#         red = after basin
#         blue = before basin


# In[ ]:# In[2167]:


for i in range(63, 64):
    plt.figure(figsize=(10, 5))

    for k in range(0):  # len(after_basin_lenet_predict[i])):
        plt.plot(
            x,
            y(
                test_peak_param2[i][k][0],
                test_peak_param2[i][k][1],
                test_peak_param2[i][k][2],
                x,
            ),
            c="black",
            linewidth=5,
        )
        plt.plot(
            x,
            y(
                after_basin_lenet_predict[i][k][0],
                after_basin_lenet_predict[i][k][1],
                after_basin_lenet_predict[i][k][2],
                x,
            ),
            c="red",
            marker="^",
        )
        plt.plot(
            x,
            y(
                predict_machinelearning_lenet[i][k][0],
                predict_machinelearning_lenet[i][k][1],
                predict_machinelearning_lenet[i][k][2],
                x,
            ),
            c="blue",
            linewidth=2,
        )

        plt.title(i)


# In[ ]:


# In[ ]:
