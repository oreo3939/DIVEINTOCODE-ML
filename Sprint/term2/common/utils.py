#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


def output_size(n_features, filter_length, stride=1, pad=0):
    return int(1 + (n_features + 2 * pad - filter_length) / stride)


def imcol2(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters  
    -----------
    input_data : (データ数,チャンネル,高さ,横幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの横幅
    stride : ストライド
    pad : パディング
    -----------
    Returns
    col : 2次元配列 
     
    """

    N, C, H, W = input_data.shape
    OH = output_size(n_features=H, filter_length=filter_h, stride=stride, pad=pad)
    OW = output_size(n_features=W, filter_length=filter_w, stride=stride, pad=pad)

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, OH, OW))

    for y in range(filter_h):
        y_max = y + stride*OH
        for x in range(filter_w):
            x_max = x + stride*OW
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*OH*OW, -1)
    return col




def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col :　変換するデーター
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h : フィルターの高さ
    filter_w : フィルターの横幅
    stride : ストライド
    pad : パディング
    Returns
    -------
    """
    N, C, H, W = input_shape
    OH = output_size(n_features=H, filter_length=filter_h, stride=stride, pad=pad)
    OW = output_size(n_features=W, filter_length=filter_w, stride=stride, pad=pad)
    col = col.reshape(N, OH, OW, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*OH
        for x in range(filter_w):
            x_max = x + stride*OW
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


# In[3]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'utils.ipynb'])


# In[ ]:




