#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np


from utils import output_size, imcol2, col2im

class Conv2d:
    
    def __init__(self, W, b, stride=1, pad=0):
        """
        W (numpy.ndarray): フィルター（重み）、形状は(FN, C, FH, FW)。
        b (numpy.ndarray): バイアス、形状は(FN)。
        stride (int, optional): ストライド、デフォルトは1。
        pad (int, optional): パディング、デフォルトは0。
        """
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.X = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None
        
    def forward(self,X):
        
        """
        X (numpy.ndarray): 入力。形状は(N, C, H, W)。

        Returns:
            numpy.ndarray: 出力。形状は(N, FN, OH, OW)。
        """
        
        
        FN, C, FH, FW = self.W.shape # FN:フィルター数、C:チャンネル数、FH:フィルターの高さ、FW:幅
        N, C, H, W = X.shape  # N:バッチサイズ、C:チャンネル数、H：入力データの高さ、W:幅
        assert C == C, 'チャンネル数の不一致'
        
        OH = output_size(H,FH,self.stride,self.pad)
        OW = output_size(W,FW,self.stride,self.pad)
        
        # 入力データを展開
        # (N, C, H, W) → (N * OH * OW, C * FH * FW) (1,49)
        col = imcol2(X,FH,FW,self.stride,self.pad)
        
        # フィルターを展開
        # (FN, C, FH, FW) → (C * FH * FW, FN),(49,5)
        col_W = self.W.reshape(FN, -1).T
        
        # (N * OH * OW, C * FH * FW)・(C * FH * FW, FN) → (N * OH * OW, FN) (1,5)
        out =  col @ col_W + self.b
        
         # (N * OH * OW, FN) → (N, OH, OW, FN) → (N, FN, OH, OW) (1,5,1,1)
        out = out.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)

        
        # 保存
        self.X = X
        self.col = col
        self.col_W = col_W

        return out
    
    
    def backward(self, dout):
        """
        dout (numpy.ndarray): 右の層から伝わってくる微分値、形状は(N, FN, OH, OW)。

        Returns
            numpy.ndarray: 微分値（勾配）、形状は(N, C, H, W)。
        """
        FN, C, FH, FW = self.W.shape
        # (N, FN, OH, OW) → (N, OH, OW, FN) → (N * OH * OW, FN)
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)  # (FN)
        self.dW = self.col.T @ dout  #  (C * FH * FW, FN)

         # (C * FH * FW, FN) → (FN, C * FH * FW) → (FN, C, FH, FW)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = dout @ self.col_W.T #  (N * OH * OW, C * FH * FW)
        
         # (N * OH * OW, C * FH * FW) → (N, C, H, W)
        dx = col2im(dcol, self.X.shape, FH, FW, self.stride, self.pad)

        return dx
    
    
class MaxPooling:
        """
        pool_h (int): プーリング領域の高さ
        pool_w (int): プーリング領域の幅
        stride (int, optional): ストライド、デフォルトは1。
        pad (int, optional): パディング、デフォルトは0。
        """ 
        def __init__(self, pool_h, pool_w, stride=1, pad=0):
            self.pool_h = pool_h
            self.pool_w = pool_w
            self.stride = stride
            self.pad = pad
            self.X = None
            self.arg_max=None

        def forward(self, X):
            """
            x (numpy.ndarray): 入力、形状は(N, C, H, W)。

            Returns:
                numpy.ndarray: 出力、形状は(N, C, OH, OW)。
            """

            N, C, H, W = X.shape

            OH = output_size(H, self.pool_h, self.stride, self.pad)
            OW = output_size(W, self.pool_w, self.stride, self.pad)

            # (N, C, H, W) → (N * OH * OW, C * PH * PW)
            col = im2col(X, self.pool_h, self.pool_w, self.stride, self.pad)

            # (N * OH * OW, C * PH * PW) → (N * OH * OW * C, PH * PW)
            col = col.reshape(-1, self.pool_h * self.pool_w)

             # 最大値の位置（インデックス）
            arg_max = np.argmax(col, axis=1)

            # (N * OH * OW * C, PH * PW) → (N * OH * OW * C)
            out = np.max(col, axis=1)

            # (N * OH * OW * C) → (N, OH, OW, C) → (N, C, OH, OW)
            out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)

            self.X = X
            self.arg_max = arg_max
            return out


        def backward(self, dout):
            """ 
            dout (numpy.ndarray): 右の層から伝わってくる微分値、形状は(N, C, OH, OW)。

            Returns:
                numpy.ndarray: 微分値（勾配）、形状は(N, C, H, W)。
            """
             # (N, C, OH, OW) → (N, OH, OW, C)
            dout = dout.transpose(0, 2, 3, 1)

            # (N * OH * OW * C, PH * PW)
            pool_size = self.pool_h * self.pool_w
            dmax = np.zeros((dout.size, pool_size))

            # 順伝播時に最大値として採用された位置にだけ、doutの微分値（＝dout）をセット
            # 順伝播時に採用されなかった値の位置は初期化時の0のまま
            dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()

            # (N * OH * OW * C, PH * PW) → (N, OH, OW, C, PH * PW)
            dmax = dmax.reshape(dout.shape + (pool_size, ))

            # (N, OH, OW, C, PH * PW) → (N * OH * OW, C * PH * PW)
            dcol = dmax.reshape(dmax.shape[0] * dmax.shape[2], -1)

             # (N * OH * OW, C * PH * PW) → (N, C, H, W)
            dx = col2im(col=dcol, input_shape=self.X.shape, filter_h=self.pool_h, 
                        filter_w=self.pool_w, stride=self.stride, pad=self.pad)
            return dx

    
class Flatten():
    def __init__(self):
        self.prev = None

    def forward(self, X):
        self.prev = X.shape
        #(N, C, H, W)を(N, C*H*W)に
        return X.reshape((X.shape[0], -1))

    def backward(self, dout):
        #(N, C*H*W)を(N, C, H, W)に
        return (dout.reshape(self.prev))
    
    
class Affine:

    def __init__(self, W, b):
        """
            W (numpy.ndarray): 重み
            b (numpy.ndarray): バイアス
        """
        self.W = W                      # 重み
        self.b = b                      # バイアス
        self.X = None                   # 入力（2次元化後）
        self.dW = None                  # 重みの微分値
        self.db = None                  # バイアスの微分値
        self.original_x_shape = None    # 元の入力の形状（3次元以上の入力時用）

        
    def forward(self, x):
        """
        X (numpy.ndarray): 入力

        Returns:
            numpy.ndarray: 出力
        """
        
        # 3次元以上（テンソル）の入力を2次元化
        self.original_x_shape = x.shape  # 形状を保存
        X = X.reshape(x.shape[0], -1)
        self.X = X

        # 出力を算出
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        """
            dout (numpy.ndarray): 右の層から伝わってくる微分値

        Returns:
            numpy.ndarray: 微分値
        """
        # 微分値算出
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)

        # 元の形状に戻す
        dx = dx.reshape(*self.original_x_shape)
        return dx    

    
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        """
        X (numpy.ndarray): 入力

        Returns:
            numpy.ndarray: 出力
        """
        self.mask = (X <= 0)
        out = X.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        """
        dout (numpy.ndarray): 右の層から伝わってくる微分値

        Returns:
            numpy.ndarray: 微分値
        """
        dout[self.mask] = 0
        dx = dout

        return dx
    
    
class Softmax():
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, X, t):
        self.t = t
        self.y = self.softmax(X)
        self.loss = self.cross_entropy_error(self.y, self.t)
        return self.loss 
    
    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        
        batch_size = y.shape[0]

        return -np.sum(t * np.log(y + 1e-7)) / batch_size

    def softmax(self, X):
        X = X - np.max(X, axis=-1, keepdims=True)
        y = np.exp(X) / np.sum(np.exp(X), axis=-1, keepdims=True)
        return y
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dX = (self.y - self.t) / batch_size
        return dX
    
    
class AdaGrad:

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None   # これまでの勾配の2乗和

    def update(self, params, grads):
        """
        params (dict): 更新対象のパラメーターの辞書、keyは'W1'、'b1'など。
        grads (dict): paramsに対応する勾配の辞書
        """

        # hの初期化
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        # 更新
        for key in params.keys():

            # hの更新
            self.h[key] += grads[key] ** 2

            # パラメーター更新、最後の1e-7は0除算回避
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# In[10]:


import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'python', 'layer.ipynb'])


# In[ ]:




