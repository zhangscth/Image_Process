#coding=utf-8
import numpy as np
def get_GLCM_feature(img_gray): # gray image
    img = img_gray.copy()
    gray_level = 256 #灰度级数
    gray_level_compose = 64 #压缩后灰度级数
    dx = 1
    dy = 1 # 搜寻灰度步进长度
    L,W = img.shape

    #2.为了减少计算量，对原始图像灰度级压缩，将img量化成16级
    for i in range(L):
        for j in range(W):
            for n in range(gray_level_compose):
                if (n - 1) * (gray_level / gray_level_compose) <= img[i, j] and \
                    img[i, j] <= (n - 1) * (gray_level / gray_level_compose) + (gray_level / gray_level_compose) - 1:
                    img[i, j] = n - 1


    #3.计算四个共生矩阵P,取距离为1，角度分别为0,45,90,135
    P=np.zeros([gray_level_compose,gray_level_compose,4])

    for m in range(gray_level_compose):
        for n in  range(gray_level_compose):
            for i in range(L):
                for j in range(W):
                    # print(i,j,i+dy,j+dy,i+dx,i+dx,j-dy)
                    if (j<W-dy and img[i , j] == m and img[i ,j+dy] ==n): #角度为0
                        P[m, n, 0] = P[m, n, 0] + 1
                    if (i <L - dx and j < W - dy and img[i, j] == m  and img[i + dx, j + dy] == n): # % 角度为45
                        P[m, n, 1] = P[m, n, 1] + 1
                    if (i <L - dx and img[i, j] == m and img[i + dx, j] == n): # % 角度为90
                        P[m, n, 2] = P[m, n, 2] + 1
                    if (j > dy and i < L - dx and img[i, j] == m  and img[i + dx, j - dy] == n ): # % 角度为135
                        P[m, n, 3] = P[m, n, 3] + 1
    print(P.shape)
    print(np.unique(P))
    #对共生矩阵归一化
    #for i in range(4):
    P = P/np.sum(P)

    #--------------------------------------------------------------------------
    # 4.
    #求共生矩阵计算(二阶矩)
    #能量、熵、惯性矩、相关、逆差分阵5个纹理参数
    # --------------------------------------------------------------------------
    Ans_data = np.zeros([4])
    energy = np.zeros([4]) # 二阶矩
    contrast = np.zeros([4]) # 对比度(惯性矩)
    correlation = np.zeros([4]) #% 相关度
    entropy = np.zeros([4]) #% 熵
    deficit = np.zeros([4]) #% 逆差分阵
    mean_x = np.zeros([4]) #% 均值
    mean_y = np.zeros([4]) #% 均值
    variance_x = np.zeros([4]) #% 均方差
    variance_y = np.zeros([4]) #% 均方差
    Exy = np.zeros([4])
    print(P.shape)
    for n in range(4):
        energy[n] = np.sum(np.square(np.sum(P[:,:,n])))
        for i in range(gray_level_compose):
            for j in range(gray_level_compose):
                if P[i,j,n]!=0:
                    entropy[n] = -P[i,j,n]*np.log(P[i,j,n])+entropy[n]
                contrast[n]=np.square((i-j))+contrast[n]
                mean_x[n] = i*P[i,j,n]+mean_x[n]
                mean_y[n] = i*P[i,j,n]+mean_y[n]

    for n in range(4):
        for i in range(gray_level_compose):
            for j in range(gray_level_compose):
                variance_x[n] = np.square((i - mean_x[n]))* P[i, j, n] + variance_x[n]
                variance_y[n] = np.square((j - mean_y[n]))* P[i, j, n] + variance_y[n]
                Exy[n] = i * j * P[i, j, n] + Exy[n]
                deficit[n] = P[i, j, n] / (1 + np.square(i - j) ) + deficit[n]

        correlation[n]= (Exy[n] - mean_x[n] * mean_y[n]) / (variance_x[n] * variance_y[n])
    # return_value=[]
    # for v in [energy,contrast,correlation,entropy,deficit,mean_x,mean_y,variance_x,variance_y,Exy]:
    #     for i in range(4):
    #         print(v[i])
    #         return_value.append(v[i])
    # print(",".join(return_value))
    print([energy[0],contrast[0],correlation[0],entropy[0],deficit[0],mean_x[0],mean_y[0],variance_x[0],variance_y[0],Exy[0],
            energy[1], contrast[1], correlation[1], entropy[1], deficit[1], mean_x[1], mean_y[1], variance_x[1], variance_y[1], Exy[1],
            energy[2], contrast[2], correlation[2], entropy[2], deficit[2], mean_x[2], mean_y[2], variance_x[2], variance_y[2], Exy[2],
            energy[3], contrast[3], correlation[3], entropy[3], deficit[3], mean_x[3], mean_y[3], variance_x[3], variance_y[3], Exy[3]])
    return [energy[0],contrast[0],correlation[0],entropy[0],deficit[0],mean_x[0],mean_y[0],variance_x[0],variance_y[0],Exy[0],
            energy[1], contrast[1], correlation[1], entropy[1], deficit[1], mean_x[1], mean_y[1], variance_x[1], variance_y[1], Exy[1],
            energy[2], contrast[2], correlation[2], entropy[2], deficit[2], mean_x[2], mean_y[2], variance_x[2], variance_y[2], Exy[2],
            energy[3], contrast[3], correlation[3], entropy[3], deficit[3], mean_x[3], mean_y[3], variance_x[3], variance_y[3], Exy[3]]

