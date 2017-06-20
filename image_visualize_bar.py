import random
from pylab import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cv2

from mpl_toolkits.mplot3d import Axes3D

pix_origin = cv2.imread("data/plot.png", 1)

pix = cv2.cvtColor(pix_origin, cv2.COLOR_RGB2GRAY)
print(pix.shape)


mpl.rcParams['font.size'] = 10

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# pix = (pix<255)*255


colors=["red","green","blue","gray","black","white","purple"]

y_all = []
for z in range(0,pix.shape[1],100):# y

    for x in range(0,pix.shape[0],100):

        y =  (0. + pix[x,z])/255.# z
        # print(ys)
        if y not in y_all:
            y_all.append(y)

        color = colors[y_all.index(y)]
        ax.bar([x,], [y,], zs=z, zdir='y', color=color, alpha=0.8,width=20,linewidth=1,edgecolor='b')

# ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xs))
# ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ys))

ax.set_xlabel('Month')
ax.set_ylabel('Year')
ax.set_zlabel('Sales Net [usd]')
#

pix = pix_origin
x, y = ogrid[0:pix.shape[0], 0:pix.shape[1]]
ax = fig.gca(projection='3d')

pix = pix/255
ax.plot_surface(x, y, 0, rstride=5, cstride=5, facecolors=pix)

plt.show()
