import os
import numpy as np 
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import random
import sys
from scipy import stats
import seaborn as sns
import pandas as pd

sns.set_style('white')

def main():
	# x = np.linspace(start=-10,stop=10,num=1000)
	# y = stats.norm.pdf(x,loc=0,scale=1.5)
	# plt.plot(x,y)
	# plt.show()
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	zdata = 15 * np.random.random(100)
	xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
	ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
	ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
	plt.show()
	#df = pd.read_csv('bimodal_example.csv')
	#print df.head(5)


if __name__ == '__main__':
	main()