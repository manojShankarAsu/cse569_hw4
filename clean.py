import os
import numpy as np 
import matplotlib.pyplot as plt
import math
import random
import sys

def main():
	curr_d = os.getcwd()
	data = os.path.join(curr_d,'Dataset_2.txt')
	data3 = os.path.join(curr_d,'Dataset_3.txt')
	f = open(data3,'w')
	with open(data) as fp:
		for line in fp:
			x_y = line.split(' ')
			for s in x_y:
				if 'e' in s:
					f.write(s.replace('\n',''))
					f.write(' ')
			f.write('\n')
			
	f.close()

if __name__ == '__main__':
	main()