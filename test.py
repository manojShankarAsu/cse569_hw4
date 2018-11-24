import os
import numpy as np 
import matplotlib.pyplot as plt
import math
import random
import sys

def main():
	curr_d = os.getcwd()
	data = os.path.join(curr_d,'Dataset_2.txt')
	data3 = os.path.join(curr_d,'newfile.txt')
	f = open(data3,'w')
	f.write('Hello')
	f.write('How are you')	
	f.write(' ')
	f.write(' sas')	
	f.close()

if __name__ == '__main__':
	main()