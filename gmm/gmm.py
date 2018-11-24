from __future__ import division
import os
import numpy as np 
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.colors as c
import math
import random
import sys
from scipy.stats import multivariate_normal
import seaborn as sns
import pandas as pd


class GMM(object):
	def __init__(self,X,K,init_option):
		self.X = X.values # n * D
		#self.X = self.X[0:9,:]
		self.D = self.X.shape[1]
		self.N = self.X.shape[0]

		#self.X = self.X.reshape((self.N,self.D))
		self.K = K				
		self.Q = np.zeros((self.N,K))
		self.priors = np.zeros((K,))
		if init_option == 1:
			# random initialization
			self.MU = np.zeros((self.D,K))
			cov_matrix = np.cov(self.X[:,0] , self.X[:,1])
			pos_factor = random.randint(0,5)
			cov_matrix = pos_factor * cov_matrix
			self.Sigma = np.zeros((K,self.D,self.D))
			for i in xrange(K):
				self.Sigma[i] = cov_matrix
				self.priors[i] = float(1/self.K)
				random_point = random.randint(0,self.N-1)
				self.MU[:,i] = self.X[random_point].T

	def model(self):
		# k means initialization - option 2
		not_converged = True
		log_likelihood_found = False
		iterations = []
		log_likes = []
		count = 0
		while not_converged:
			# Expectation Step
			count += 1
			iterations.append(count)
			for i in xrange(self.N):
				for k in xrange(self.K):
					pdf = multivariate_normal.pdf(self.X[i],mean = self.MU[:,k],cov=self.Sigma[k])
					#print 'pdf: {0}'.format(pdf)					
					#print 'Priors: {0}'.format(self.priors[k])
					self.Q[i][k] = pdf * self.priors[k]
				total = np.sum(self.Q[i])
				self.Q[i] /=  total


			# Maximization
			N_K = np.sum(self.Q,axis=0) #[k1,k2]
			#print 'N_K'
			#print N_K
			self.priors = N_K / self.N
			self.MU = np.dot(self.X.T,self.Q) / N_K
			self.Sigma = np.zeros((self.K,self.D,self.D))
			for k in xrange(self.K):
				for i in xrange(self.N):
					X_Mu= np.reshape(self.X[i]-self.MU[:,k],(self.D,1))
					self.Sigma[k] += self.Q[i,k] * np.dot(X_Mu,X_Mu.T)
				self.Sigma[k] /= N_K[k]

			if not log_likelihood_found:
				log_likelihood = self.calc_log_like()
				log_likelihood_found = True
				log_likes.append(log_likelihood)
			else:
				old_log_likelihood = log_likelihood
				log_likelihood = self.calc_log_like()
				log_likes.append(log_likelihood)
				#print 'new log_likelihood {0}'.format(log_likelihood)
				if math.fabs(log_likelihood - old_log_likelihood) < 0.1:
					not_converged = False
		plt.scatter(iterations, log_likes, c='blue', edgecolor='black')
		plt.xlabel('Iterations')
		plt.ylabel('Log Likelihood')
		plt.title("Iterations vs Log likelihood of Dataset 2: Random initialization of Mean vector")
		data_dir = os.getcwd()
		pa = os.path.join(data_dir,'Dataset2/itervslog.png')
		plt.savefig(pa)
		#plt.show()
		return self.Q


	def calc_log_like(self):
		sum = 0.0
		for i in xrange(self.N):
			for k in xrange(self.K):
				# print 'MU'
				# print self.MU[:,k]
				# print 'Sigma'
				# print self.Sigma[k]
				sum += math.log(self.priors[k])
				sum += multivariate_normal.logpdf(self.X[i],mean=self.MU[:,k],cov=self.Sigma[k])
		return sum


def vis_data_shade(X, Q,K):
	# Use specific RGB values instead of discrete char identifiers if num_classes > 8
	char_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
	rgb_colors = []
	for i in range(K):
		rgb_colors.append(c.to_rgb(char_colors[i]))
	rgb_colors = np.array(rgb_colors).flatten()
	colorings = np.zeros((Q.shape[0], 3))
	for n in range(Q.shape[0]):
		for k in range(Q.shape[1]):
			# Calculate R,G,B contributions
			colorings[n,0] += Q[n,k]*rgb_colors[k*3]
			colorings[n,1] += Q[n,k]*rgb_colors[k*3+1]
			colorings[n,2] += Q[n,k]*rgb_colors[k*3+2]
	plt.clf()
	plt.scatter(X[:,0], X[:,1], c=colorings, edgecolor='black')
	plt.title("Gaussian Clustering Dataset 2: Random initialization of Mean vector")
	plt.xlabel('X')
	plt.ylabel('Y')
	data_dir = os.getcwd()
	pa = os.path.join(data_dir,'Dataset2/gmm.png')
	plt.savefig(pa)
	#plt.show()

def main():
	data_dir = os.getcwd()
	file = os.path.join(data_dir,'Dataset_2.txt')
	data = pd.read_csv(file,sep=' ',header=None)
	data.columns = ['X','Y','empty']
	points = data[['X','Y']]
	k = 3
	gmm = GMM(points,k,1)
	Q = gmm.model()
	vis_data_shade(points.values,Q,k)


if __name__ == '__main__':
	main()