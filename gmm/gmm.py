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
		elif init_option == 2:
			for i in xrange(self.K):
					self.priors[i] = float(1/self.K)
			if self.K == 2:
				self.MU = np.array([[3.0269399725,0.04039995125],[-0.12112793375,-0.04725860125]]).T
				self.Sigma = np.zeros((K,self.D,self.D))
				self.Sigma[0] = np.array([[0.78454199,0.01197598],[0.01197598,0.95989469]])
				self.Sigma[1] = np.array([[0.81922146,-0.00963075],[-0.00963075,0.96540313]])
			elif self.K == 3:
				self.MU = np.array([[-1.17925694828,1.23781304023],[-0.169391248038,-0.396167298273],[1.79224925399,0.175532635704]]).T
				self.Sigma = np.zeros((K,self.D,self.D))
				self.Sigma[0] = np.array([[0.38409962,-0.25538808],[-0.25538808,0.33538165]])
				self.Sigma[1] = np.array([[0.51268655,0.23746005],[0.23746005,0.45334124]])
				self.Sigma[2] = np.array([[0.22008666,-0.05003686],[-0.05003686,1.27600262]])


			#K means mean vectors and covariance matrices
			# k = 2
			# dataset 1 
			# m1 - [3.0269399725,0.04039995125]
			# m2 - [-0.12112793375 -0.04725860125]
			# c1 - [[0.78454199 0.01197598]
 			#		[0.01197598 0.95989469]]
 			# c2 - [[ 0.81922146 -0.00963075]
 			#      [-0.00963075  0.96540313]]
 			#
 			# k =3 
 			#m1 = [-0.52175498762 0.0382393810179]
 			#m2 = [1.72186328191 1.04772567021]
 			# m3 - [1.64650398992 -0.850836261965]
 			#
 			# c1 = [[ 0.55007501 -0.04858143]
 			#	[-0.04858143  0.86404997]]
 			#
 			#c2 - [[0.28104922 0.03141673]
 				#[0.03141673 0.37528025]]
 			#
 			# c3 - [[ 0.31232164 -0.01573548]
 			#  [-0.01573548  0.36061645]]


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
		plt.title("Iterations vs Log likelihood of Dataset 2: K means initialization")
		data_dir = os.getcwd()
		pa = os.path.join(data_dir,'Dataset2_k/itervslog.png')
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


def plot_clusters(X, Q,K):
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
	plt.title("Gaussian Clustering Dataset 2: K means initialization")
	plt.xlabel('X')
	plt.ylabel('Y')
	data_dir = os.getcwd()
	pa = os.path.join(data_dir,'Dataset2_k/gmm.png')
	plt.savefig(pa)
	#plt.show()

def main():
	data_dir = os.getcwd()
	file = os.path.join(data_dir,'Dataset_2.txt')
	data = pd.read_csv(file,sep=' ',header=None)
	data.columns = ['X','Y','empty']
	points = data[['X','Y']]
	k = 3
	gmm = GMM(points,k,2)
	Q = gmm.model()
	plot_clusters(points.values,Q,k)


if __name__ == '__main__':
	main()