import os
import numpy as np 
import matplotlib.pyplot as plt
import math
import random
import sys

x = []
y = []

class Cluster(object):

	def __init__(self,c_x,c_y,no):
		self.no = no
		self.points = set()
		self.centroid_x = c_x
		self.centroid_y = c_y

	def add_point(self,point):
		self.points.add(point)

	def remove_point(self,point):
		self.points.remove(point)

	def distance(self,x,y):
		x1_x2 = (self.centroid_x - x ) ** 2
		y1_y2 = (self.centroid_y - y) ** 2
		sum_sq = x1_x2 + y1_y2
		return math.sqrt(sum_sq)

	def sum_sq_error(self):
		total = 0.0
		for idx in self.points:
			total += self.distance(x[idx],y[idx])
		return total

	def recalculate_centroid(self):
		x_sum = 0.0
		y_sum = 0.0
		for idx in self.points:
			x_sum += x[idx]
			y_sum += y[idx]
		self.centroid_x = x_sum / len(self.points)
		self.centroid_y = y_sum / len(self.points)


def k_means(k,rr):
	centroids = set()
	clusters = []
	n = len(x)
	# choose k random centroids
	while len(centroids) < k:
		random_point = random.randint(0,n)
		if random_point not in centroids:
			centroids.add(random_point)
			cluster = Cluster(x[random_point],y[random_point],len(clusters))
			clusters.append(cluster)

	
	# assign points to nearest clusters
	for idx in xrange(len(x)):
		min_dist = sys.float_info.max
		min_dist_cluster = -1
		# compare distance between all centroids
		for j in xrange(k):
			dist = clusters[j].distance(x[idx],y[idx])
			if dist < min_dist:
				min_dist = dist
				min_dist_cluster = j
		clusters[min_dist_cluster].add_point(idx)

	
	
	centroids_changed = True
	while centroids_changed:
		centroids_changed = False
		for j in xrange(k):
			clusters[j].recalculate_centroid()

		for idx in xrange(len(x)):
			current_cluster = -1
			for m in xrange(k):
				if idx in clusters[m].points:
					current_cluster = m


			min_dist = sys.float_info.max
			min_dist_cluster = -1
			# compare distance between all centroids
			for j in xrange(k):
				dist = clusters[j].distance(x[idx],y[idx])
				if dist < min_dist:
					min_dist = dist
					min_dist_cluster = j
			if current_cluster != min_dist_cluster:
				clusters[current_cluster].remove_point(idx)
				clusters[min_dist_cluster].add_point(idx)
				centroids_changed = True

	#plt.show()
	
	total_sse = 0.0
	for j in xrange(k):
		total_sse += clusters[j].sum_sq_error()
	total_sse /= n

	plt.clf()
	fig = plt.figure(figsize=(18,9))
	tit = 'K Means Clustering K={0}  SSE={1}'.format(k,total_sse)
	fig.suptitle(tit, fontsize=15)
	ax1 = fig.add_subplot(111)
	colors = ['red','blue','green','yellow']
	for j in xrange(k):
		x_temp = []
		y_temp = []
		for idx in clusters[j].points:
			x_temp.append(x[idx])
			y_temp.append(y[idx])
		ax1.scatter(x_temp,y_temp,c = colors[j],label='Cluster:{0}'.format(j+1))
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.legend(loc='upper left')
	curr_d = os.getcwd()
	graph_d = os.path.join(curr_d,'k_means_graph/Dataset_1/k_2/graph_{0}.png'.format(rr))
	plt.savefig(graph_d)
	#plt.show()
	return total_sse

def k_means_r(k,r):
	iterations = []
	sses = []
	for i in xrange(1,r+1):
		sse = k_means(k,i)
		iterations.append(i)
		sses.append(sse)
	plt.clf()
	plt.scatter(iterations,sses,c='blue')
	plt.xlabel('r')
	plt.ylabel('SSE')
	plt.title('r vs SSE Dataset:1 ')
	curr_d = os.getcwd()
	graph_d = os.path.join(curr_d,'k_means_graph/Dataset_1/k_2/graph_{0}.png'.format(random.randint(0,1000)))
	plt.savefig(graph_d)
	#plt.show()

def main():
	curr_d = os.getcwd()
	data = os.path.join(curr_d,'Dataset_1.txt')
	with open(data) as fp:
		for line in fp:
			x_y = line.split(' ')
			x.append(float(x_y[0]))
			y.append(float(x_y[1]))
	k = 2
	r = 5
	k_means_r(k,r)



if __name__ == '__main__':
	main()