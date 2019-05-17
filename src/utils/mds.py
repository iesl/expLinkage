"""
Copyright (C) 2019 University of Massachusetts Amherst.
This file is part of "expLinkage"
http://github.com/iesl/expLinkage
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sklearn.datasets import load_digits
from sklearn.manifold import MDS
import random, itertools

from utils.plotting import plot_clusters

def runMDSDummy():
	X, _ = load_digits(return_X_y=True)
	embedding = MDS(n_components=2)
	X_transformed = embedding.fit_transform(X[:100])
	
	pointToCluster = {}
	for x in X_transformed:
		pointToCluster[tuple(x)] = random.randint(0, 2)
	
	edges = []
	for p1, p2 in itertools.combinations(pointToCluster, 2):
		if pointToCluster[p1] == pointToCluster[p2]:
			edges += [(p1[0], p1[1], p2[0], p2[1])]
	
	plot_clusters(pointToCluster=pointToCluster, filename='../results/testMDS.png', edgeList=edges)

def runMDS(simMatrix, pidToCluster,filename):
	
	embedding = MDS(n_components=2,dissimilarity='precomputed')
	X_transformed = embedding.fit_transform(simMatrix)
	
	pointToCluster = {}
	for pid,x in enumerate(X_transformed):
		pointToCluster[tuple(x)] = pidToCluster[pid]
		
	numPoints  = len(pidToCluster)
	edges = []
	for p1,p2 in itertools.combinations( pointToCluster, 2):
		if pointToCluster[p1] == pointToCluster[p2]:
			edges += [(p1[0],p1[1],p2[0],p2[1])]
			
	plot_clusters(pointToCluster=pointToCluster, filename='../results/testMDS_{}.png'.format(filename), edgeList=edges)
	plot_clusters(pointToCluster=pointToCluster, filename='../results/testMDSWithout_{}.png'.format(filename))
	

if __name__ == "__main__":
	
	
	# runMDS()
	pass