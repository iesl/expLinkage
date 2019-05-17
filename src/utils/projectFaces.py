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

import sys
import numpy as np
from sklearn.decomposition import PCA

from utils.basic_utils import read_clusters


def projectFaces(filename, dim):
	clusterData = read_clusters(filename)
	
	pointList = []
	indices = {}
	for cid in clusterData:
		start = len(pointList)
		pointList += clusterData[cid]
		end = len(pointList)
		indices[cid] = (start, end)
	
	pointList = [list(point) for point in pointList]
	pointList = np.array(pointList)
	# print(pointList.shape)
	
	pca = PCA(n_components=dim, random_state=0)
	X_prime = pca.fit_transform(pointList)
	print("Explained variance ratio for {} components\t{}\n{}".format(dim,pca.explained_variance_ratio_,sum(pca.explained_variance_ratio_)))
	
	# print(X_prime.shape)
	newClusterData = {}
	for cid in clusterData:
		start, end = indices[cid]
		newClusterData[cid] = X_prime[start:end]
	# print(newClusterData[cid].shape)
	
	with open("../data/faceData_{}.tsv".format(dim), "w") as writer:
		pointId = 0
		for cid in newClusterData:
			for point in newClusterData[cid]:
				row = "{}\t{}\t".format(pointId, cid)
				row += "\t".join("{:.2f}".format(x) for x in point)
				# print(row)
				writer.write(row + "\n")
				pointId += 1


def normalizeFaces(filename):
	clusterData = read_clusters(filename)
	
	pointList = []
	indices = {}
	for cid in clusterData:
		start = len(pointList)
		pointList += clusterData[cid]
		end = len(pointList)
		indices[cid] = (start, end)
	
	
	maxVal = 0.
	for cid in clusterData:
		for point in clusterData[cid]:
			tempMax = np.max([abs(x) for x in point])
			maxVal = max(tempMax, maxVal)
	
	maxVal = 100
	newFilename = filename[:-4] + "_norm_10.tsv"
	with open(newFilename, "w") as writer:
		pointId = 0
		for cid in clusterData:
			for point in clusterData[cid]:
				row = "{}\t{}\t".format(pointId, cid)
				# Z = sum(point)
				Z = np.linalg.norm(point)
				origPoint = point
				point = [x/maxVal for x in point]
				# point = [x/Z for x in point]
				row += "\t".join("{:.2f}".format(x) for x in point)
				# row += "\t".join("{:.2f}".format(x) for x in origPoint)
				print(row)
				writer.write(row + "\n")
				pointId += 1

	print(maxVal)
	
if __name__ == "__main__":
	
	dim = int(sys.argv[1])
	projectFaces("../data/faceData.tsv",dim)
