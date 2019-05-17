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

import csv, os, argparse, itertools, math
from pathlib import Path

from utils.basic_utils import read_canopy_data, read_clusters, read_clusters_synth
from utils.plotting import plot_clusters, plot_clusters_w_edges

def rotate(point,theta,anchor=(0,0)):
	point = point[0]-anchor[0],point[1]-anchor[1]
	point = math.cos(theta) * point[0] - math.sin(theta) * point[1], math.sin(theta) * point[0] + math.cos(theta) * point[1]
	point = point[0] + anchor[0], point[1] + anchor[1]
	return point
	
# Reads clusters from file=filename and creates pairwise features for these clusters and stores them in dataDir
def create_pairwise_data(filename, dataDir, squared=False):
	clusters = read_clusters_synth(filename)
	
	pointData  = {} # Maps each pointId to (point,cid) tuple
	pid = 0
	for cid in clusters:
		for point in clusters[cid]:
			pointData[pid] = (point,cid)
			pid+=1
	
	Path(dataDir).mkdir(exist_ok=True, parents=True)
	with open("{}/gtClusters.tsv".format(dataDir),"w") as f:
		csvWriter = csv.writer(f, delimiter="	")
		for pid in pointData.keys():
			row = [pid,pointData[pid][1]]
			csvWriter.writerow(row)

	with open("{}/pairFeatures.csv".format(dataDir),"w") as f:
		csvWriter = csv.writer(f)
		for pid1, pid2 in itertools.combinations(pointData.keys(),2):
			
			featureVec = [abs(x1-x2) for x1,x2 in zip(pointData[pid1][0], pointData[pid2][0])]
			if squared:
				featureVec = [x**2 for x in featureVec]
				
			row = [pid1, pid2] + featureVec
			if pointData[pid1][1] == pointData[pid2][1]:
				row.append(1)
			else:
				row.append(0)
			
			csvWriter.writerow(row)

# Reads clusters from file=filename and creates pairwise features for these clusters and stores them in dataDir
# This one is specially written for generating different datasets for spiral clusters
def create_pairwise_spiral(filename, dataDir, squared=False, theta=0., trimAt=None, pushMinValTo=None):
	clusters = read_clusters_synth(filename)
	
	pointData = {}  # Maps each pointId to (point,cid) tuple
	pointToCluster = {}  # Maps each point to its cluster
	pntCtr,pid = 0,0
	for cid in clusters:
		if cid == 2: continue
		# np.random.shuffle(clusters[cid])
		for point in clusters[cid][40:90]:
			pntCtr += 1
			if pntCtr % 5 != 0: continue
			if cid == 1:
				newPoint = rotate(point, theta, (16, 15))
			else:
				newPoint = point
			
			pointData[pid] = (newPoint, cid)
			pointToCluster[newPoint] = cid
			pid += 1
			
	Path(dataDir).mkdir(exist_ok=True, parents=True)
	plot_clusters(pointToCluster, dataDir + "/origData_{:.2f}.png".format(theta))
	with open(dataDir+"/orig2D.txt","w") as writer:
		for point in pointToCluster:
			writer.write("{}\t{}\t{}\n".format(point[0],point[1],pointToCluster[point]))
	
	with open(dataDir+"/pidToPoint.txt","w") as writer:
		for pid in pointData:
			point = pointData[pid]
			writer.write("{}\t{}\t{}\n".format(pid,point[0][0],point[0][1]))
	
	with open("{}/gtClusters.tsv".format(dataDir), "w") as f:
		csvWriter = csv.writer(f, delimiter="	")
		for pid in pointData.keys():
			row = [pid, pointData[pid][1]]
			csvWriter.writerow(row)
	
	with open("{}/pairFeatures.csv".format(dataDir), "w") as f:
		csvWriter = csv.writer(f)
		for pid1, pid2 in itertools.combinations(pointData.keys(), 2):
			
			featureVec = [abs(x1 - x2) for x1, x2 in zip(pointData[pid1][0], pointData[pid2][0])]
			if squared:
				featureVec = [x ** 2 for x in featureVec]
			
			if trimAt is not None and featureVec[0] > trimAt:
				featureVec[0], featureVec[1] = featureVec[1], featureVec[0]
				featureVec[0] = min(trimAt, featureVec[0])
			
			if pushMinValTo is not None and featureVec[0] + featureVec[1] < pushMinValTo:
				if featureVec[0] < featureVec[1]:
					featureVec[1] = pushMinValTo
				else:
					featureVec[0] = pushMinValTo
			
			row = [pid1, pid2] + featureVec
			if pointData[pid1][1] == pointData[pid2][1]:
				row.append(1)
			else:
				row.append(0)
			
			csvWriter.writerow(row)

if __name__ == "__main__":
	
	# Command to generate spiral dataset with some rotation, with 2 spiral where MST and allPairs differ significantly
	# python scripts / create_synth_dataset.py - -file =../ data / sprial.txt - -outDir =../ data / spiralSmallRotated
	
	parser = argparse.ArgumentParser(description='Create dataset with edges = |p1-p2| from points in Rd')

	parser.add_argument('--file', type=str, required=True, help='File containing points in Rd')
	parser.add_argument('--outDir', type=str, required=True, help='Directory for newly created dataset')
	parser.add_argument('--sq', action="store_true", default=False, help='Square each component of edge?')

	args = parser.parse_args()

	filename = args.file # filename 	= "../data/sprial.txt"
	dataDir  = args.outDir # dataDir = "../data/spiral_pw_sqd"


	# for theta in np.arange(0,3.14,0.1):
	for theta in [0.8]:
		create_pairwise_spiral(filename=filename,dataDir=dataDir+"/1",squared=args.sq,theta=theta)
		canopy = read_canopy_data(dataDir)
		plot_clusters_w_edges(canopy=canopy, model=None, filename=dataDir + "/1/edgeData_{:.2f}.png".format(theta))
	
	# clusters = readClusters_synth(filename)
	# points = {}
	# for cid in clusters:
	# 	for point in clusters[cid]:
	# 		points[point] = cid
	# plotClusters(points, dataDir+"/1/origData.png")
			
	# canopy = readCanopyData(dataDir)
	# plotClustersEdges(canopy=canopy, model=None, filename=dataDir+"/1/edgeData")

	# dataDir = "../data/spiral_pw/1"
	# create_pairwise_spiral(filename, dataDir, False)
	
	# dataDir = "../data/spiral_pw_sqd/1"
	# create_pairwise_spiral(filename, dataDir, True)
	
	# dataDir = "../data/spiral_pw_sqd_trimmed/1"
	# create_pairwise_spiral(filename, dataDir, True, 20)
	
	# dataDir = "../data/spiral_pw_sqd_trimmed_larger/1"
	# create_pairwise_spiral(filename, dataDir, True, 20,8)
