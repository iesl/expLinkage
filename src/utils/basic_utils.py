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

import logging
from pathlib import Path
import  math
import numpy as np
import torch
import itertools
from utils.Config import Config

def create_logger(config, logFile=None, currLogger=None):
	
	assert isinstance(config, Config)
	if currLogger is not None:
		for handler in currLogger.handlers[:]:
			currLogger.removeHandler(handler)
	
	if logFile is not None:
		config.logFile = logFile
		
	l = logging.getLogger("new_logger")
	fileHandler = logging.FileHandler("{}/{}".format(config.resultDir, config.logFile), mode='a')
	l.setLevel(logging.DEBUG)
	l.addHandler(fileHandler)
	
	if config.logConsole: # Add console to logger
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
		l.addHandler(console)
	
	newLogger = logging.getLogger("new_logger")
	return newLogger

def load_canopy_data_splits(config):
	
	canopyData 		= read_canopy_data(config.dataDir)
	allCanopyIds 	= list(canopyData.keys())
	
	numTrain 	= int(math.ceil(len(allCanopyIds) * config.trainFrac))
	numTest 	= int(math.ceil(len(allCanopyIds) * config.testFrac))
	numDev 		= int(math.ceil(len(allCanopyIds) * config.devFrac))

	if config.shuffleData:
		np.random.shuffle(allCanopyIds)

	trainCanopyIds 	= allCanopyIds[:numTrain]
	testCanopyIds 	= allCanopyIds[numTrain:numTrain + numTest]
	devCanopyIds 	= allCanopyIds[numTrain + numTest: numTrain + numTest + numDev]

	trainCanopies 	= {id: canopyData[id] for id in trainCanopyIds}
	testCanopies 	= {id: canopyData[id] for id in testCanopyIds}
	devCanopies 	= {id: canopyData[id] for id in devCanopyIds}
	
	return trainCanopies, testCanopies, devCanopies

def load_vec_data_splits(config):
	
	if "face" in config.clusterFile:
		clusters = read_clusters(config.clusterFile)
	elif "synthetic" in config.clusterFile:
		clusters = read_clusters_synth(config.clusterFile)
	else:
		clusters = read_clusters_synth(config.clusterFile)
	
	while True:
		trainCanopies, devCanopies, testCanopies = split_vec_data(config=config, clusters=clusters)
		if not config.shuffleData:
			print("\n\nNot checking is split is balanced or not \n\n")
			break
		
		if "pathbased" in config.clusterFile or "synthetic" in config.clusterFile:
			break
		
		if eval_split_quality(trainCanopies, testCanopies, devCanopies):
			break
	
	return trainCanopies, testCanopies, devCanopies
			
def split_vec_data(config, clusters):
	''' Puts trainFrac fraction of clustres in trainClusters and remaining clusters in testClusters, but each
	  but in batches where each batch has testFrac fraction of clusters.'''
	
	trainFrac = config.trainFrac
	testFrac = config.testFrac
	devFrac = config.devFrac
	
	numTotal = len(clusters)
	numTrain = math.ceil(trainFrac * numTotal)
	numTest = math.ceil(testFrac * numTotal)
	if isinstance(devFrac, float):
		numDev = math.ceil(devFrac * numTotal)
	else:
		numDev = 0
	
	allCids = list(clusters.keys())
	if config.shuffleData:
		np.random.shuffle(allCids)
	
	########################## Create trainCanopy ###################################
	trainCids = allCids[:numTrain]
	trainClusters = {cid: clusters[cid] for cid in trainCids}
	trainPoints = {}
	pid = 0
	for cid in trainClusters:
		for point in trainClusters[cid]:
			trainPoints[pid] = (point, cid)
			pid += 1
	
	trainCanopies = [{"clusters": trainClusters, "points": trainPoints}]
	
	########################## Create devCanopy ###################################
	if numDev > 0:
		devCids = allCids[numTrain:numTrain + numDev]
		devClusters = {cid: clusters[cid] for cid in devCids}
		devPoints = {}
		pid = 0
		for cid in devClusters:
			for point in devClusters[cid]:
				devPoints[pid] = (point, cid)
				pid += 1
		
		devCanopies = [{"clusters": devClusters, "points": devPoints}]
	else:
		devCanopies = []
	
	########################## Create testCanopy ###################################
	testCanopies = []
	testStartCid = numTrain + numDev
	testEndCid = testStartCid + numTest
	numTestCanopies = 0
	while testStartCid < len(allCids):
		numTestCanopies += 1
		
		testCids = allCids[testStartCid:testEndCid]
		testClusters = {cid: clusters[cid] for cid in testCids}
		pid = 0
		testPoints = {}
		for cid in testClusters:
			for point in testClusters[cid]:
				testPoints[pid] = (point, cid)
				pid += 1
		
		canopy = {"clusters": testClusters, "points": testPoints}
		testCanopies.append(canopy)
		testStartCid += numTest
		testEndCid += numTest
		
	
	return trainCanopies, devCanopies, testCanopies
	
def eval_split_quality(trainCanopies, testCanopies, devCanopies):

	avgDist = {"train": 0, "test": 0, "dev": 0}
	for canopy in trainCanopies:
		centroid = []
		for cid in canopy["clusters"]:
			centroid.append(np.mean(canopy["clusters"][cid], axis=0))
		
		avgDist["train"] = []
		for c1, c2 in itertools.combinations(centroid, 2):
			avgDist["train"] += [np.linalg.norm(c1 - c2)]
		avgDist["train"] = np.mean(avgDist["train"])
	
	for canopy in devCanopies:
		centroid = []
		for cid in canopy["clusters"]:
			centroid.append(np.mean(canopy["clusters"][cid], axis=0))
		
		avgDist["dev"] = []
		for c1, c2 in itertools.combinations(centroid, 2):
			avgDist["dev"] += [np.linalg.norm(c1 - c2)]
		avgDist["dev"] = np.mean(avgDist["dev"])
	
	for canopy in testCanopies:
		centroid = []
		for cid in canopy["clusters"]:
			centroid.append(np.mean(canopy["clusters"][cid], axis=0))
		
		avgDist["test"] = []
		for c1, c2 in itertools.combinations(centroid, 2):
			avgDist["test"] += [np.linalg.norm(c1 - c2)]
		avgDist["test"] = np.mean(avgDist["test"])
	
	print("{}\n".format(avgDist))
	if abs(avgDist["dev"] - avgDist["test"]) < 0.05 * min(avgDist["dev"], avgDist["test"]):
		return True
	
	return False

def read_clusters_synth(filename):
	'''
	
	:param filename: Name of file with data
			Format of file: <x1> <x2> <clusterId>
	:return:
	clusters: Dictionaty with key as clusterIds, and value as list of points in that cluster
	'''
	clusters	= {}
	with open(filename,"r") as reader:
		for line in reader:
			splitLine  	= line.strip().split()
			point 		= tuple(float(x) for x in splitLine[0:2]) # First 2 numbers are floats and are coordinates of point
			cid		 	= int(splitLine[2]) # 3rd value contain cluster of this point
			if cid not in clusters:
				clusters[cid] = []

			clusters[cid].append(point)

	return clusters

def read_clusters(filename):
	'''
	
	:param filename: Name of file with data
			Format of file: <point_id> <cluster_id> <dim1> <dim2> ....
	:return:
	clusters: Dictionaty with key as clusterIds, and value as list of points in that cluster
	'''

	clusters = {}
	with open(filename, "r") as reader:
		for line in reader:
			splitLine = line.strip().split()
			point = tuple(float(x) for x in splitLine[2:])
			cid 	= 	splitLine[1] # 2nd value contain cluster of this point
			pointId = 	splitLine[0] # 1st value contains id of he point

			if cid not in clusters:
				clusters[cid] = []

			clusters[cid].append(point)

	return clusters

def read_canopy_data(dataDir):
	'''
	
	:param dataDir: DataDir has folder for each canopy, and each folder contains pairFeatures.csv and gtClusters.tsv
	:return: canopyData: Dictionary with canopyId as keys, and
						canopyData[id] is  in turn a dictionary with keys
							pidToCluster : Dict mapping each point to its cluster
							clusterToPids: Dict mapping each cluster to list of ids of points in ti
							pairFeatures : Dict with (pid1,pid2) as key and feature vector over pid1 and pid2 as value
						
							
	'''
	
	canopyData = {}
	folderList = [str(f) for f in Path(dataDir).glob("*") if f.is_dir()]
	print("Number of files:{}".format(len(folderList)))

	for ctr, folder in enumerate(folderList):
		print("processing folder:{}\t{}".format(ctr,folder))
		pidToCluster = {}
		pidToIntPid = {}  # Maps each point Id in original file to an integer ID in range(0,numPoints-1) so that I can use them as indices
		intPidCtr 	= 0
		with open("{}/gtClusters.tsv".format(folder)) as gtFile:
			for line in gtFile:
				line = line.strip().split()
				pid, cid = int(line[0]), int(line[1])
				if pid not in pidToIntPid:
					pidToIntPid[pid] = intPidCtr
					intPidCtr += 1

				intPid = pidToIntPid[pid]
				pidToCluster[intPid] = cid

		assert intPidCtr == len(pidToCluster)
		clusterToPids = {}
		for pid in pidToCluster:
			cid = pidToCluster[pid]
			try:
				clusterToPids[cid].append(pid)
			except:
				clusterToPids[cid] = [pid]
		
		pairFeatures = {}
		with open("{}/pairFeatures.csv".format(folder)) as featFile:
			for line in featFile:
				line = line.strip().split(",")
				featureVec = [float(x) for x in line[2:-1]]
				pid1, pid2 = int(line[0]), int(line[1])

				# Replace pids read from file with their corresponding mapped ids
				pid1, pid2 = pidToIntPid[pid1], pidToIntPid[pid2]

				if pid1 > pid2:
					pid1, pid2 = pid2, pid1
				
				pairFeatures[(pid1, pid2)] = (featureVec)
				assert (pid1 <= pid2)
				if line[-1] == "1":
					assert (pidToCluster[pid1] == pidToCluster[pid2])
				elif line[-1] == "0":
					assert (pidToCluster[pid1] != pidToCluster[pid2])
				else:
					print(line)
					raise Exception("Unexcepted end token in line. Expected 1 or 0")
				
		canopyId = folder.split("/")[-1]
		canopyData[canopyId] = {"pidToCluster": pidToCluster, "clusterToPids": clusterToPids,
								"pairFeatures": pairFeatures}
	return canopyData

def get_filename_list(parameters, template):
	filenameList = [template]
	paramList = list(parameters.keys())
	
	for paramCtr, param in enumerate(paramList):
		updateFilenameList = []
		for template in filenameList:
			if isinstance(parameters[param],list):
				for paramValue in parameters[param]:
					dict = {param: paramValue}
					for restParam in paramList[paramCtr + 1:]:
						dict[restParam] = "{" + str(restParam) + "}"
					filename = template.format(**dict)
					updateFilenameList.append(filename)
			else:
				dict = {param: parameters[param]}
				for restParam in paramList[paramCtr + 1:]:
					dict[restParam] = "{" + str(restParam) + "}"
				filename = template.format(**dict)
				updateFilenameList.append(filename)
				
		
		filenameList = updateFilenameList
	
	# for filename in filenameList:
	# 	print(filename)
	return filenameList

def generate_statistics(trainer):

	# assert isinstance(trainer, PairFeatureTrainer)
	
	microAvgNumPtsInCluster = {"all":[]}
	avgNumPtsInCluster 	= {"all":[]}
	numSingleton	 	= {"all":[]}
	fracSingleton		= {"all":[]}
	numPoints 			= {"all":[]}
	numClusters 		= {"all":[]}
	for canopyId in trainer.trainCanopies:
		canopy = trainer.trainCanopies[canopyId]
		ptsInCluster = [len(canopy["clusterToPids"][cid]) for cid in canopy["clusterToPids"]]
		
		avgNumPtsInCluster[canopyId] = np.mean(ptsInCluster)
		numClusters[canopyId] 	= len(canopy["clusterToPids"])

		numSingleton[canopyId]  = sum([1 for cid in canopy["clusterToPids"] if len(canopy["clusterToPids"][cid]) == 1])
		numPoints[canopyId] = sum(ptsInCluster)
	
		microAvgNumPtsInCluster["all"] += ptsInCluster
		avgNumPtsInCluster["all"] += [avgNumPtsInCluster[canopyId]]
		numClusters["all"] += [numClusters[canopyId]]
		numSingleton["all"] += [numSingleton[canopyId]]
		numPoints["all"] += [numPoints[canopyId]]
		fracSingleton["all"] += [float(numSingleton[canopyId])/numPoints[canopyId]]
		
	
	trainer.logger.info("Macro:Average Number of Points in each cluster:\t{:.3f}\t{:.3f}".format( np.mean( avgNumPtsInCluster["all"]), np.std( avgNumPtsInCluster["all"] ) ))
	trainer.logger.info("Micro:Average Number of Points in each cluster:\t{:.3f}\t{:.3f}".format( np.mean( microAvgNumPtsInCluster["all"]), np.std( microAvgNumPtsInCluster["all"] ) ))
	trainer.logger.info("Macro:Average Number of Singleton in canopies :\t{:.3f}\t{:.3f}".format( np.mean( numSingleton["all"]), np.std( numSingleton["all"] ) ))
	trainer.logger.info("Micro:Average Number of Singleton in canopies :\t{:.3f}\t{:.3f}".format( np.mean( fracSingleton["all"]), np.std( fracSingleton["all"] ) ))
	trainer.logger.info("Average Number of Clusters :\t{:.3f}\t{:.3f}".format( np.mean( numClusters["all"]), np.std( numClusters["all"] ) ))
	trainer.logger.info("Average Number of Points :\t{:.3f}\t{:.3f}".format( np.mean( numPoints["all"]), np.std( numPoints["all"] ) ))
	trainer.logger.info("Average Number of Points in each cluster:\t{:.3f}\t{:.3f}".format( np.mean( avgNumPtsInCluster["all"]), np.std( avgNumPtsInCluster["all"] ) ))
	# plt.clf()
	# plt.xscale('log')
	# plt.hist(microAvgNumPtsInCluster["all"],int(np.max(microAvgNumPtsInCluster["all"])))
	# plt.savefig("{}/clusterSizes.png".format(trainer.config.resultDir))
	#
	# plt.close()
	# plt.clf()
	# plt.xscale('log')
	# plt.hist(microAvgNumPtsInCluster["all"],100)
	# plt.savefig("{}/clusterSizes_100Bins.png".format(trainer.config.resultDir))
	# plt.close()
	#
	# plt.clf()
	# plt.xscale('log')
	# plt.hist(microAvgNumPtsInCluster["all"],10)
	# plt.savefig("{}/clusterSizes_10Bins.png".format(trainer.config.resultDir))
	# plt.close()

def calc_batch_size(numPoints, pointDim):
	'''
	# Find allocated GPU Memory and uses that to estimate largest batch size that it can probably support
	# Assumes that 10GB is total memory available for use
	:param numPoints: Number of points for which we want to compute adjacency matrix
	:param pointDim: Dimension of each point
	:return: batchSize: Largest value of m such that numPoints x m dimensional matrix can be created of pointDim
						dimensional points
	'''
	
	MAX_AVAIL_MB = 10* (2**10)
	
	memAllocMB = torch.cuda.memory_allocated() / (2 ** 20)
	memAvailMB = (MAX_AVAIL_MB - memAllocMB)  # ASSUMING TOTAL OF 10 GB memory is available for allocation on GPU
	
	batchSize = int((2 ** 18) * memAvailMB / (numPoints *pointDim))
	
	return  batchSize
