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

import time
import numpy as np
import torch
import math
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, bmat
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from eval.evalF1 import comp_prec_rec_f1
from eval.evalDendPurity import calc_dend_purity

from models.mahalabonis import MahalanobisDist
from hier_clust.expLink import runHAC, runHAC_allEdges
from utils.basic_utils import calc_batch_size
from utils.plotting import plot_clusters
from hier_clust.recursive_sparsest_cut import run_sparsest_cut
from hier_clust.random_split import run_random_split

perMethodMetrics = ["precision", "recall", "f1", "randIndex", "dendPurity", "nmi"]

def eval_model_vect(config, model, canopies, inferenceMethods, threshDict, metricsForEval):
	
	metricList = []
	for method in inferenceMethods:
		for metric in perMethodMetrics:
			metricList.append(method + "_" + metric)
		
	if threshDict is None:
		print("Threshold is None and so not performing any inference")
		scores = {metric: (0,0) for metric in metricList}
		return scores
	
	start = time.time()
	tempScores = {metric: [] for metric in metricList}
	for ctr, canopy in enumerate(canopies):
		currTestScore = perform_inference(model=model, points=canopy["points"], threshDict=threshDict,
										  inferenceMethods=inferenceMethods, metricsForEval=metricsForEval, scaleDist=config.scaleDist)
		for metric in metricList:
			tempScores[metric] += [currTestScore[metric] if metric in currTestScore else 0.]

	scores = {}
	for metric in metricList:
		scores[metric] = np.mean(tempScores[metric]), np.std(tempScores[metric])
	
	end = time.time()
	# logger.info("==" * 20)
	# for metric in metricList:
	# 	logger.info("Scores\t{:40}{:.3f}\t\t{:.3f}".format(metric, scores[metric][0], scores[metric][1]))
	# logger.info("=="*20)
	# logger.info("Total evaluation time:{:.3f}\n\n\n".format(end - start))
	return scores

# Computes adjMatrix for points in pointList in batches, and converts it into a sparse representation
def comp_sparse_adj_mat(model, pointList, threshold):
	printLog = False
	numPoints = len(pointList)
	
	batchSize = calc_batch_size(numPoints, model.inputDim)
	numBatches = int(math.ceil(numPoints / batchSize))
	if printLog: print("BatchSize:{}\tNumBatches:{}\tnumPoints\t{}".format(batchSize,numBatches, numPoints))
	
	assert (numBatches * batchSize >= numPoints)
	sparseMatrixList = []
	for batchNum in range(numBatches):  # TODO : Avoid computation of lower half of matrix as much as possible
		if printLog: print("batchNum:{}".format(batchNum))
		startIdx = batchNum * batchSize
		endIdx = min(numPoints, (batchNum + 1) * batchSize)
		currPoints = pointList[startIdx: endIdx]
		batchAdjMatrix = model.batchForwardAcross(currPoints, pointList)
		
		applyThreshold = torch.nn.Threshold(-1 * threshold, 0)
		batchAdjMatrix = -1*applyThreshold(-1 * batchAdjMatrix)
		batchAdjMatrix = batchAdjMatrix.cpu().data.numpy()
		
		sparseMatrixList.append([coo_matrix(batchAdjMatrix)])
	
	sparseMatrix = bmat(sparseMatrixList)
	torch.cuda.empty_cache()
	numNonZero = len(sparseMatrix.nonzero()[0])
	if printLog: print("Fraction of zeros:{:.4f}\tNumEntries:{}".format(1 - numNonZero / (numPoints * numPoints), numNonZero))
	return sparseMatrix

def comp_sparse_adj_mat_slow(model, pointList, threshold):
	numPoints = len(pointList)
	data = []
	rows = []
	cols = []
	
	''' TODO: This is an adhoc way of deciding batchSize, use some routine that first
	finds amount of available GPU memory and then do it
	'''
	batchSize = int(2160 * 2160 / numPoints)
	numBatches = int(math.ceil(numPoints / batchSize))
	print("BatchSize:{}\tNumBatches:{}\tnumPoints\t{}".format(batchSize, numBatches, numPoints))
	print("Available memory:{}".format(torch.cuda.memory_allocated()))
	assert (numBatches * batchSize >= numPoints)
	
	dataAccTime = 0
	matrixTime = 0
	for batchNum in range(numBatches):  # TODO : Avoid computation of lower half of matrix as much as possible
		matrixTime1 = time.time()
		startIdx = batchNum * batchSize
		endIdx = min(numPoints, (batchNum + 1) * batchSize)
		currPoints = pointList[startIdx: endIdx]
		batchAdjMatrix = model.batchForwardAcross(currPoints, pointList)
		batchAdjMatrix = batchAdjMatrix.cpu().data.numpy()
		matrixTime2 = time.time()
		
		dataAccT1 = time.time()
		for x in range(startIdx, endIdx):
			for y in range(numPoints):
				if x == y:  # No need to add self-loops to adjMatrix
					continue
				
				if batchAdjMatrix[x - startIdx][y] <= threshold:
					data += [batchAdjMatrix[x - startIdx][y]]
					rows += [x]
					cols += [y]
		dataAccT2 = time.time()
		dataAccTime += dataAccT2 - dataAccT1
		matrixTime1 += matrixTime2 - matrixTime1
	
	print("Time spent on matrix computation:{:.3f}\t on applying threshold:{:.3f}".format(matrixTime, dataAccTime))
	print("Fraction of zeros:{:.4f}\tNumEntries:{}".format(1 - len(data) / (numPoints * numPoints),
														   len(data)))
	torch.cuda.empty_cache()
	sparseMatrix = csr_matrix((data, (rows, cols)), shape=(numPoints, numPoints))
	
	return sparseMatrix

def perform_inference(model, points, inferenceMethods, threshDict, metricsForEval, scaleDist):
	"""
	
	:param model:
	:param points: Dictionary with key as pid and value as (pointVector, clusterId)
	:param inferenceMethods:
	:param threshDict:
	:param metricsForEval:
	:param scaleDist:
	:return:
	"""
	torch.cuda.empty_cache()
	start = time.time()

	pidList 	= sorted(list(points.keys()))
	pointList 	= [points[pid][0] for pid in pidList]
	pidToGtClust= {pid:points[pid][1] for pid in pidList}
	gtList 		= {points[pid][1]:0 for pid in pidList}
	for pid in pidList:
		gtList[points[pid][1]] += 1
		
	numPoints 	= len(pointList)

	results = {}
	transformedPointList = None
	numComponents = 0
	
	if isinstance(model, MahalanobisDist):  # Tranform points if using Mahalanobis distance
		transformedPointList = model.transformPoints(pointList)
		linkMetric = "euclidean"
	else:
		raise Exception("Can not perform inference in this function with model type={}".format(type(model)))

	dendPurity = 0
	torchDistMat = model.batchForwardWithin(pointList)
	distMat_NP 	 = torchDistMat.cpu().data.numpy()
	
	y_true = []
	for idx, pid in enumerate(pidList):
		y_true.append(pidToGtClust[pid])
		
	for method in inferenceMethods:
		mStart = time.time()
		if method == "connComp":
			t1 = time.time()
			connCompThresh = threshDict["connComp"]
			sparseMatrix = comp_sparse_adj_mat(model, pointList, connCompThresh)
			t2 = time.time()
			print("Time taken for computing sparseMatrix:{:.3f}".format(t2-t1))
			
			x = connected_components(sparseMatrix)
			numComponents 		= x[0]
			connectedComponents = x[1]

			y_pred = []
			for idx,pid in enumerate(pidList):
				y_pred.append( connectedComponents[idx] )
		elif method == "recSparsest":
			
			labels = np.array([points[pid][1] for pid in pidList])
			new_dist_mat_NP =np.max(distMat_NP) - distMat_NP
			linkTree = run_sparsest_cut(new_dist_mat_NP, labels)
			y_pred = y_true
			if "dendPurity" in metricsForEval:
				dendPurity = calc_dend_purity(linkTree=linkTree, pidList=pidList, y_true=y_true)
			
		elif method == "random":
			y_pred, dendPurity = run_random_split(pidToCluster=pidToGtClust, k=len(gtList))

		elif method.startswith("linkage"):
			if method == "linkage_min" or method == "linkage_max":
				# raise Exception("Use singleLink or compLink inference method instead")
				linkageAlpha = method[-3:]
				flatClusters, dendPurity = runHAC(origDistMat=distMat_NP, k=len(gtList), linkAlpha=linkageAlpha, numPoints=numPoints, pidToCluster=pidToGtClust, threshold=None, scaleDist=scaleDist)
				y_pred = flatClusters
			elif method == "linkage_min@t" or method == "linkage_max@t":
				# raise Exception("Use singleLink or compLink inference method instead")
				linkageAlpha = method[-5:-2]
				threshold = threshDict[method]
				flatClusters, dendPurity = runHAC(origDistMat=distMat_NP, k=None, linkAlpha=linkageAlpha, numPoints=numPoints, pidToCluster=None, threshold=threshold, scaleDist=scaleDist)
				y_pred = flatClusters
			else:
				if method.startswith("linkage_auto"):
					if hasattr(model, "linkAlpha"):
						linkageAlpha = float(model.linkAlpha.data.cpu().numpy()[0])
					else:
						print("Not evaluating for method = {}".format(method, str(model)))
						continue
				else:
					try:
						if method.endswith("@t"):
							linkageAlpha = float(method[:-2].split("_")[-1])
						else:
							linkageAlpha = float(method.split("_")[-1])
					except:
						raise Exception("Invalid value of linkageAlpha = {}. Eg use method=linkage_1.0".format(method))
				
				if method.endswith("@t"): # Use a threshold to get flat clusters
					threshold = threshDict[method]
					flatClusters, dendPurity = runHAC_allEdges(origDistMat=distMat_NP, k=None, linkAlpha=linkageAlpha, numPoints=numPoints, pidToCluster=None, threshold=threshold, scaleDist=scaleDist)
				else: # Use number of gt clusters to get flat clusters
					if "dendPurity" in metricsForEval:
						flatClusters, dendPurity = runHAC_allEdges(origDistMat=distMat_NP, k=len(gtList), linkAlpha=linkageAlpha, numPoints=numPoints, pidToCluster=pidToGtClust, threshold=None, scaleDist=scaleDist)
					else: # No need to pass pidToCluster as we don't need to compute dendPurity
						flatClusters, dendPurity = runHAC_allEdges(origDistMat=distMat_NP, k=len(gtList), linkAlpha=linkageAlpha, numPoints=numPoints, pidToCluster=None, threshold=None, scaleDist=scaleDist)
						
				y_pred = flatClusters
				
				# ptToPredClusters = {point:y_pred[ctr] for ctr,point in enumerate(pointList)}
				# print("Plotting in file=",method + ".pdf")
				# plot_clusters(pointToCluster=ptToPredClusters,filename=method + ".pdf")
				# ptToGtClusters = {point:y_true[ctr] for ctr,point in enumerate(pointList)}
				# plot_clusters(pointToCluster=ptToGtClusters,filename=method + "_orig.pdf")
				
				
		else:
			if method.startswith("singleLink"):
				threshold = threshDict["singleLink@t"] if "singleLink@t" in threshDict else None
				linkTree = linkage(transformedPointList, "single",metric=linkMetric)
			elif method.startswith("avgLink"):
				threshold = threshDict["avgLink@t"] if "avgLink@t" in threshDict else None
				linkTree = linkage(transformedPointList, "average",metric=linkMetric)
			elif method.startswith("compLink"):
				threshold = threshDict["compLink@t"] if "compLink@t" in threshDict else None
				linkTree = linkage(transformedPointList, "complete",metric=linkMetric)
			else:
				linkTree = None
				print("Invalid inference method:{}".format(method))
				raise Exception("Invalid inference method:{}".format(method))
			
			if method.endswith("@t"):
				flatClusters = fcluster(Z=linkTree, t=threshold, criterion="distance")
			else:
				flatClusters = fcluster(Z=linkTree, t=len(gtList), criterion="maxclust")
				
			y_pred = flatClusters
			# ptToPredClusters = {point:y_pred[ctr] for ctr,point in enumerate(pointList)}
			# plot_clusters(pointToCluster=ptToPredClusters,filename=method + ".pdf")
			
			if "dendPurity" in metricsForEval:
				dendPurity = calc_dend_purity(linkTree=linkTree, pidList=pidList, y_true=y_true)
			
		
		mEnd = time.time()
		print("Time taken by inference method:{} = {:.3f}".format(method, mEnd - mStart))
		if "f1" in metricsForEval:
			tempResult = comp_prec_rec_f1(y_true, y_pred)
			for metric in tempResult:
				results[method + "_" + metric] 	= tempResult[metric]
		if "randIndex" in metricsForEval:
			results[method + "_randIndex"] = adjusted_rand_score(y_true, y_pred)
		if "nmi" in metricsForEval:
			results[method + "_nmi"] = adjusted_mutual_info_score(y_true, y_pred, average_method="arithmetic")
		if "dendPurity" in metricsForEval:
			results[method + "_dendPurity"] = 0 if method == "connComp" else dendPurity
		
	print("Inference Time:{:.3f} on {} points".format(time.time() - start,numPoints))
	return results
