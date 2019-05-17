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

import time, copy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

from eval.evalMUCF1 import calc_muc_score
from eval.evalF1 import comp_prec_rec_f1
from eval.evalDendPurity import calc_dend_purity
from hier_clust.recursive_sparsest_cut import run_sparsest_cut
from hier_clust.expLink import runHAC
from hier_clust.random_split import run_random_split


# TODO: See if this can be done away with
perMethodMetrics_pw = ["precision","precision_num","precision_den",
					   "recall", "recall_num", "recall_den",
					   "f1","randIndex", "dendPurity"]

def get_affinity_mat(model, pairFeatures, numPoints, dType, getNPMat):
	'''
	:param model: Scoring model
	:param pairFeatures: Dictionary with keys (pid1,pid2) containing value as feature vector for (pid1,pid2)
	:param numPoints: Number of points
	:param dType: Type of entries in affinity matrix: Should they be similaritie or dissimilarities
				  Takes two valid values: sim or dist
	:param getNPMat: Return an NP Matrix?
					If false, returns a pyTorch matrix
	:return: affinity matrix where (i,j) entry is affinity between point i and j
	
	'''
	if model.config.outDisSim:
		if dType == "sim":
			sign = -1
		elif dType == "dist":
			sign = 1
		else:
			raise Exception("Invalid value for dType={}. Use dType='sim' if you want to compute similarity matrix, "
							"or dType='dist' if you want to compute distance matrix".format(dType))
	else:
		if dType == "sim":
			sign = 1
		elif dType == "dist":
			sign = -1
		else:
			raise Exception("Invalid value for dType={}. Use dType='sim' if you want to compute similarity matrix, or"
							" dType='dist' if you want to compute distance matrix".format(dType))
			
	numFeature = 0
	for pair in pairFeatures:
		numFeature 	= len(pairFeatures[pair])
		break
		
	zeroVec 	= [0 for x in range(numFeature)]
	pairFeatureMat = []
	for pid1 in range(numPoints):
		row = []
		for pid2 in range(numPoints):
			p1, p2 = min(pid1, pid2), max(pid1,pid2)
			if pid1 != pid2:
				row.append(pairFeatures[(p1, p2)])
			else:
				zeroCopy = copy.copy(zeroVec)
				row.append(zeroCopy)
				
		pairFeatureMat.append(row)
	
	pairFeatureMatTorch = model.pairBatchForward(pairFeatureMat)
	pairFeatureMatTorch = sign*pairFeatureMatTorch.view(numPoints, numPoints)
	if getNPMat:
		npMat = pairFeatureMatTorch.cpu().data.numpy()
		npMat = npMat - np.diag(np.diag(npMat)) # Make diagonal matrix zero
		return npMat
	else:
		return pairFeatureMatTorch
	
def comp_sparse_mat_pair_feat(model, pidToIdx, pairFeatures, threshold):
	'''
	
	:param model: Model to compute score between a pair of points
	:param pidToIdx: Maps each pid to idx in range(0,numPoints)
	:param pairFeatures: Dictionary mapping a pair of pids to feature vector over them
	:param threshold: Threshold to compare to when removing entries in matrix
	
	:return: Sparse Matrix with values smaller than threshold if model is outputting distance
			or with values larger than threshold if model is outputting similarity
	'''
	
	# TODO Make this fast by using torch.Threshold
	numPoints = len(pidToIdx)
	data = []
	rows = []
	cols = []
	
	pidPairList = list(pairFeatures.keys())
	pidPairToIdx = {pidPair: idx for idx, pidPair in enumerate(pidPairList)}
	pairFeatureList = [pairFeatures[pidPair] for pidPair in pidPairList]

	
	batchAdjMatrix = model.pairBatchForward(pairFeatureList)
	batchAdjMatrix = batchAdjMatrix.cpu().data.numpy()
	
	for pidPair in sorted(pidPairList,reverse=True,key= lambda pair: pair[1]):
		
		edgeWeight = batchAdjMatrix[pidPairToIdx[pidPair]][0]
		if model.config.outDisSim:
			addToSparseMatrix = edgeWeight < threshold
		else:
			addToSparseMatrix = edgeWeight > threshold
			
		
		if addToSparseMatrix:
			data += [edgeWeight]
			rows += [pidToIdx[pidPair[0]]]
			cols += [pidToIdx[pidPair[1]]]
			
			data += [edgeWeight]
			rows += [pidToIdx[pidPair[1]]]
			cols += [pidToIdx[pidPair[0]]]
			
	# print("Fraction of zeros:{:.6f}\tNumEntries:{}/{}".format(1 - len(data) / (2*numPairs),len(data),2*numPairs))
	sparseMatrix = csr_matrix((data, (rows, cols)), shape=(numPoints, numPoints))
	return sparseMatrix

def eval_model_pair_feat(config, model, canopies, inferenceMethods, threshDict, metricsForEval):
	metricList = []
	for method in inferenceMethods:
		for metric in perMethodMetrics_pw:
			metricList.append(method + "_" + metric)
			
	tempScores = {metric: [] for metric in metricList}
	# print("Evaluating model with infMethod = {} and metrics = {}".format(inferenceMethods, metricsForEval))
	
	for canopyId in canopies:
		canopy = canopies[canopyId]
		currScores = perform_inf_pair_feat(model=model, pairFeatures=canopy["pairFeatures"], pidToCluster=canopy["pidToCluster"],
										   clusterToPids=canopy["clusterToPids"], inferenceMethods=inferenceMethods, threshDict=threshDict,
										   metricsForEval=metricsForEval, scaleDist=config.scaleDist)
		for metric in metricList:
			tempScores[metric] += [currScores[metric] if metric in currScores else 0.]
			
	scores = avg_scores(tempScores, metricList)

	return scores

def eval_model_pair_feat_per_canopy(model, canopies, inferenceMethods, threshDict, logger, metricsForEval):
	metricList = []
	for method in inferenceMethods:
		for metric in perMethodMetrics_pw:
			metricList.append(method + "_" + metric)
			
	start = time.time()
	tempScores = {metric: {} for metric in metricList}
	# print("Evaluating model with infMethod = {} and metrics = {}".format(inferenceMethods, metricsForEval))
	
	for canopyId in canopies:
		canopy = canopies[canopyId]
		currScores = perform_inf_pair_feat(model=model, pairFeatures=canopy["pairFeatures"], pidToCluster=canopy["pidToCluster"],
										   clusterToPids=canopy["clusterToPids"], inferenceMethods=inferenceMethods, threshDict=threshDict,
										   metricsForEval=metricsForEval)
		for metric in metricList:
			tempScores[metric][canopyId] = currScores[metric] if metric in currScores else 0.
			
	
	scores = tempScores
	# scores = averageScores_perCanopy(tempScores, metricList)

	# logger.info("==" * 20)
	# for metric in metricList:
	# 	logger.info("Scores\t{:_<40}{:.3f}\t{:.3f}".format(metric, scores[metric][0], scores[metric][1]))
	# logger.info("==" * 20)
	
	end = time.time()
	logger.info("Time taken to evaluate model::{:.3f}".format(end - start))
	return scores

def perform_inf_pair_feat(model, pairFeatures, pidToCluster, clusterToPids, inferenceMethods, threshDict, metricsForEval, scaleDist):
	"""
	:param model:
	:param pairFeatures:
	:param pidToCluster:
	:param clusterToPids:
	:param inferenceMethods:
	:param threshDict:
	:param metricsForEval:
	:param scaleDist:
	:return:
	"""
	printLog = False
	start = time.time()
	
	# Should create a copy as I am changing value of threshold by using smallest distance in the distance
	# matrix to make all values strictly positive. Need to make all values positive otherwise scipy complains
	threshDict = copy.copy(threshDict)
	
	pidList = sorted(list(pidToCluster.keys()))
	numPoints = len(pidList)
	pidToIdx = {pid:idx for idx,pid in enumerate(pidList)}
	
	results = {}
	t1 = time.time()
	distMatrix 		= get_affinity_mat(model=model, pairFeatures=pairFeatures, numPoints=numPoints, dType='dist', getNPMat=True)

	flatDistMatrix = None
	for method in inferenceMethods:
		if method.startswith("singleLink") or method.startswith("compLink") or method.startswith("avgLink"):
			flatDistMatrix 	= squareform(distMatrix)
			minDist 		= np.min(flatDistMatrix)
			flatDistMatrix 	= flatDistMatrix - minDist + 1
			# Add minDist to threshold as we are adding minDist to all values in distMatrix
			# when using it with singleLink, avgLink or compLink linkages
			if "singleLink@t" in threshDict:
				threshDict["singleLink@t"] 	= threshDict["singleLink@t"] - minDist + 1
			if "avgLink@t" in threshDict:
				threshDict["avgLink@t"] 	= threshDict["avgLink@t"] 	- minDist + 1
			if "compLink@t" in threshDict:
				threshDict["compLink@t"] 	= threshDict["compLink@t"] 	- minDist + 1
			break
			
	t2 = time.time()
	if printLog: print("\t\tTime taken by getAffinity matrix  = {:.3f}".format(t2 -  t1))
	
	y_true = [pidToCluster[pid] for pid in pidList] # y_pred = flatClusters # This is okay only when pids are already in range(0,numPoints-1)
	
	for method in inferenceMethods:
		mStart = time.time()
		if method == "connComp":
			threshold = threshDict["connComp"] if "connComp" in threshDict else None
			sparseMatrix = comp_sparse_mat_pair_feat(model=model, pairFeatures=pairFeatures, pidToIdx=pidToIdx, threshold=threshold)
			x = connected_components(sparseMatrix)
			numComponents = x[0]
			connectedComponents = x[1]
			y_pred = [connectedComponents[idx] for idx, pid in enumerate(pidList)]
			dendPurity = 0
		elif method == "recSparsest":
			
			labels = np.array([pidToCluster[pid] for pid in pidList])
			new_dist_mat_NP =np.max(distMatrix) - distMatrix
			linkTree = run_sparsest_cut(new_dist_mat_NP, labels)
			y_pred = y_true
			if "dendPurity" in metricsForEval:
				dendPurity = calc_dend_purity(linkTree=linkTree, pidList=pidList, y_true=y_true)
			
			
		elif method == "random":
			y_pred, dendPurity = run_random_split(pidToCluster=pidToCluster, k=len(clusterToPids))
		
		elif method.startswith("linkage"):
			#TODO Quick debug: similaritMatrix with single linkage should give exactly same result distance matrix with complete linkage
			linkageAlpha = None
			if method.startswith("linkage_min"):
				linkageAlpha = "min"
				threshold = threshDict[method] if method == "linkage_min@t" else None
			elif method.startswith("linkage_max"):
				linkageAlpha = "max"
				threshold = threshDict[method] if method == "linkage_max@t" else None
			elif method.startswith("linkage_auto"):
				if hasattr(model, "linkAlpha"):
					linkageAlpha = float(model.linkAlpha.cpu().data.numpy()[0])
				else:
					print("Trying to evaluate on method = {}, and model does not have linkAlpha parameter".format(method))
					continue
			elif method.startswith("linkage"):
				try:
					if method.endswith("@t"):
						linkageAlpha = float(method[:-2].split("_")[-1])
					else:
						linkageAlpha = float(method.split("_")[-1])
				except Exception as e:
					raise Exception("Invalid value of linkageAlpha = {}. Eg use method=linkage_1.0".format(method))
				
			if method.endswith("@t"):
				threshold = threshDict[method]
				flatClusters, dendPurity = runHAC(origDistMat=distMatrix, k=None, linkAlpha=linkageAlpha,
												numPoints=numPoints, pidToCluster=None, threshold=threshold, scaleDist=scaleDist)
			else:
				flatClusters, dendPurity = runHAC(origDistMat=distMatrix, k=len(clusterToPids), linkAlpha=linkageAlpha,
												numPoints=numPoints, pidToCluster=pidToCluster, threshold=None, scaleDist=scaleDist)
				
			y_pred = flatClusters # This is okay only when pids are already in range(0,numPoints-1)
		
		else:
			lt1 = time.time()
			threshold = threshDict[method] if method in threshDict else None
			if method.startswith("singleLink"):
				linkTree = linkage(y=flatDistMatrix, method="single")
			elif method.startswith("compLink"):
				linkTree = linkage(y=flatDistMatrix, method="complete")
			elif method.startswith("avgLink"):
				linkTree = linkage(y=flatDistMatrix, method="average")
			else:
				raise Exception("Invalid linkage method:{}".format(method))
	
			lt2 = time.time()
			
			ft1 = time.time()
			if method.endswith("@t"): # Obtain flat clusters by cutting tree using a threshold
				flatClusters = fcluster(Z=linkTree, t=threshold, criterion="distance")
			else: # Obtain flat clustering by cutting tree so that we get same number of clusters as in ground truth
				flatClusters = fcluster(Z=linkTree, t=len(clusterToPids), criterion="maxclust")
			ft2 = time.time()
			
			y_pred = flatClusters
			if "dendPurity" in metricsForEval:
				dendPurity = calc_dend_purity(linkTree=linkTree, pidList=pidList, y_true=y_true)
			else:
				dendPurity = 0
			if printLog: print("\t\tTime taken by inference method:{} Link+Flat = {:.3f} + {:.3f}".format(method, lt2 - lt1, ft2 - ft1))
			
		mEnd = time.time()
		
		if printLog: print("\t\tTime taken by inference method:{} = {:.3f}".format(method, mEnd - mStart))
		if printLog: print("Time taken by inference method:{} = {:.3f}".format(method, mEnd - mStart))
		
		if "muc_f1" in metricsForEval:
			mucScore   = calc_muc_score(pidToCluster_pred=y_pred, pidToCluster_true=y_true)
			for metric in mucScore:
				results[method + "_"+metric] = mucScore[metric]
				
		if "f1" in metricsForEval:
			tempResult = comp_prec_rec_f1(y_true, y_pred)
			for metric in tempResult:
				results[method + "_" + metric] = tempResult[metric]
				
		if "randIndex" in metricsForEval:
			results[method + "_randIndex"] 	= adjusted_rand_score(y_true, y_pred)
		if "nmi" in metricsForEval:
			results[method + "_nmi"] 		= adjusted_mutual_info_score(y_true, y_pred, average_method="arithmetic")
		if "dendPurity" in metricsForEval:
			results[method + "_dendPurity"] = 0 if (method == "connComp" or method.endswith("@t")) else dendPurity
		
	if printLog :print("Inference Time:{:.3f} on {} points\n\n".format(time.time() - start, numPoints))
	
	return results

def get_conn_comp_pair_feat(model, pairFeatures, pidToCluster, threshold):
	
	pidList = sorted(list(pidToCluster.keys()))
	pidToIdx = {pid: idx for idx, pid in enumerate(pidList)}
	
	sparseMatrix = comp_sparse_mat_pair_feat(model=model, pairFeatures=pairFeatures, pidToIdx=pidToIdx,
											 threshold=threshold)
	
	x = connected_components(sparseMatrix)
	connectedComponents = x[1]
	
	pidToCluster = {}
	for idx, pid in enumerate(pidList):
		pidToCluster[pid] = connectedComponents[idx]
	
	return pidToCluster

# Averages scores for all canopies into 1 final score dictionary
def avg_scores_per_canopy(tempScores, metricList):
	scores = {}
	for metric in metricList:

		if metric.endswith("precision") or metric.endswith("recall"):
			scores[metric + "_macro"] = np.mean(tempScores[metric]), np.std(tempScores[metric])
			numerator  	= sum(tempScores[metric + "_num"])
			denominator = sum(tempScores[metric + "_den"])
			scores[metric] = numerator / denominator if denominator != 0 else 1.
			scores[metric] = scores[metric], 0
		else:
			scores[metric] = np.mean(tempScores[metric]), np.std(tempScores[metric])

	# Update F1 using micro precision and micro recall
	for metric in metricList:
		if metric.endswith("_f1"):
			scores[metric + "_macro"] = scores[metric]
			tempPrec = scores[metric[:-2] + "precision"][0]
			tempRecall = scores[metric[:-2] + "recall"][0]
			scores[metric] = 2 * tempPrec * tempRecall / (tempPrec + tempRecall) if tempRecall + tempPrec != 0 else 0
			scores[metric] = scores[metric], 0

	return scores

# Averages scores for all canopies into 1 final score dictionary
def avg_scores(tempScores, metricList):
	scores = {}
	for metric in metricList:

		if metric.endswith("precision") or metric.endswith("recall"):
			scores[metric + "_macro"] = np.mean(tempScores[metric]), np.std(tempScores[metric])
			numerator  	= sum(tempScores[metric + "_num"])
			denominator = sum(tempScores[metric + "_den"])
			scores[metric] = numerator / denominator if denominator != 0 else 1.
			scores[metric] = scores[metric], 0
		else:
			scores[metric] = np.mean(tempScores[metric]), np.std(tempScores[metric])

	# Update F1 using micro precision and micro recall
	for metric in metricList:
		if metric.endswith("_f1"):
			scores[metric + "_macro"] = scores[metric]
			tempPrec = scores[metric[:-2] + "precision"][0]
			tempRecall = scores[metric[:-2] + "recall"][0]
			scores[metric] = 2 * tempPrec * tempRecall / (tempPrec + tempRecall) if tempRecall + tempPrec != 0 else 0
			scores[metric] = scores[metric], 0

	return scores
