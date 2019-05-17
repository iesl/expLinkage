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


import itertools, math, os
import numpy as np
import torch
import copy, time

from scipy.special import softmax

# Runs HAC with ExpLink for given value of linkAlpha and returns k-flatClusters and dendrogram purity
def runHAC(origDistMat, k, linkAlpha, numPoints, pidToCluster, threshold, scaleDist):
	"""
	Runs HAC with ExpLink for given value of linkAlpha and returns k-flatClusters and dendrogram purity
	
	:param origDistMat: Numpy Distance Matrix
	:param k: Number of clusters to use to cut tree. If None, then tree is cut using threshold value
	:param linkAlpha: Value of alpha parameter of expLink
	:param numPoints: Number of points
	:param pidToCluster: Dictionary that maps point id to ground-truth cluster id
	:param threshold: Threshold to use to cut tree. If none then tree is cut using k(given)
	:param scaleDist: Should all distances in the matrix be scaled using largest edge in distance matrix? Useful of preventing numerical overflows
	:return: (y_pred, dendPurity)
			y_pred : List that maps pid to its cluster id. Flat Clusters obtained by cutting tree using threshold or k
			dendPurity: Dendrogram purity of cluster tree build by greedy HAC with given linkAlpha
	
	"""
	
	assert (k is None) or (threshold is None)
	assert (k is not None) or (threshold is not None)
	
	activeClusters = [pid for pid in range(numPoints)]
	newCid          = numPoints

	pidToParent = {}
	children    = {pid:None for pid in activeClusters}
	if k is not None and len(activeClusters) <= k:
		# Technically I should construct a tree and then compute
		# dendrogram purity in this case as well, but I am not doing it...
		return activeClusters, 0

	distMat = getDistMat(origDistMat=origDistMat, linkAlpha=linkAlpha, numPoints=numPoints,
						 matType="simple", scaleDist=scaleDist, dataStructType="matrix")
	y_pred = None
	mergeTime = 0
	updateTime = 0
	while len(activeClusters) > 1:

		# Find clusters to merge
		t1 = time.time()
		c1,c2  = findMerger_matVersion(linkAlpha=linkAlpha, distMat=distMat, currClusterList=activeClusters)
		
		mergeTime += time.time() - t1
		
		# Get flat clusters by cutting tree at threshold. y_pred being None means that flat clusters have not been obtained even once
		# If flat clusters are obtained once by cutting the tree, then we need not do so in consecuting steps and y_pred is None condition prevents us from doingit
		if (threshold is not None) and (y_pred is  None):
			
			cutTree = False
			if (linkAlpha == "min" or linkAlpha == "max"):
				if distMat[c1][c2] > threshold: cutTree = True
			else:
				if distMat[0][c1][c2]/distMat[1][c1][c2] > threshold: cutTree = True
			
			# if len(activeClusters) == numPoints:
			# 	print("\n\n\n\nReturning just at the beginnning..{}..{}\t{}\t{}......\n\n\n\n".format(cutTree,distMat[0][c1][c2]/distMat[1][c1][c2],distMat[0][c1][c2],distMat[1][c1][c2]))
			# 	print("numPoints:{}".format(numPoints))
			# 	print(origDistMat[c1][c2])
			
			if cutTree: # Should I cut tree now
				pidToPredCluster_thresh = getPidToPredClusters(numPoints=numPoints, pidToParent=pidToParent)
				y_pred = [pidToPredCluster_thresh[ pid ] for pid in range(numPoints)]
				if pidToCluster is None: # No need to continue performing HAC because pidToCluster is None, so dendPurity can not be computed anyway
					# if len(activeClusters) == numPoints:
					# 	print("\n\n\n\nReturning just at the beginnning..........\n\n\n\n")
						
					return y_pred, 0
		
		try:
			# Remove merged clusters for list
			activeClusters.remove(c1)
			activeClusters.remove(c2)
		except Exception as e:
			print(c1,c2)
			raise e
			
		t1 = time.time()
		# Update distances of the merged cluster with all remaining clusters
		distMat = updateDistMat_matVersion(linkAlpha=linkAlpha, distMat=distMat, matType="simple",
										   currClusterList=activeClusters, newCid=newCid, oldC1=c1, oldC2=c2)
		updateTime += time.time() - t1
		
		activeClusters.append(newCid)
		
		children[newCid] 	= (c1,c2)
		pidToParent[c1] 	= newCid
		pidToParent[c2] 	= newCid

		if k is not None and len(activeClusters) == k: # Get flat clusters such that there are k clusters
			pidToPredCluster_k = getPidToPredClusters(numPoints=numPoints, pidToParent=pidToParent)
			y_pred = [pidToPredCluster_k[ pid ] for pid in range(numPoints)]
			if pidToCluster is None: # No need to continue performing HAC because pidToCluster is None, so dendPurity can not be computed anyway
				# print("\t\tTime taken to update distance  matrix (fast):{:.4f}".format(updateTime))
				# print("\t\tTime taken to merger clusters         (fast):{:.4f}".format(mergeTime))

				return y_pred, 0

		newCid += 1
	
	if y_pred is None: # This is triggered when while loop terminated without forming flat clusters. it means that all points are put in 1 cluster
		y_pred = [1 for x in range(numPoints)]
	
	if pidToCluster is None:
		dendPurity = 0
	else:
		dendPurity = computeDendPurity(pidToCluster=pidToCluster, children=children, pidToParent=pidToParent)
	
	# print("\t\tTime taken to update distance  matrix :{:.4f}".format(updateTime))
	# print("\t\tTime taken to merger clusters         :{:.4f}".format(mergeTime))
	return y_pred, dendPurity


def calc_linkage_numpy(linkAlpha, values):
	if linkAlpha == "min":
		return np.min(values)
	elif linkAlpha == "max":
		return np.max(values)
	else:
		weights = softmax(linkAlpha*values)
		return np.sum(weights*values)
	
# Runs HAC with ExpLink for given value of linkAlpha and returns k-flatClusters and dendrogram purity
def runHAC_allEdges(origDistMat, k, linkAlpha, numPoints, pidToCluster, threshold, scaleDist):
	
	"""
	Runs HAC with ExpLink for given value of linkAlpha and returns k-flatClusters and dendrogram purity
	
	:param origDistMat: Numpy Distance Matrix
	:param k: Number of clusters to use to cut tree. If None, then tree is cut using threshold value
	:param linkAlpha: Value of alpha parameter of expLink
	:param numPoints: Number of points
	:param pidToCluster: Dictionary that maps point id to ground-truth cluster id
	:param threshold: Threshold to use to cut tree. If none then tree is cut using k(given)
	:param scaleDist: Should all distances in the matrix be scaled using largest edge in distance matrix? Useful of preventing numerical overflows
	:return: (y_pred, dendPurity)
			y_pred : List that maps pid to its cluster id. Flat Clusters obtained by cutting tree using threshold or k
			dendPurity: Dendrogram purity of cluster tree build by greedy HAC with given linkAlpha
	
	"""
	
	assert (k is None) or (threshold is None)
	assert (k is not None) or (threshold is not None)
	
	activeClusters = [pid for pid in range(numPoints)]
	newCid          = numPoints

	pidToParent = {}
	children    = {pid:None for pid in activeClusters}
	if k is not None and len(activeClusters) <= k:
		# Technically I should construct a tree and then compute
		# dendrogram purity in this case as well, but I am not doing it...
		return activeClusters, 0

	distMat = getDistMat(origDistMat=origDistMat, linkAlpha=linkAlpha, numPoints=numPoints,
						 matType="simple", scaleDist=scaleDist, dataStructType="allEdges")
	y_pred = None
	mergeTime = 0
	updateTime = 0
	cache = {}
	ctr = 0
	while len(activeClusters) > 1:
		ctr+=1
		# Find clusters to merge
		t1 = time.time()
		(c1,c2), merge_val  = findMerger_allEdges(linkAlpha=linkAlpha, distMat=distMat, currClusterList=activeClusters, cache=cache)
		
		# (c1_debug,c2_debug)  = findMerger_matVersion(linkAlpha=linkAlpha, distMat=distMat_debug, currClusterList=activeClusters)
		# merge_val_debug = distMat_debug[0][c1_debug][c2_debug]/distMat_debug[1][c1_debug][c2_debug]
		# # if merge_val != merge_val_debug:
		# print("round",ctr)
		# # print("currClustList:",len(activeClusters), activeClusters)
		# print("Merge vals,",c1,c2, merge_val)
		# print("Merge vals debug,",c1_debug,c2_debug, merge_val_debug, distMat_debug[0][c1_debug][c2_debug], distMat_debug[1][c1_debug][c2_debug])
		# # print("orig dist,",origDistMat[c1_debug][c2_debug])
		# print("Linkage value of those chosen by mat version",calc_linkage(linkAlpha, distMat[c1_debug,c2_debug]))
		# if min(c1,c2) != min(c1_debug,c2_debug ) or max(c1_debug,c2_debug) != max(c1,c2):
		# 	raise Exception("Diiferen t.......")
		# if round(abs(merge_val- merge_val_debug),2) != 0:
		#
		# 	raise Exception("Different ")
		#
		
		mergeTime += time.time() - t1
		
		# Get flat clusters by cutting tree at threshold. y_pred being None means that flat clusters have not been obtained even once
		# If flat clusters are obtained once by cutting the tree, then we need not do so in consecuting steps and y_pred is None condition prevents us from doingit
		if (threshold is not None) and (y_pred is  None):
			
			cutTree = False
			if merge_val > threshold: cutTree = True
			
			if cutTree: # Should I cut tree now
				pidToPredCluster_thresh = getPidToPredClusters(numPoints=numPoints, pidToParent=pidToParent)
				y_pred = [pidToPredCluster_thresh[ pid ] for pid in range(numPoints)]
				if pidToCluster is None: # No need to continue performing HAC because pidToCluster is None, so dendPurity can not be computed anyway
					# if len(activeClusters) == numPoints:
					# 	print("\n\n\n\nReturning just at the beginnning..........\n\n\n\n")
					
					return y_pred, 0
		
		try:
			# Remove merged clusters for list
			activeClusters.remove(c1)
			activeClusters.remove(c2)
		except Exception as e:
			print(c1,c2)
			raise e
			
		t1 = time.time()
		# Update distances of the merged cluster with all remaining clusters
		distMat = updateDistMat_allEdges(linkAlpha=linkAlpha, distMat=distMat, matType="simple",
										 currClusterList=activeClusters, newCid=newCid, oldC1=c1, oldC2=c2)
		# distMat_debug = updateDistMat_matVersion(linkAlpha=linkAlpha, distMat=distMat_debug, matType="simple",
		# 								 currClusterList=activeClusters, newCid=newCid, oldC1=c1_debug, oldC2=c2_debug)
		updateTime += time.time() - t1
		
		activeClusters.append(newCid)
		
		children[newCid] 	= (c1,c2)
		pidToParent[c1] 	= newCid
		pidToParent[c2] 	= newCid

		if k is not None and len(activeClusters) == k: # Get flat clusters such that there are k clusters
			pidToPredCluster_k = getPidToPredClusters(numPoints=numPoints, pidToParent=pidToParent)
			y_pred = [pidToPredCluster_k[ pid ] for pid in range(numPoints)]
			if pidToCluster is None: # No need to continue performing HAC because pidToCluster is None, so dendPurity can not be computed anyway
				# print("\t\tTime taken to update distance  matrix (fast):{:.4f}".format(updateTime))
				# print("\t\tTime taken to merger clusters         (fast):{:.4f}".format(mergeTime))

				return y_pred, 0

		newCid += 1
	
	if y_pred is None: # This is triggered when while loop terminated without forming flat clusters. it means that all points are put in 1 cluster
		y_pred = [1 for x in range(numPoints)]
	
	if pidToCluster is None:
		dendPurity = 0
	else:
		dendPurity = computeDendPurity(pidToCluster=pidToCluster, children=children, pidToParent=pidToParent)
	
	# print("\t\tTime taken to update distance  matrix :{:.4f}".format(updateTime))
	# print("\t\tTime taken to merger clusters         :{:.4f}".format(mergeTime))
	
	# y_pred_1, dendPurity_1 = runHAC(origDistMat, k, linkAlpha, numPoints, pidToCluster, threshold, scaleDist)
	# if dendPurity != dendPurity_1:
	# 	print(dendPurity, dendPurity_1)
	# 	assert round(abs(dendPurity_1 - dendPurity),6) == 0
	# 	exit(0)
	return y_pred, dendPurity

# Run HAC on all points and try to compute loss
# This version uses two matrices to store numerator and denominator of ExpLink linkage separately,
# This has some numerical stability issues
def runHAC_torch_num_den(origDistMat, origTorchDistMat, linkAlpha, linkAlphaTorch, pidToGtCluster, numPoints, scaleDist):
	"""
	Runs HAC with ExpLink for given value of linkAlpha and tensors for some pure and impure merger values
	This version uses two matrices to store numerator and denominator of ExpLink linkage separately,
	This has some numerical stability issues

	
	:param origDistMat: Numpy Distance Matrix
	:param origTorchDistMat: Torch distance matrix
	:param linkAlpha: Value of alpha parameter of expLink
	:param linkAlphaTorch: Torch variable storing value of alpha parameter of explink
	:param pidToCluster: Dictionary that maps point id to ground-truth cluster id
	:param numPoints: Number of points
	:param threshold: Threshold to use to cut tree. If none then tree is cut using k(given)
	:param scaleDist: Should all distances in the matrix be scaled using largest edge in distance matrix? Useful of preventing numerical overflows
	:return: posLinkageVals, negLinkageVals : Both are torch tensors
			posLinkageVals: Linkage value of pure mergers that are worse than some impure merger
			negLinkageVals: Linkge value of impure mergers that are better than some pure merger
	"""
	
	# Map each sub-cluster to its corresponding gt-cluster. This is helpful in finding if two clusters being merged are part of same gt-cluster or not
	subClusterToGtCluster = copy.deepcopy(pidToGtCluster)
	
	activeClusters = [pid for pid in range(numPoints)]
	newCid 			= numPoints

	posLinkageVals = torch.cuda.FloatTensor(np.zeros((numPoints-1, 1)))
	negLinkageVals = torch.cuda.FloatTensor([]) # Empty tensor
		
	pidToParent = {}
	children 	= {pid: None for pid in activeClusters}
	
	t1 = time.time()
	distMat		= getDistMat(origDistMat=origDistMat, linkAlpha=linkAlpha, numPoints=numPoints,
							 matType="simple", scaleDist=scaleDist, dataStructType="tuple")
	t2 = time.time()
	torchDistMat= getDistMat(origDistMat=origTorchDistMat, linkAlpha=linkAlphaTorch, numPoints=numPoints,
							 matType="pytorch", scaleDist=scaleDist, dataStructType="tuple")
	t3 = time.time()
	print("\t\tTime taken to get compatible matrices:{:.4f}\t{:.4f}\t{:.4f}".format(t3 - t1, t2 - t1, t3 - t2))
	
	mergeTime, updateTimeTorch, updateTimeSimple, linkCalcTime = 0,0,0,0
	while len(activeClusters) > 1:
		
		# Find clusters to merge
		t1 = time.time()
		(c1, c2), impureLinkages = findPureMerger_tuple(linkAlpha=linkAlpha, distMat=distMat,currClusterList=activeClusters,
														subClusterToGtCluster=subClusterToGtCluster)
		mergeTime += time.time() - t1
		
		if (c1 is None) and (c2 is None): # Could not find 2 pure clusters to merge. At this time, we have agglomerated all gt-clusters into separate sub-trees
			break
		
		
		subClusterToGtCluster[newCid] = subClusterToGtCluster[c1]
		assert subClusterToGtCluster[c1] == subClusterToGtCluster[c2]
		
		t1 = time.time()
		
		
		if linkAlpha == "min" or linkAlpha == "max":
			posLinkageVals[newCid - numPoints] = torchDistMat[(c1, c2)]
		else:
			posLinkageVals[newCid - numPoints] = torchDistMat[(c1, c2)][0]/torchDistMat[(c1, c2)][1]
		
		tempNegLinkages = torch.cuda.FloatTensor(np.zeros(len(impureLinkages), 1))
		if len(impureLinkages) > 0:
			for ctr, (impC1, impC2) in enumerate(impureLinkages):
				if linkAlpha == "min" or linkAlpha == "max":
					tempNegLinkages[ctr] =  torchDistMat[(impC1, impC2)]
				else:
					tempNegLinkages[ctr] =  torchDistMat[(impC1, impC2)][0] / torchDistMat[(impC1, impC2)][1]
		
		negLinkageVals = torch.cat( (negLinkageVals, tempNegLinkages) )

			
		linkCalcTime += time.time() - t1
		
		# Remove merged clusters for list
		activeClusters.remove(c1)
		activeClusters.remove(c2)
		
		t1 = time.time()
		# Update distances of the merged cluster with all remaining clusters
		distMat = updateDistMat_tuple(linkAlpha=linkAlpha, distMat=distMat, currClusterList=activeClusters,
									  newCid=newCid, oldC1=c1, oldC2=c2)
		updateTimeSimple += time.time() - t1

		t1 = time.time()
		torchDistMat = updateDistMat_tuple(linkAlpha=linkAlphaTorch, distMat=torchDistMat, currClusterList=activeClusters,
										   newCid=newCid, oldC1=c1, oldC2=c2)
		
		updateTimeTorch += time.time() - t1
		
		activeClusters.append(newCid)
		
		children[newCid] = (c1, c2)
		pidToParent[c1] = newCid
		pidToParent[c2] = newCid
		
		newCid += 1
		
	# print("\t\tLinkCalc Time:{:.4f}".format(linkCalcTime))
	# print("\t\tMerge Time:{:.4f}".format(mergeTime))
	# print("\t\tupdateTimeTorch Time:{:.4f}".format(updateTimeTorch))
	# print("\t\tupdateTimeSimple Time:{:.4f}".format(updateTimeSimple))
	return posLinkageVals, negLinkageVals

# Run HAC on all points and try to compuate loss
# This version accumulates all edges going between clusters as clustering proceeds and then takes softMax followed by weighted average of
# those edges to compute affinity between two clusters. This has better numerical stabilities.
# Still using separate numerator and denominator numpy matrices to speed up finding best pure merger.
def runHAC_torch_allEdges(origDistMat, origTorchDistMat, linkAlpha, linkAlphaTorch, pidToGtCluster, numPoints, scaleDist, getBestImpure):
	"""
	Runs HAC with ExpLink for given value of linkAlpha and tensors for some pure and impure merger values
	This version accumulates all edges going between clusters as clustering proceeds and then takes softMax followed by weighted average of
	those edges to compute affinity between two clusters. This has better numerical stabilities.
	Still using separate numerator and denominator numpy matrices to speed up finding best pure merger.

	
	:param origDistMat: Numpy Distance Matrix
	:param origTorchDistMat: Torch distance matrix
	:param linkAlpha: Value of alpha parameter of expLink
	:param linkAlphaTorch: Torch variable storing value of alpha parameter of explink
	:param pidToCluster: Dictionary that maps point id to ground-truth cluster id
	:param numPoints: Number of points
	:param threshold: Threshold to use to cut tree. If none then tree is cut using k(given)
	:param scaleDist: Should all distances in the matrix be scaled using largest edge in distance matrix? Useful of preventing numerical overflows
	:param getBestImpure: Should we just add best impure merger to list of negLinkages or should we add all impure mergers
	that are better than best pure merger to list of negLinkages?
	:return: posLinkageVals, negLinkageVals : Both are torch tensors
			posLinkageVals: Linkage value of pure mergers that are worse than some impure merger
			negLinkageVals: Linkage value of impure mergers that are better than some pure merger
	"""
	SoftMax = torch.nn.Softmax(dim=0)
	
	# Map each sub-cluster to its corresponding gt-cluster. This is helpful in finding if two clusters being merged are part of same gt-cluster or not
	subClustToGtClust = copy.deepcopy(pidToGtCluster)
	
	activeClusters = [pid for pid in range(numPoints)]
	newCid 			= numPoints
	
	
	posLinkageVals = torch.cuda.FloatTensor(np.zeros((numPoints-1,1)))
	negLinkageVals = torch.cuda.FloatTensor([]) # Empty tensor
		
	pidToParent = {}
	children 	= {pid: None for pid in activeClusters}
	
	t1 = time.time()
	distMat 	= getDistMat(origDistMat=origDistMat, linkAlpha=linkAlpha, numPoints=numPoints,
							matType="simple", scaleDist=scaleDist, dataStructType="matrix")
	t2 = time.time()
	torchDistMat= getDistMat(origDistMat=origTorchDistMat, linkAlpha=linkAlphaTorch, numPoints=numPoints,
							 matType="pytorch", scaleDist=scaleDist, dataStructType="allEdges")
	t3 = time.time()
	print("\t\tTime taken to get compatible matrices:{:.4f}\t{:.4f}\t{:.4f}".format(t3 - t1, t2 - t1, t3 - t2))
	
	pureMergeExists = True
	mergeTime, updateTimeTorch, updateTimeSimple, linkCalcTime = 0,0,0,0
	while len(activeClusters) > 1:
		
		# Find clusters to merge
		t1 = time.time()
		
		(pureC1, pureC2), impureLinkages = findPureMerger_matVersion(linkAlpha=linkAlpha, distMat=distMat,
																	 currClusterList=activeClusters, subClusToGtClust=subClustToGtClust)
		
		if (pureC1 is None) and (pureC2 is None): # Could not find 2 pure clusters to merge. At this time, we have agglomerated all gt-clusters into separate sub-trees
			pureMergeExists = False
			break  # Break if we don't want to consider loss using just impure agglomerations
		
		mergedC1, mergedC2 = pureC1, pureC2
		subClustToGtClust[newCid] = subClustToGtClust[pureC1]
		assert subClustToGtClust[pureC1] == subClustToGtClust[pureC2]
			
		mergeTime += time.time() - t1
		
		t1 = time.time()
		
		betterNegLinkExists=True   # Setting it to True because for now I want to add all pos linkage to loss
		tempNegLinkages =  torch.cuda.FloatTensor( np.zeros((len(impureLinkages), 1)) )
		for ctr, (impC1, impC2) in enumerate(impureLinkages):
			
			if impC1 != pureC1 and impC1 != pureC2 and impC2 != pureC1 and impC2 != pureC2:
				# Don't consider this impure agglomeration as it is does not involve either c1 or c2
				continue
			else:
				betterNegLinkExists=True
				
			if linkAlpha == "min":
				tempNegLinkages[ctr] =  torch.min(torchDistMat[(impC1, impC2)])
			elif linkAlpha == "max":
				tempNegLinkages[ctr] 	=  torch.max(torchDistMat[(impC1, impC2)])
			else:
				weights = SoftMax(linkAlphaTorch*torchDistMat[(impC1, impC2)])
				tempNegLinkages[ctr] =  torch.sum( weights*torchDistMat[(impC1, impC2)] )
	
		if getBestImpure and tempNegLinkages.shape[0] > 0:
			bestNegLinkage = torch.min(tempNegLinkages)
			negLinkageVals = torch.cat( (negLinkageVals, bestNegLinkage) )
		else:
			negLinkageVals = torch.cat( (negLinkageVals, tempNegLinkages) )
		
		if betterNegLinkExists and (pureC1 is not None) and (pureC2 is not None): # Add this to list of posLinkages only if there is a better impure agglomeration involving c1 and c2
			if linkAlpha == "min":
				posLinkageVals[newCid - numPoints] = torch.min(torchDistMat[(pureC1, pureC2)])
			elif  linkAlpha == "max":
				posLinkageVals[newCid - numPoints] = torch.max(torchDistMat[(pureC1, pureC2)])
			else:
				weights = SoftMax(linkAlphaTorch*torchDistMat[(pureC1,pureC2)])
				posLinkageVals[newCid - numPoints] = torch.sum(weights*torchDistMat[(pureC1,pureC2)])
	
		
			
		t2 = time.time()
		linkCalcTime += t2 - t1
		# Remove merged clusters for list
		activeClusters.remove(mergedC1)
		activeClusters.remove(mergedC2)
		
		t1 = time.time()
		# Update distances of the merged cluster with all remaining clusters
		distMat = updateDistMat_matVersion(linkAlpha=linkAlpha, distMat=distMat, matType="simple",
										   currClusterList=activeClusters, newCid=newCid, oldC1=mergedC1, oldC2=mergedC2)
		
		updateTimeSimple += time.time() - t1

		t1 = time.time()
		# TODO: Can we avoid updating entrie torchDistMatrix? Can we just optimize to just compute linkageVals directly, maybe after the tree is constructed?
		torchDistMat = updateDistMat_allEdges(linkAlpha=linkAlphaTorch, distMat=torchDistMat, matType="pytorch",
											  currClusterList=activeClusters, newCid=newCid, oldC1=mergedC1, oldC2=mergedC2)
		
		updateTimeTorch += time.time() - t1
		
		activeClusters.append(newCid)
		
		children[newCid] = (mergedC1, mergedC2)
		pidToParent[mergedC1] = newCid
		pidToParent[mergedC2] = newCid
		
		newCid += 1
	
	# print("\t\tLinkCalc Time:{:.4f}".format(linkCalcTime))
	# print("\t\tMerge Time:{:.4f}".format(mergeTime))
	# print("\t\tupdateTimeTorch Time:{:.4f}".format(updateTimeTorch))
	# print("\t\tupdateTimeSimple Time:{:.4f}".format(updateTimeSimple))
	return posLinkageVals, negLinkageVals


def runHAC_torch_allEdges_faces(origDistMat, origTorchDistMat, linkAlpha, linkAlphaTorch, pidToGtCluster, numPoints, scaleDist, getBestImpure):
	"""
	This version keeps all edges around to calculate linkage between two clusters even for NP distance matrix.
	This is very slow but avoids having overflow/underflow issues for Face Dataset which has distance of the order of
	1000 units which when exponentiated often results in numerical overflows
	
	Runs HAC with ExpLink for given value of linkAlpha and tensors for some pure and impure merger values
	This version accumulates all edges going between clusters as clustering proceeds and then takes softMax followed by weighted average of
	those edges to compute affinity between two clusters. This has better numerical stabilities.
	Still using separate numerator and denominator numpy matrices to speed up finding best pure merger.

	
	:param origDistMat: Numpy Distance Matrix
	:param origTorchDistMat: Torch distance matrix
	:param linkAlpha: Value of alpha parameter of expLink
	:param linkAlphaTorch: Torch variable storing value of alpha parameter of explink
	:param pidToCluster: Dictionary that maps point id to ground-truth cluster id
	:param numPoints: Number of points
	:param threshold: Threshold to use to cut tree. If none then tree is cut using k(given)
	:param scaleDist: Should all distances in the matrix be scaled using largest edge in distance matrix? Useful of preventing numerical overflows
	:param getBestImpure: Should we just add best impure merger to list of negLinkages or should we add all impure mergers
	that are better than best pure merger to list of negLinkages?
	:return: posLinkageVals, negLinkageVals : Both are torch tensors
			posLinkageVals: Linkage value of pure mergers that are worse than some impure merger
			negLinkageVals: Linkage value of impure mergers that are better than some pure merger
	"""
	SoftMax = torch.nn.Softmax(dim=0)
	
	# Map each sub-cluster to its corresponding gt-cluster. This is helpful in finding if two clusters being merged are part of same gt-cluster or not
	subClustToGtClust = copy.deepcopy(pidToGtCluster)
	
	activeClusters = [pid for pid in range(numPoints)]
	newCid 			= numPoints
	
	
	posLinkageVals = torch.cuda.FloatTensor(np.zeros((numPoints-1,1)))
	negLinkageVals = torch.cuda.FloatTensor([]) # Empty tensor
	
	pidToParent = {}
	children 	= {pid: None for pid in activeClusters}
	
	t1 = time.time()
	distMat 	= getDistMat(origDistMat=origDistMat, linkAlpha=linkAlpha, numPoints=numPoints,
							matType="simple", scaleDist=scaleDist, dataStructType="allEdges")
	t2 = time.time()
	torchDistMat= getDistMat(origDistMat=origTorchDistMat, linkAlpha=linkAlphaTorch, numPoints=numPoints,
							 matType="pytorch", scaleDist=scaleDist, dataStructType="allEdges")
	t3 = time.time()
	print("\t\tTime taken to get compatible matrices:{:.4f}\t{:.4f}\t{:.4f}".format(t3 - t1, t2 - t1, t3 - t2))
	
	pureMergeExists = True
	mergeTime, updateTimeTorch, updateTimeSimple, linkCalcTime = 0,0,0,0
	cache = {}
	while len(activeClusters) > 1:
		
		# Find clusters to merge
		t1 = time.time()
		
		(pureC1, pureC2), impureLinkages = findPureMerger_allEdges(linkAlpha=linkAlpha, distMat=distMat, cache=cache,
																   currClusterList=activeClusters, subClusterToGtCluster=subClustToGtClust)
		
		if (pureC1 is None) and (pureC2 is None): # Could not find 2 pure clusters to merge. At this time, we have agglomerated all gt-clusters into separate sub-trees
			pureMergeExists = False
			break  # Break if we don't want to consider loss using just impure agglomerations
		
		mergedC1, mergedC2 = pureC1, pureC2
		subClustToGtClust[newCid] = subClustToGtClust[pureC1]
		assert subClustToGtClust[pureC1] == subClustToGtClust[pureC2]
		
		mergeTime += time.time() - t1
		
		t1 = time.time()
		
		betterNegLinkExists=True   # Setting it to True because for now I want to add all pos linkage to loss
		tempNegLinkages =  torch.cuda.FloatTensor( np.zeros((len(impureLinkages), 1)) )
		for ctr, (impC1, impC2) in enumerate(impureLinkages):
			
			if impC1 != pureC1 and impC1 != pureC2 and impC2 != pureC1 and impC2 != pureC2:
				# Don't consider this impure agglomeration as it is does not involve either c1 or c2
				continue
			else:
				betterNegLinkExists=True
				
			if linkAlpha == "min":
				tempNegLinkages[ctr] =  torch.min(torchDistMat[(impC1, impC2)])
			elif linkAlpha == "max":
				tempNegLinkages[ctr] 	=  torch.max(torchDistMat[(impC1, impC2)])
			else:
				weights = SoftMax(linkAlphaTorch*torchDistMat[(impC1, impC2)])
				tempNegLinkages[ctr] =  torch.sum( weights*torchDistMat[(impC1, impC2)] )
	
		if getBestImpure and tempNegLinkages.shape[0] > 0:
			bestNegLinkage = torch.min(tempNegLinkages)
			negLinkageVals = torch.cat( (negLinkageVals, bestNegLinkage) )
		else:
			negLinkageVals = torch.cat( (negLinkageVals, tempNegLinkages) )
		
		if betterNegLinkExists and (pureC1 is not None) and (pureC2 is not None): # Add this to list of posLinkages only if there is a better impure agglomeration involving c1 and c2
			if linkAlpha == "min":
				posLinkageVals[newCid - numPoints] = torch.min(torchDistMat[(pureC1, pureC2)])
			elif  linkAlpha == "max":
				posLinkageVals[newCid - numPoints] = torch.max(torchDistMat[(pureC1, pureC2)])
			else:
				weights = SoftMax(linkAlphaTorch*torchDistMat[(pureC1,pureC2)])
				posLinkageVals[newCid - numPoints] = torch.sum(weights*torchDistMat[(pureC1,pureC2)])
	
		
			
		t2 = time.time()
		linkCalcTime += t2 - t1
		# Remove merged clusters for list
		activeClusters.remove(mergedC1)
		activeClusters.remove(mergedC2)
		
		t1 = time.time()
		# Update distances of the merged cluster with all remaining clusters
		distMat = updateDistMat_allEdges(linkAlpha=linkAlpha, distMat=distMat, matType="simple",
										   currClusterList=activeClusters, newCid=newCid, oldC1=mergedC1, oldC2=mergedC2)
		
		updateTimeSimple += time.time() - t1

		t1 = time.time()
		# TODO: Can we avoid updating entrie torchDistMatrix? Can we just optimize to just compute linkageVals directly, maybe after the tree is constructed?
		torchDistMat = updateDistMat_allEdges(linkAlpha=linkAlphaTorch, distMat=torchDistMat, matType="pytorch",
											  currClusterList=activeClusters, newCid=newCid, oldC1=mergedC1, oldC2=mergedC2)
		
		updateTimeTorch += time.time() - t1
		
		activeClusters.append(newCid)
		
		children[newCid] = (mergedC1, mergedC2)
		pidToParent[mergedC1] = newCid
		pidToParent[mergedC2] = newCid
		
		newCid += 1
	
	# print("\t\tLinkCalc Time:{:.4f}".format(linkCalcTime))
	# print("\t\tMerge Time:{:.4f}".format(mergeTime))
	# print("\t\tupdateTimeTorch Time:{:.4f}".format(updateTimeTorch))
	# print("\t\tupdateTimeSimple Time:{:.4f}".format(updateTimeSimple))
	return posLinkageVals, negLinkageVals


################### FIND CLUSTERS TO MERGER USING DATA STRUCTURE STORING DISTANCE BETWEEN CLUSTERS #####################

def findMerger_matVersion(linkAlpha, distMat, currClusterList):
	"""
	Finds and return ids of two clusters to merge according to linkage function given by linkAlpha
	
	:param linkAlpha: Alpha parameter of ExpLink to use for calculating linkage value between clusters
	:param distMat: Pair of matrices storing expLink linkage value b/w 2 clusters separately as numerators and denominators
	:param currClusterList: List of cluster ids which correspond to roots of partially formed trees until this time
	:return: ids of two clusters to merge according to linkage function given by linkAlpha
	"""
	ixgrid  = np.ix_(currClusterList, currClusterList)
	if linkAlpha == "min" or linkAlpha == "max":
		newMat = distMat[ixgrid]
	else:
		newMat = np.nan_to_num(distMat[0][ixgrid]/distMat[1][ixgrid]) # Remove nans
		
	newMat = newMat + np.diag([np.inf for i in currClusterList]) # Add inf along diagonal to avoid it when finding closest clusters to merge
	flatIdx = np.argmin(newMat) # Find smallest value and get row and col corresponding to that value
	
	fc1,fc2 = int(flatIdx/len(currClusterList)), flatIdx%len(currClusterList)
	
	# Now get actual clustersIds using currClustersList as fc1, and fc2 are relative to submatrix on which min was calculated
	fc1 = currClusterList[fc1]
	fc2 = currClusterList[fc2]
	if fc1 < fc2: # Just make sure that first cluster id is larger than second one
		fc1,fc2 = fc2,fc1
	
	return (fc1, fc2)
	
	# Verifications
	# currMin = None
	# clusterPair = None,None
	# for c1,c2 in itertools.combinations(currClusterList, 2):
	# 	if c1 <= c2:
	# 		c1,c2  = c2, c1
	#
	# 	if linkAlpha == 'min' or linkAlpha == 'max':
	# 		tempDist = distMat[c1][c2]
	# 	else:
	# 		try:
	# 			tempDist = distMat[0][c1][c2]/distMat[1][c1][c2]
	# 		except Exception as e:
	# 			print(c1,c2,"\n")
	# 			print(distMat,"\n")
	# 			print(currClusterList,"\n")
	# 			raise e
	#
	# 	if currMin is None or tempDist < currMin :
	# 		currMin     = tempDist
	# 		clusterPair = c1,c2
	#
	#
	# if fastMin != currMin or fc1 != clusterPair[0] or fc2 != clusterPair[1]:
	# 	print(fc1, fc2, clusterPair)
	# 	print(fastMin, currMin)
	# 	print(newMat.shape, len(currClusterList), newMat)
	# 	# print(newMat2.shape, newMat2)
	# 	# print(distMat[0][ixgrid])
	# 	# print(distMat[1][ixgrid])
	# return clusterPair
	pass

def findMerger_allEdges(linkAlpha, distMat, currClusterList, cache):
	"""
	Returns best 2 sub-clusters to merge and linkage value of that merger

	:param linkAlpha: Alpha parameter of ExpLink to use for calculating linkage value between clusters
	:param distMat: Dictionary that maps a pair of cluster ids to (numerator,denominator) used to find expLink linkage b/w 2 clusters
	:param currClusterList: List of cluster ids which correspond to roots of partially formed trees until this time
	
	:return: Returns best 2 sub-clusters to merge and linkage value of that merger
	"""
	
	currMin = None
	bestClusterPair = None, None
	for c1, c2 in itertools.combinations(currClusterList, 2):
		if (c1,c2) in cache:
			tempDist = cache[(c1,c2)]
		elif (c2,c1) in cache:
			tempDist = cache[(c2,c1)]
		else:
			tempDist = calc_linkage_numpy(linkAlpha, distMat[(c1, c2)])
			cache[(c1,c2)] = tempDist
			cache[(c2,c1)] = tempDist
			
		if (currMin is None or tempDist < currMin):
			currMin 	= tempDist
			bestClusterPair = (c1, c2)
	
	return bestClusterPair, currMin

def findPureMerger_tuple(linkAlpha, distMat, currClusterList, subClusterToGtCluster):
	"""
	Returns best 2 sub-clusters(of same gt-clusters to merge, along with other mergers between
	two sub-clusters(belonging to two different sub-clusters) which are better than that

	
	:param linkAlpha: Alpha parameter of ExpLink to use for calculating linkage value between clusters
	:param distMat: Dictionary that maps a pair of cluster ids to (numerator,denominator) used to find expLink linkage b/w 2 clusters
	:param currClusterList: List of cluster ids which correspond to roots of partially formed trees until this time
	:param subClusterToGtCluster: Maps each current cluster to its pure gt-cluster. Each current cluster can map to just 1 ground truth cluster because
	during training we only perform pure mergers in function runHAC_torch_allEdges and runHAC_torch_num_den
	
	:return: Returns best 2 sub-clusters(of same gt-clusters to merge, along with other mergers between
	two sub-clusters(belonging to two different sub-clusters) which are better than that
	"""
	
	currPureMin = None
	pureClusterPair = None, None
	for c1, c2 in itertools.combinations(currClusterList, 2):
		if linkAlpha == 'min' or linkAlpha == 'max':
			tempDist = distMat[(c1, c2)]
		else:
			tempDist = distMat[(c1, c2)][0] / distMat[(c1, c2)][1]
		
		isPureMerge = subClusterToGtCluster[c1] == subClusterToGtCluster[c2] # Are c1 and c2 two sub-clusters of the same gt-cluster
		
		if (currPureMin is None or tempDist < currPureMin) and isPureMerge:
			currPureMin 	= tempDist
			pureClusterPair = c1, c2
	
	if currPureMin is None:
		return (None, None), []
	
	impureMerges = []
	for c1, c2 in itertools.combinations(currClusterList, 2):
		if linkAlpha == 'min' or linkAlpha == 'max':
			tempDist = distMat[(c1, c2)]
		else:
			tempDist = distMat[(c1, c2)][0] / distMat[(c1, c2)][1]
		
		# No need to check if c1 and c2 are sub-clusters of same gt-cluster as we already picked best possible same class clusters to merge
		# Now, there could be agglomerations that are between two clusters from different classes that might be better than currPureMin
		if tempDist < currPureMin:
			impureMerges += [(c1,c2)]
			
	return pureClusterPair, impureMerges

def findPureMerger_allEdges(linkAlpha, distMat, currClusterList, subClusterToGtCluster, cache):
	"""
	Returns best 2 sub-clusters(of same gt-clusters to merge, along with other mergers between
	two sub-clusters(belonging to two different sub-clusters) which are better than that

	
	:param linkAlpha: Alpha parameter of ExpLink to use for calculating linkage value between clusters
	:param distMat: Dictionary that maps a pair of cluster ids to (numerator,denominator) used to find expLink linkage b/w 2 clusters
	:param currClusterList: List of cluster ids which correspond to roots of partially formed trees until this time
	:param subClusterToGtCluster: Maps each current cluster to its pure gt-cluster. Each current cluster can map to just 1 ground truth cluster because
	during training we only perform pure mergers in function runHAC_torch_allEdges and runHAC_torch_num_den
	
	:return: Returns best 2 sub-clusters(of same gt-clusters to merge, along with other mergers between
	two sub-clusters(belonging to two different sub-clusters) which are better than that
	"""
	
	currPureMin = None
	pureClusterPair = None, None
	for c1, c2 in itertools.combinations(currClusterList, 2):
		if (c1,c2) in cache:
			tempDist = cache[(c1,c2)]
		elif (c2,c1) in cache:
			tempDist = cache[(c2,c1)]
		else:
			tempDist = calc_linkage_numpy(linkAlpha, distMat[(c1, c2)])
			
			cache[(c1,c2)] = tempDist
			cache[(c2,c1)] = tempDist
			
		isPureMerge = subClusterToGtCluster[c1] == subClusterToGtCluster[c2] # Are c1 and c2 two sub-clusters of the same gt-cluster
		
		if (currPureMin is None or tempDist < currPureMin) and isPureMerge:
			currPureMin 	= tempDist
			pureClusterPair = c1, c2
	
	if currPureMin is None:
		return (None, None), []
	
	impureMerges = []
	for c1, c2 in itertools.combinations(currClusterList, 2):
		if (c1,c2) in cache:
			tempDist = cache[(c1,c2)]
		elif (c2,c1) in cache:
			tempDist = cache[(c2,c1)]
		else:
			tempDist = calc_linkage_numpy(linkAlpha, distMat[(c1, c2)])
			
			cache[(c1,c2)] = tempDist
			cache[(c2,c1)] = tempDist
		
		# No need to check if c1 and c2 are sub-clusters of same gt-cluster as we already picked best possible same class clusters to merge
		# Now, there could be agglomerations that are between two clusters from different classes that might be better than currPureMin
		if tempDist < currPureMin:
			impureMerges += [(c1,c2)]
			
	return pureClusterPair, impureMerges

def findPureMerger_matVersion(linkAlpha, distMat, currClusterList, subClusToGtClust):
	"""
	Returns best 2 sub-clusters(of same gt-clusters to merge, along with other mergers between
	two sub-clusters(belonging to two different sub-clusters) which are better than that
	:param linkAlpha:
	:param distMat:
	:param currClusterList:
	:param subClusToGtClust:
	:return:
	"""

	currPureMin = None
	pureClusterPair = None, None
	for c1, c2 in itertools.combinations(currClusterList, 2):
		isPureMerge = subClusToGtClust[c1] == subClusToGtClust[c2] # Are c1 and c2 two sub-clusters of the same gt-cluster
		if not isPureMerge:
			continue
			
		if linkAlpha == 'min' or linkAlpha == 'max':
			tempDist = distMat[c1, c2]
		else:
			tempDist = distMat[0][c1, c2] / distMat[1][c1, c2]
		
		
		
		if (currPureMin is None or tempDist < currPureMin) and isPureMerge:
			currPureMin 	= tempDist
			pureClusterPair = c1, c2
	
	if currPureMin is None:
		return (None, None), []
	
	impureMerges = []
	
	for c1, c2 in itertools.combinations(currClusterList, 2):
		if linkAlpha == 'min' or linkAlpha == 'max':
			tempDist = distMat[c1, c2]
		else:
			tempDist = distMat[0][c1, c2] / distMat[1][c1, c2]
		
		# No need to check if c1 and c2 are sub-clusters of same gt-cluster as we already picked best possible same class clusters to merge
		# Now, there could be agglomerations that are between two clusters from different classes that might be better than currPureMin
		if tempDist < currPureMin:
			impureMerges += [(c1,c2)]
			
	return pureClusterPair, impureMerges

########################################################################################################################

################################ UPDATE DATA STRUCTURE STORING DISTANCE BETWEEN CLUSTERS ###############################

def updateDistMat_tuple(linkAlpha, distMat, currClusterList, newCid, oldC2, oldC1):

	for cid in currClusterList:
		if isinstance(linkAlpha,str) and linkAlpha == 'min':
			distMat[(cid, newCid)] = min(distMat[(cid, oldC1)], distMat[(cid, oldC2)])
			distMat[(newCid, cid)] = min(distMat[(oldC1, cid)], distMat[(oldC2, cid)])
		elif isinstance(linkAlpha,str) and  linkAlpha == 'max':
			distMat[(cid, newCid)] = max(distMat[(cid, oldC1)], distMat[(cid, oldC2)])
			distMat[(newCid, cid)] = max(distMat[(oldC1, cid)], distMat[(oldC2, cid)])
		elif (isinstance(linkAlpha, torch.autograd.Variable)) or isinstance(linkAlpha, float) or isinstance(linkAlpha, int):
			distMat[(cid, newCid)] = ( distMat[(cid, oldC1)][0] + distMat[(cid, oldC2)][0], distMat[(cid, oldC1)][1] + distMat[(cid, oldC2)][1] )
			distMat[(newCid, cid)] = ( distMat[(oldC1, cid)][0] + distMat[(oldC2, cid)][0], distMat[(oldC1, cid)][1] + distMat[(oldC2, cid)][1] )
		else:
			raise Exception("Invalid value for linkAlpha = {}".format(linkAlpha))

	return distMat

def updateDistMat_allEdges(linkAlpha, distMat, currClusterList, newCid, oldC2, oldC1, matType="pytorch"):

	if matType == "simple":
		for cid in currClusterList:
			distMat[(newCid, cid)] = np.concatenate( (distMat[(cid, oldC1)], distMat[(cid, oldC2)]) )
			distMat[(cid, newCid)] = np.concatenate( (distMat[(oldC1, cid)], distMat[(oldC2, cid)]) )
			
	elif matType == "pytorch":
		for cid in currClusterList:
			distMat[(newCid, cid)] = torch.cat( (distMat[(cid, oldC1)], distMat[(cid, oldC2)]) )
			distMat[(cid, newCid)] = torch.cat( (distMat[(oldC1, cid)], distMat[(oldC2, cid)]) )
	else:
		raise Exception("Invalid matType={}".format(matType))

	return distMat

def updateDistMat_matVersion(linkAlpha, distMat, matType, currClusterList, newCid, oldC2, oldC1):

	if isinstance(linkAlpha,str) and linkAlpha == 'min':
		if matType == "simple":
			distMat[newCid,:] = np.minimum(distMat[oldC1,:], distMat[oldC2,:])
			distMat[:,newCid] = np.minimum(distMat[:,oldC1], distMat[:,oldC2])
		elif matType == "pytorch":
			distMat[newCid,:] = torch.min(distMat[oldC1,:], distMat[oldC2,:])
			distMat[:,newCid] = torch.min(distMat[:,oldC1], distMat[:,oldC2])
		else:
			raise Exception("Invalid matrix type= {}".format(matType))
	elif isinstance(linkAlpha,str) and  linkAlpha == 'max':
		if matType == "simple":
			distMat[newCid,:] = np.maximum(distMat[oldC1,:], distMat[oldC2,:])
			distMat[:,newCid] = np.maximum(distMat[:,oldC1], distMat[:,oldC2])
		elif matType == "pytorch":
			distMat[newCid,:] = torch.max(distMat[oldC1,:], distMat[oldC2,:])
			distMat[:,newCid] = torch.max(distMat[:,oldC1], distMat[:,oldC2])
		else:
			raise Exception("Invalid matrix type= {}".format(matType))
	elif (isinstance(linkAlpha, torch.autograd.Variable)) or isinstance(linkAlpha, float) or isinstance(linkAlpha, int):
		if matType == "simple":
			distMat[0][newCid,:] = distMat[0][oldC1,:] + distMat[0][oldC2,:]
			distMat[0][:,newCid] = distMat[0][:,oldC1] + distMat[0][:,oldC2]
			
			distMat[1][newCid,:] = distMat[1][oldC1,:] + distMat[1][oldC2,:]
			distMat[1][:,newCid] = distMat[1][:,oldC1] + distMat[1][:,oldC2]
		elif matType == "pytorch":
			distMat[0][newCid,:] = distMat[0][oldC1,:] + distMat[0][oldC2,:]
			distMat[0][:,newCid] = distMat[0][:,oldC1] + distMat[0][:,oldC2]

			distMat[1][newCid,:] = distMat[1][oldC1,:] + distMat[1][oldC2,:]
			distMat[1][:,newCid] = distMat[1][:,oldC1] + distMat[1][:,oldC2]
			
		else:
			raise Exception("Invalid matrix type= {}".format(matType))
	else:
		raise Exception("Invalid value for linkAlpha = {}".format(linkAlpha))
	
	return distMat

########################################################################################################################

################################ GET DATA STRUCTURE TO STORE DISTANCE BETWEEN CLUSTERS #################################

def getDistMat(origDistMat, numPoints, linkAlpha, matType, dataStructType ,scaleDist):
	"""
	Return distance matrix used by HAC
	:param origDistMat:
	:param numPoints:
	:param linkAlpha:
	:param matType:
	:param scaleDist:
	:return:
	"""
	# Dividing each edge by the largest edge to make the computation scale invariant
	if scaleDist:
		print("\n\n\nScaling all distances using largest edges....\n\n\n\n")
		if matType == "simple":
			largestEdge = np.amax(origDistMat,axis=None)
			assert largestEdge != np.inf
			assert largestEdge != np.nan
			origDistMat = origDistMat/largestEdge
		elif matType == "pytorch":
			largestEdge = torch.max(origDistMat)
			origDistMat = origDistMat/largestEdge
		else:
			raise Exception("Invalid matrix type:{}".format(matType))
	else:
		pass
		# print("\n\n\n NOT NOT NOT NOT Scaling distances....\n\n\n\n")
	
	if dataStructType == "tuple":
		''' Distance matrix is a dictionary. Key is pair of cluster ids, and value is either (numerator and denominator)
		 used to calculate linkage value or value is linkage value itself.
		'''
		return getDistMatrix_tuple(origDistMat=origDistMat, numPoints=numPoints, linkAlpha=linkAlpha, matType=matType)
	elif dataStructType == "matrix":
		''' If linkAlpha is a number, then distance matrix is actually 2 matrices.
		(i,j) of 1st matrix is numerator and of 2nd matrix is denominator for linkage value between cluster i and j.
		If linkAlpha is min or max then there is only one matrix with (i,j) storing linkage value between i and j
		'''
		return getDistMatrix_matVersion(origDistMat=origDistMat, numPoints=numPoints, linkAlpha=linkAlpha, matType=matType)
	elif dataStructType == "allEdges":
		'''
		distance Matrix is a dictionary. Key is pair of cluster ids, and value is list of all edges between cluster i & j
		'''
		return getDistMatrix_allEdges(origDistMat=origDistMat, numPoints=numPoints, matType=matType)
	else:
		raise Exception("Invalud dataStructType={}".format(dataStructType))
	
def getDistMatrix_tuple(origDistMat, numPoints, linkAlpha, matType):
	
	# Brute force way
	distMat = {}
	for pid1, pid2 in itertools.combinations(range(numPoints),2):
		tempDist  = origDistMat[pid1][pid2]
		if isinstance(linkAlpha,str) and ( linkAlpha == 'min' or linkAlpha == 'max' ):
			distMat[(pid1,pid2)] = tempDist
			distMat[(pid2,pid1)] = tempDist
		elif matType == "simple" and (isinstance(linkAlpha, float) or isinstance(linkAlpha, int)):
			distMat[(pid1, pid2)] = np.exp(linkAlpha*tempDist)*tempDist, math.exp(linkAlpha*tempDist)
			distMat[(pid2, pid1)] = np.exp(linkAlpha*tempDist)*tempDist, math.exp(linkAlpha*tempDist)
		elif matType == "pytorch" and ( isinstance(linkAlpha, float) or isinstance(linkAlpha, int) or isinstance(linkAlpha, torch.autograd.Variable)):
			distMat[(pid1, pid2)] = torch.exp(linkAlpha*tempDist)*tempDist, torch.exp(linkAlpha*tempDist)
			distMat[(pid2, pid1)] = torch.exp(linkAlpha*tempDist)*tempDist, torch.exp(linkAlpha*tempDist)
		else:
			raise Exception("Invalid linkage alpha :{} of type:{} or matType:{} of type:{}".format(linkAlpha,type(linkAlpha), matType, type(matType)))
		
	return distMat

def getDistMatrix_allEdges(origDistMat, numPoints, matType):
	distMat = {}
	if matType == "simple":
		for pid1, pid2 in itertools.combinations(range(2*numPoints-1),2):
			if pid1 < numPoints and pid2 < numPoints:
				tempDist  = origDistMat[pid1][pid2]
				distMat[(pid1, pid2)] = np.array([tempDist])
				distMat[(pid2, pid1)] = np.array([tempDist])
				# print("getDist",tempDist, type(distMat[(pid1,pid2)]),distMat[(pid1,pid2)] )
			else:
				distMat[(pid1, pid2)] = np.array([])
				distMat[(pid2, pid1)] = np.array([])
		
	elif matType == "pytorch":
		
		for pid1, pid2 in itertools.combinations(range(2*numPoints-1),2):
			if pid1 < numPoints and pid2 < numPoints:
				tempDist  = origDistMat[pid1][pid2]
				distMat[(pid1, pid2)] = tempDist.view(1)
				distMat[(pid2, pid1)] = tempDist.view(1)
			else:
				distMat[(pid1, pid2)] = []
				distMat[(pid2, pid1)] = []
	else:
		raise Exception("Invalud matType={}".format(matType))
	
	return distMat

def getDistMatrix_matVersion(origDistMat, numPoints, linkAlpha, matType):
	

	if isinstance(linkAlpha, str) and (linkAlpha == "min" or linkAlpha == "max"):
		if matType == "simple":
			idxList = list(range(numPoints))
			ixgrid  = np.ix_(idxList, idxList)
			distMatSimple = np.zeros((2*numPoints - 1, 2*numPoints-1))
			distMatSimple[ixgrid] = origDistMat
			return distMatSimple
		elif matType == "pytorch":
			idxList = list(range(numPoints))
			ixgrid  = np.ix_(idxList, idxList)
			if isinstance(origDistMat, torch.cuda.FloatTensor):
				distMatSimple = torch.cuda.FloatTensor( np.zeros((2*numPoints - 1, 2*numPoints-1)) )
			else:
				distMatSimple = torch.FloatTensor(np.zeros((2*numPoints - 1, 2*numPoints-1)))
			
			distMatSimple[ixgrid] = origDistMat
			return distMatSimple
		else:
			raise Exception("Invalid linkage alpha :{} of type:{} or matType:{} of type:{}".format(linkAlpha,type(linkAlpha), matType, type(matType)))
	
	elif matType == "simple" and (isinstance(linkAlpha, float) or isinstance(linkAlpha, int)):
	
		distMatNum = np.zeros((2*numPoints - 1, 2*numPoints-1))
		distMatDen = np.zeros((2*numPoints - 1, 2*numPoints-1))

		idxList = list(range(numPoints))
		ixgrid  = np.ix_(idxList, idxList)

		distMatDen[ixgrid] = np.exp(origDistMat*linkAlpha)
		distMatNum[ixgrid] = np.multiply(distMatDen[ixgrid], origDistMat)
		return distMatNum, distMatDen
	elif matType == "pytorch" and ( isinstance(linkAlpha, float) or isinstance(linkAlpha, int) or isinstance(linkAlpha, torch.autograd.Variable)):

		if isinstance(origDistMat, torch.cuda.FloatTensor):
			distMatNum  = torch.cuda.FloatTensor( 2*numPoints - 1, 2*numPoints-1 ).fill_(0)
			distMatDen  = torch.cuda.FloatTensor( 2*numPoints - 1, 2*numPoints-1 ).fill_(0)
		else:
			distMatNum = torch.zeros(2*numPoints - 1, 2*numPoints-1)
			distMatDen = torch.zeros(2*numPoints - 1, 2*numPoints-1)

		idxList = list(range(numPoints))
		ixgrid  = np.ix_(idxList, idxList)
		
		distMatDen[ixgrid] = torch.exp(origDistMat*linkAlpha)
		distMatNum[ixgrid] = torch.mul(distMatDen[ixgrid], origDistMat)
		
		return distMatNum, distMatDen
	else:
		raise Exception("Invalid linkage alpha :{} of type:{} or matType:{} of type:{}".format(linkAlpha,type(linkAlpha), matType, type(matType)))

########################################################################################################################

def write_tree(treeFilename, pidToCluster, children, pidToParent):

	root = list(pidToCluster.keys())[0]
	while root in pidToParent :
		root = pidToParent[root]

	with open(treeFilename,"w") as writer:
		for nodeId in children:
			if children[nodeId] is None: continue

			child0,child1 	= children[nodeId]
			child0Label 	= pidToCluster[child0] if child0 in pidToCluster else "None"
			child1Label 	= pidToCluster[child1] if child1 in pidToCluster else "None"

			writer.write("{}\t{}\t{}\n".format(child0, nodeId, child0Label))
			writer.write("{}\t{}\t{}\n".format(child1, nodeId, child1Label))

		writer.write("{}\tNone\tNone\n".format(root))

def computeDendPurity(pidToCluster, children, pidToParent):

	dendPurity = 0
	XCLUSTER_ROOT = os.getenv("XCLUSTER_ROOT")
	filenum = time.time()
	treeFilename = "{}/perchTree_{}.tree".format(XCLUSTER_ROOT, filenum)
	write_tree(treeFilename=treeFilename, pidToCluster=pidToCluster, children=children, pidToParent=pidToParent)

	assert os.path.isfile(treeFilename)

	command =  "cd $XCLUSTER_ROOT && source bin/setup.sh && pwd && "
	command += "sh bin/util/score_tree.sh {} algo data 24 None > treeResult_{}".format(treeFilename,filenum)
	# print("Executing command = {}".format(command))
	os.system(command)

	resultFileName = "{}/treeResult_{}".format(XCLUSTER_ROOT, filenum)
	with open(resultFileName,"r") as reader:
		for line in reader:
			algo, data, dendPurity = line.split()
			dendPurity = float(dendPurity)
			break
	
	
	command = "rm {} && rm {}".format(treeFilename, resultFileName)
	# print("Removing files:{}".format(command))
	os.system(command)
	assert not os.path.isfile(treeFilename)
	assert not os.path.isfile(resultFileName)
	return dendPurity

# Returns flat clustering for tree built so far
def getPidToPredClusters(pidToParent, numPoints):
	pidToPredCluster = {}
	for pid in range(numPoints):
		currPid = pid
		parentPid = currPid
		while currPid in pidToParent:
			parentPid = pidToParent[currPid]

			currPid = parentPid

		pidToPredCluster[pid] = parentPid
	return pidToPredCluster

if __name__ == "__main__":
	pass
