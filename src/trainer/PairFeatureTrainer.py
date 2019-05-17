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
import pprint
import numpy as np
import itertools
import random

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import torch
from torch.autograd import Variable

from eval.finalEval import eval_all_data
from eval.evalPairFeat import eval_model_pair_feat, get_affinity_mat
from eval.threshold import choose_threshold
from utils.plotting import plot_scores, write_scores_comb
from utils.basic_utils import load_canopy_data_splits, create_logger

from models import create_new_model
from hier_clust.expLink import runHAC_torch_allEdges
from models.linearClassifier import AvgLinearClassifier

from trainer.BaseTrainer import BaseTrainer


'''
Data is organized in canopies using a dictionary. Each canopy has a canopyId which acts as key to dictionary containing canopyData" \
For each canopy, there are 3 dictionaries:
pidToCluster  : Dict with key as point ids and value as cluster ids
clusterToPids : Dict with cluster ids as key and value as list containing ids of points in that cluster
pairFeatures    : Dict with key as (pid1,pid2) tuple and value as feature vector for pair (pid1, pid2)
'''

class PairFeatureTrainer(BaseTrainer):
	"""docstring for Pair Feature Trainer"""

	def __init__(self, config):
		super(PairFeatureTrainer, self).__init__(config)
		
		self.config = config
		self.model 	= create_new_model(self.config)
		self.trainCanopies, self.testCanopies, self.devCanopies = load_canopy_data_splits(self.config)
		self.logger = create_logger(config=config, logFile=config.logFile, currLogger=None)
		
		self.optimizer = None
		self.resetOptimizer()

		if self.config.useGPU:
			self.logger.info("Shifting model to cuda because GPUs are available!")
			self.model = self.model.cuda()
		
		self.logger.info("Successfully initialized model trainer...")
		self.logger.info(str(self))
		self.config.save_config(self.config.resultDir)
		
		self.evalFunc = eval_model_pair_feat
		
	def __str__(self):
		printDict = {}
		
		printDict["numTrainCanopies"] 			= len(self.trainCanopies)
		printDict["trainCanopyClusters"] 		= [canopyId for canopyId in self.trainCanopies]
		printDict["trainCanopyClustersSizes"] 	= [len(self.trainCanopies[canopyId]["clusterToPids"]) for canopyId in self.trainCanopies]
		printDict["trainCanopyNumPoints"] 		= [len(self.trainCanopies[canopyId]["pidToCluster"]) for canopyId in self.trainCanopies]
		
		printDict["numTestCanopies"] 			= len(self.testCanopies)
		printDict["testCanopyClusters"] 		= [canopyId for canopyId in self.testCanopies]
		printDict["testCanopyClustersSizes"] 	= [len(self.testCanopies[canopyId]["clusterToPids"]) for canopyId in self.testCanopies]
		printDict["testCanopyNumPoints"] 		= [len(self.testCanopies[canopyId]["pidToCluster"]) for canopyId in self.testCanopies]
		
		printDict["numDevCanopies"] 			= len(self.devCanopies)
		printDict["devCanopyClusters"] 			= [canopyId for canopyId in self.devCanopies]
		printDict["devCanopyClusterSizes"] 		= [len(self.devCanopies[canopyId]["clusterToPids"]) for canopyId in self.devCanopies]
		printDict["devCanopyNumPoints"] 		= [len(self.devCanopies[canopyId]["pidToCluster"]) for canopyId in self.devCanopies]
		
		printDict["Model Params"]  = str(self.model)
		printDict["Optimizer"]	   = str(self.optimizer) + "\n" + str(self.optimizer.param_groups) + "\n"
		
		printDict["Config"] = pprint.pformat(self.config.to_json())
		return pprint.pformat(printDict)
	
	def printModelWeights(self):
		
		weightStr = self.model.getWeightStr()
		self.logger.info(weightStr)
	
	def train(self):
		
		if self.config.trainObj.startswith("linkage"):
			self.train_general()
		elif self.config.trainObj.startswith("triplet"):
			self.train_general()
		elif "Within" in self.config.trainObj and "Across" in self.config.trainObj:
			self.train_general()
		else:
			self.logger.info("Invalid training objective::{}".format(self.config.trainObj))
			raise Exception("Invalid training objective::{}".format(self.config.trainObj))

	# Function for training on objectives: <bestWithin,allWithin,mstWithin> +<bestAcross,allAcross>
	def train_general(self):

		allLosses = {"train":{},"dev":{},"test":{}}
		allScores = {"train":{},"dev":{},"test":{}}
		
		if self.config.evalBeforeTrain:
			threshDict = {}
			for method in self.config.inferenceMethods:
				if method == "connComp" or method.endswith("@t"):
					threshDict[method] = choose_threshold(trainer=self,infMethod=method, epoch=-1)
	
			allScores["train"][-1], allScores["test"][-1], allScores["dev"][-1] = eval_all_data(trainer=self,threshDict=threshDict)
			
		for epoch in range(self.config.numEpoch):
			
			start = time.time()
			totalWithinLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
			totalWithinLoss = Variable(totalWithinLoss, requires_grad=True)
			
			totalAcrossLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
			totalAcrossLoss = Variable(totalAcrossLoss, requires_grad=True)
			
			withinTime, acrossTime = 0., 0.
			for canopyId in sorted(self.trainCanopies):
				canopyStart = time.time()
				
				if self.config.trainObj.startswith("linkage"):
					try:
						linkAlpha = float(self.config.trainObj.split("_")[-1])
					except ValueError:
						linkAlpha = self.config.trainObj.split("_")[-1]
					
					canopy = self.trainCanopies[canopyId]
					withinLoss, acrossLoss = self.calcOverallHACLoss(canopy=canopy, linkAlpha=linkAlpha)
					totalLoss = withinLoss + acrossLoss
				elif self.config.trainObj.startswith("triplet"):
					canopy = self.trainCanopies[canopyId]
					withinLoss, acrossLoss = self.calcTripletLoss(canopy=canopy)
					totalLoss = withinLoss + acrossLoss
				else:
					
					t1 = time.time()
					######################### Loss for within cluster Edges #####################################
					if self.config.trainObj.startswith("bestWithin"):
						withinLoss, _ = self.calcWithinEdgeLoss(canopy=self.trainCanopies[canopyId], onlyBestWithinEdges=True, useAvgWeights=False)
					elif self.config.trainObj.startswith("allWithin"):
						withinLoss, _ = self.calcWithinEdgeLoss(canopy=self.trainCanopies[canopyId], onlyBestWithinEdges=False, useAvgWeights=False)
					elif self.config.trainObj.startswith("mstWithin"):
						withinLoss, _ = self.calcMSTLoss(canopy=self.trainCanopies[canopyId])
					else:
						self.logger.info("Invalid training objective:{}".format(self.config.trainObj))
						raise Exception("Invalid training objective:{}".format(self.config.trainObj))
					withinTime += time.time() - t1
					
					######################### Loss for Across Cluster Edges #####################################
					t1 = time.time()
					if self.config.trainObj.endswith("bestAcross"):
						acrossLoss = self.calcAcrossEdgeLoss(canopy=self.trainCanopies[canopyId], onlyBestAcrossEdges=True, useAvgWeights=False)
					elif self.config.trainObj.endswith("allAcross"):
						acrossLoss = self.calcAcrossEdgeLoss(canopy=self.trainCanopies[canopyId], onlyBestAcrossEdges=False, useAvgWeights=False)
					else:
						self.logger.info("Invalid training objective:{}".format(self.config.trainObj))
						raise Exception("Invalid training objective:{}".format(self.config.trainObj))
					acrossTime += time.time() - t1
					
					totalLoss = withinLoss + acrossLoss
				
				self.optimizer.zero_grad()
				totalLoss.backward()
				self.optimizer.step()
				
				if isinstance(self.model, AvgLinearClassifier):
					self.model.updateAvgWeights()  # Update average weights after updating current weights
				
				canopyEnd = time.time()
				totalWithinLoss = totalWithinLoss + withinLoss
				totalAcrossLoss = totalAcrossLoss + acrossLoss
				
				self.logger.info("\nEpoch:{}\tCanopy:{}\tTime\t{:.4f}".format(epoch, canopyId, canopyEnd - canopyStart))
				self.logger.info("\tWithinLoss\t{:.3f}".format(withinLoss.data[0]))
				self.logger.info("\tAcrossLoss\t{:.3f}".format(acrossLoss.data[0]))

			totalWithinLoss = totalWithinLoss.data.cpu().numpy()[0]
			totalAcrossLoss = totalAcrossLoss.data.cpu().numpy()[0]
			
			totalWithinLoss = totalWithinLoss/len(self.trainCanopies) if len(self.trainCanopies) > 0. else 0.
			totalAcrossLoss = totalAcrossLoss/len(self.trainCanopies) if len(self.trainCanopies) > 0. else 0.
		
			allLosses["train"][epoch] = (totalWithinLoss, totalAcrossLoss)
			allLosses["dev"][epoch]   = self.calcTotalLoss(self.devCanopies, useAvgWeights=True)
			if hasattr(self.model, "linkAlpha"):
				linkAlpha = self.model.linkAlpha
				allLosses["test"][epoch] = float(linkAlpha.data.cpu().numpy()[0])
			else:
				allLosses["test"][epoch] = 0.
			
			end = time.time()
			self.logger.info("\nEpoch:{}\tTime taken::\t{:.3f} = {:.3f} + {:.3f}".format(epoch, end - start, withinTime, acrossTime))
			self.logger.info("\tTotalWithinLoss\t{:.3f}".format(totalWithinLoss))
			self.logger.info("\tTotalAcrossLoss\t{:.3f}".format(totalAcrossLoss))
			self.logger.info("\tAlpha:{:.4f}".format(allLosses["test"][epoch]))
			if "{}".format(allLosses["test"][epoch]) == "nan":
				self.logger.info("This run is wasted as linkAlpha has value nan")
				self.plotAndWriteData(allLosses, allScores)
				return False
				
			if (epoch + 1) % self.config.epochToEval == 0:
				threshDict = {}
				for method in self.config.inferenceMethods:
					if method == "connComp" or method.endswith("@t"):
						threshDict[method] = choose_threshold(trainer=self,infMethod=method, epoch=-1)
		
				allScores["train"][-1], allScores["test"][-1], allScores["dev"][-1] = eval_all_data(trainer=self,threshDict=threshDict)
				self.plotAndWriteData(allLosses=allLosses, allScores=allScores)
			
			if self.stopTraining(epoch=epoch, allLosses=allLosses):
				break
				
		self.plotAndWriteData(allLosses=allLosses, allScores=allScores)
		return True
	
	def stopTraining(self, epoch, allLosses):
		
		numEpToAvg = self.config.numEpToAvg
		if epoch <= numEpToAvg: return False
		
		dtype = "dev" if len(self.devCanopies) > 0 else "train"  # Which canopies to determine convergence...
		if self.config.threshold is None:  # Used when using ranking style training, or when trainObj is linkage_{}
			avgLoss = np.mean([allLosses[dtype][e] for e in range(epoch - 1)][-1 * numEpToAvg:])  # Average loss over last numEpToAvg epoch...
			
			if avgLoss == 0 or abs(allLosses[dtype][epoch] - avgLoss) / abs(avgLoss) < self.config.epsilon:
				self.logger.info("{} loss has saturated so stopping training...{}\t{}".format(dtype, abs(allLosses[dtype][epoch] - avgLoss), abs(avgLoss)))
				return True
		
		else:  # Used when train_general function where within and across cluster losses are calculated separately
			avgLossWithin = np.mean([allLosses[dtype][e][0] for e in range(epoch - 1)][-1 * numEpToAvg:])  # Average loss over last numEpToAvg epoch...
			avgLossAcross = np.mean([allLosses[dtype][e][1] for e in range(epoch - 1)][-1 * numEpToAvg:])  # Average loss over last numEpToAvg epoch...
			
			if (avgLossWithin == 0 or abs(allLosses[dtype][epoch][0] - avgLossWithin) / abs(avgLossWithin) < self.config.epsilon) and \
					(avgLossAcross == 0 or abs(allLosses[dtype][epoch][1] - avgLossAcross) / abs(avgLossAcross) < self.config.epsilon):
				self.logger.info("{} within loss has saturated so stopping training...{}\t{}".format(dtype, abs(allLosses[dtype][epoch][0] - avgLossWithin), abs(avgLossWithin)))
				self.logger.info("{} across loss has saturated so stopping training...{}\t{}".format(dtype, abs(allLosses[dtype][epoch][1] - avgLossAcross), abs(avgLossAcross)))
				return True
		
		return False
	
	def plotAndWriteData(self, allLosses, allScores):
		
		if self.config.makeScorePlots:
			plot_scores(allLosses=allLosses, allScores=allScores, currResultDir=self.config.resultDir, xlabel="Epoch")
		else:
			plot_scores(allLosses=allLosses, allScores={"train":{}, "test":{}, "dev":{}}, currResultDir=self.config.resultDir, xlabel="Epoch")
			
		write_scores_comb(allLosses=allLosses, allScores=allScores, currResultDir=self.config.resultDir, xlabel="Epoch")
		
	# Calculates (totalWithinLoss,totalAcrossLoss) for given canopies and return a tuple of these losses after converting them to scalars
	def calcTotalLoss(self, canopies, useAvgWeights):

		totalWithinLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		totalWithinLoss = Variable(totalWithinLoss, requires_grad=True)

		totalAcrossLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		totalAcrossLoss = Variable(totalAcrossLoss, requires_grad=True)

		for canopyId in sorted(canopies):

			######################### Loss for within cluster Edges #####################################
			if self.config.trainObj.startswith("linkage"):
				try:
					linkAlpha = float(self.config.trainObj.split("_")[-1])
				except ValueError:
					linkAlpha = self.config.trainObj.split("_")[-1]
				
				canopy = canopies[canopyId]
				withinLoss, acrossLoss = self.calcOverallHACLoss(canopy=canopy, linkAlpha=linkAlpha)
				totalWithinLoss = totalWithinLoss + withinLoss
				totalAcrossLoss = totalAcrossLoss + acrossLoss
			elif self.config.trainObj.startswith("triplet"):
				canopy = canopies[canopyId]
				withinLoss, acrossLoss = self.calcTripletLoss(canopy=canopy)
				totalWithinLoss = totalWithinLoss + withinLoss
				totalAcrossLoss = totalAcrossLoss + acrossLoss
				
			else:
				if self.config.trainObj.startswith("bestWithin"):
					withinLoss, _ = self.calcWithinEdgeLoss(canopy=canopies[canopyId], onlyBestWithinEdges=True, useAvgWeights=useAvgWeights)
				elif self.config.trainObj.startswith("allWithin"):
					withinLoss, _ = self.calcWithinEdgeLoss(canopy=canopies[canopyId], onlyBestWithinEdges=False, useAvgWeights=useAvgWeights)
				elif self.config.trainObj.startswith("mstWithin"):
					withinLoss, _ = self.calcMSTLoss(canopy=canopies[canopyId])
				else:
					self.logger.info("Invalid training objective:{}".format(self.config.trainObj))
					raise Exception("Invalid training objective:{}".format(self.config.trainObj))
	
				######################### Loss for Across Cluster Edges #####################################
				if self.config.trainObj.endswith("bestAcross"):
					acrossLoss = self.calcAcrossEdgeLoss(canopy=canopies[canopyId], onlyBestAcrossEdges=True, useAvgWeights=useAvgWeights)
				elif self.config.trainObj.endswith("allAcross"):
					acrossLoss = self.calcAcrossEdgeLoss(canopy=canopies[canopyId], onlyBestAcrossEdges=False, useAvgWeights=useAvgWeights)
				else:
					self.logger.info("Invalid training objective:{}".format(self.config.trainObj))
					raise Exception("Invalid training objective:{}".format(self.config.trainObj))
	
				totalWithinLoss = totalWithinLoss + withinLoss
				totalAcrossLoss = totalAcrossLoss + acrossLoss
				

		# Convert loss to scalars from pytorch tensors
		totalWithinLoss = totalWithinLoss.data.cpu().numpy()[0]
		totalAcrossLoss = totalAcrossLoss.data.cpu().numpy()[0]

		totalWithinLoss = totalWithinLoss/len(canopies) if len(canopies) > 0. else 0.
		totalAcrossLoss = totalAcrossLoss/len(canopies) if len(canopies) > 0. else 0.
		
		return totalWithinLoss, totalAcrossLoss

	# Calculates loss for across cluster edges either using best edges or all edges
	# Best means min if model outputs distance, and best means max if model outputs similarity
	def calcAcrossEdgeLoss(self, canopy, onlyBestAcrossEdges, useAvgWeights):  # TODO Can I avoid iterating over each point? Can bestDist for all points be computed at once?
		
		tempTensor = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		totalBestDistAcross = Variable(tempTensor, requires_grad=True)
		
		numEdgesInLoss = 0
		pairFeatures = canopy["pairFeatures"]
		for cid in canopy["clusterToPids"]:
			# Iterate over cluster other than current cluster and accumulate ids of points in those clusters
			otherPoints = []
			for otherCid in canopy["clusterToPids"]:
				if otherCid == cid: continue  # Need to iterate over clusters other than cid
				otherPoints += canopy["clusterToPids"][otherCid]
			
			for pid in canopy["clusterToPids"][cid]:
				acrossClusterFeatures = []  # Accumulate all across edge features in this list
				for acrossPid in otherPoints:
					p1, p2 = min(pid, acrossPid), max(pid, acrossPid)
					acrossClusterFeatures.append(pairFeatures[(p1, p2)])
				
				if len(acrossClusterFeatures) == 0:
					continue
				
				if useAvgWeights and isinstance(self.model, AvgLinearClassifier):
					adjVector = self.model.pairAvgBatchForward(acrossClusterFeatures)
				else:
					adjVector = self.model.pairBatchForward(acrossClusterFeatures)
				
				if self.config.outDisSim: # Output of model denotes distance or dissimilarity
					if onlyBestAcrossEdges:  # For each point, use the best across cluster edge to compute loss
						bestDistAcross = torch.min(adjVector)  # Best edge is the smallest edge as edgeWeight here denote distance
						if self.config.threshold is None:
							totalBestDistAcross = totalBestDistAcross - bestDistAcross ** 2
							numEdgesInLoss += 1
						elif (bestDistAcross < self.config.threshold + self.config.margin):  # Add -bestDistAcross**2 to loss to make it larger
							totalBestDistAcross = totalBestDistAcross - bestDistAcross
							numEdgesInLoss += 1
					else:  # For each point, take all across cluster edge for computing loss
						if self.config.threshold is None:  # Make across edges larger
							totalBestDistAcross = totalBestDistAcross - torch.sum(adjVector ** 2)
							numEdgesInLoss += adjVector.shape[0]
						else:
							applyThreshold = torch.nn.Threshold(-1 * (self.config.threshold + self.config.margin), 0)
							adjVector = -1 * applyThreshold(-1 * adjVector)  # Remove edges greater than threshold + margin from adjVector
							
							totalBestDistAcross = totalBestDistAcross - torch.sum(adjVector)
							numEdgesInLoss += adjVector.nonzero().shape[0]
				else: # Output of model denotes similarity
					if onlyBestAcrossEdges:  # For each point, find the best across cluster edge
						bestDistAcross = torch.max(adjVector)  # Best edge is the largest edge
						if self.config.threshold is None:  # Regress across edges to zero
							totalBestDistAcross = totalBestDistAcross + bestDistAcross ** 2
							numEdgesInLoss += 1
						elif (bestDistAcross > self.config.threshold - self.config.margin):
							totalBestDistAcross = totalBestDistAcross + bestDistAcross
							numEdgesInLoss += 1
					else:  # For each point, take all across cluster edge for computing loss
						
						if self.config.threshold is None:  # Regress across edges to zero
							totalBestDistAcross = totalBestDistAcross + torch.sum(adjVector ** 2)
							numEdgesInLoss += adjVector.shape[0]
						else:
							applyThreshold = torch.nn.Threshold(self.config.threshold - self.config.margin, 0)
							adjVector = applyThreshold(adjVector)  # Remove edges less than threshold - margin
							
							totalBestDistAcross = totalBestDistAcross + torch.sum(adjVector)
							numEdgesInLoss += adjVector.nonzero().shape[0]
				

		
		if onlyBestAcrossEdges:
			if self.config.normalizeLoss and numEdgesInLoss > 0:
				totalBestDistAcross = totalBestDistAcross / numEdgesInLoss
				
			return totalBestDistAcross  # Every edges need not be counted twice
			
		else:
			if self.config.normalizeLoss and numEdgesInLoss > 0:
				return totalBestDistAcross/numEdgesInLoss
			else:
				return totalBestDistAcross / 2  # This is because every edge gets counted twice
	
	# Calculates loss for within cluster edges either using best edges or all edges
	# Best means min if model outputs distance, and best means max if model outputs similarity
	def calcWithinEdgeLoss(self, canopy, onlyBestWithinEdges, useAvgWeights):  # TODO: Does batching all GPU Computation help? Currently, it is batched for at granularity of each point
		
		totalWithinLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		totalWithinLoss = Variable(totalWithinLoss, requires_grad=True)
		
		totalNumEdges = 0
		for clusterId in canopy["clusterToPids"]:
			numEdges = 0
			pidList = canopy["clusterToPids"][clusterId]
			pairFeatures = canopy["pairFeatures"]
			if len(pidList) <= 1:  # compute loss only when there is more than 1 point in cluster
				continue
			
			for pid1 in pidList:
				relevantPairFeatures = []
				for pid2 in pidList:
					if pid1 == pid2: continue
					minP, maxP = min(pid1, pid2), max(pid1, pid2)
					relevantPairFeatures.append(pairFeatures[(minP, maxP)])
				
				if useAvgWeights and isinstance(self.model, AvgLinearClassifier):
					adjVector = self.model.pairAvgBatchForward(relevantPairFeatures)
				else:
					adjVector = self.model.pairBatchForward(relevantPairFeatures)
					
				
				if self.config.outDisSim: # Model outputs dissimilarity or distance
					if onlyBestWithinEdges:  # Use only bestWithin edge to compute loss
						bestEdge = torch.min(adjVector)
						if self.config.threshold is None or bestEdge > self.config.threshold - self.config.margin:  # Make bestEdge smaller(as edgeWeights denote distance)
							totalWithinLoss = totalWithinLoss + bestEdge
							numEdges += 1
					else:  # Use all within edges to compute loss
						if self.config.threshold is None:
							totalWithinLoss = totalWithinLoss + torch.sum(adjVector)
							numEdges += adjVector.shape[0]
						else:
							applyThreshold = torch.nn.Threshold(self.config.threshold - self.config.margin, 0)
							adjVector = applyThreshold(adjVector)  # Retain edges larger than thresh-margin to compute loss
							totalWithinLoss = totalWithinLoss + torch.sum(adjVector)
							numEdges += adjVector.nonzero().shape[0]
				else: # Model outputs similarity
					if onlyBestWithinEdges:  # Use only bestWithin edge to compute loss
						bestEdge = torch.max(adjVector)
						if self.config.threshold is None:  # Regress bestEdge to 1
							totalWithinLoss = totalWithinLoss  - bestEdge
							numEdges += 1
						elif bestEdge < self.config.threshold + self.config.margin:  # Make bestEdge larger if it is less than threshold + margin
							totalWithinLoss = totalWithinLoss - bestEdge
							numEdges += 1
					else:  # Use all within edges to compute loss
						if self.config.threshold is None:  # Regress all edges to 1
							totalWithinLoss = totalWithinLoss - torch.sum(adjVector)
							numEdges += adjVector.shape[0]
						else:
							applyThreshold = torch.nn.Threshold(-1 * (self.config.threshold + self.config.margin), 0)
							adjVector = -1 * applyThreshold(-1 * adjVector)  # Retain edges smaller than thesh+margin to compute loss
							
							totalWithinLoss = totalWithinLoss - torch.sum(adjVector)
							numEdges += adjVector.nonzero().shape[0]

			
			totalNumEdges += numEdges
		# print("cid:{},numEdges:{}".format(clusterId, numEdges))
		if onlyBestWithinEdges:
			if self.config.normalizeLoss and totalNumEdges > 0:
				totalWithinLoss = totalWithinLoss / totalNumEdges
			return totalWithinLoss, totalNumEdges
		else:
			if self.config.normalizeLoss and totalNumEdges > 0:
				totalWithinLoss = totalWithinLoss / totalNumEdges
				return totalWithinLoss, totalNumEdges  # This is because every edge gets added twice
			else:
				return totalWithinLoss / 2, totalNumEdges  # This is because every edge gets added twice
	
	def calcMSTLoss(self, canopy):  # TODO: Speedup by using batchComputation using GPUs
		''' Creates an MST over each gt cluster in canopy and returns loss for mst edges which
		 are worse than a threshold or sum of all mst edges if not using a threshold
		 (worse could mean  lower or greater than the threshold depending on whether the model is
		 outputting a similarity score or distance score between two points)
		 ABUSE of Notation: When model outputs similarity, MST does not contain the smallest edges but largest possible edges(because higher the
		similarity, the better it is.'''
		
		
		totalMSTLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		totalMSTLoss = Variable(totalMSTLoss, requires_grad=True)
		totalnumMSTEdges = 0
		for clusterId in canopy["clusterToPids"]:
			pidList = canopy["clusterToPids"][clusterId]
			pairFeatures = canopy["pairFeatures"]
			if len(pidList) <= 1:  # compute loss only when there is more than 1 point in cluster
				continue
			pidToIdx = {pid: idx for idx, pid in enumerate(pidList)}
			idxToPid = {idx: pid for idx, pid in enumerate(pidList)}
			numPoints = len(pidList)
			torchAdjMatrix = {}  # To store edges in form of pytorch variables
			data = []
			rows = []
			cols = []
			
			pairList = [pair for pair in itertools.combinations(sorted(pidList), 2)]
			pairToIdx = {pair: idx for idx, pair in enumerate(pairList)}
			pairFeaturesList = [pairFeatures[pair] for pair in pairList]
			edgeWeights = self.model.pairBatchForward(pairFeaturesList)
			edgeWeightsNP = edgeWeights.cpu().data.numpy()
			
			for pid1, pid2 in pairToIdx:
				pairIdx = pairToIdx[(pid1, pid2)]
				torchAdjMatrix[(pid1, pid2)] = edgeWeights[pairIdx]
				torchAdjMatrix[(pid2, pid1)] = edgeWeights[pairIdx]
				edgeWeight = edgeWeightsNP[pairIdx][0]
				rows += [pidToIdx[pid1]]
				cols += [pidToIdx[pid2]]
				
				if self.config.outDisSim:
					data += [edgeWeight]
				else:
					data += [-1 * edgeWeight]  # Taking -ve here as model outputs similarity(larger the better)& MST is going to pick smallest edges
					
			sparseMatrix = csr_matrix((data, (rows, cols)), shape=(numPoints, numPoints))
			mst = minimum_spanning_tree(sparseMatrix)
			mst = mst.todok()  # Get edges in minimum spanning tree
			
			currMSTLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
			currMSTLoss = Variable(currMSTLoss, requires_grad=True)
			
			numMSTEdges = 0
			for (idx1, idx2) in mst.keys():  # Iterate over edges in MST and add loss for edges that were not predicted
				pid1, pid2 = idxToPid[idx1], idxToPid[idx2]
				# assert (torchAdjMatrix[(pid1, pid2)].data == -1*mst[(idx1, idx2)])
				
				if self.config.outDisSim: # Model outputs distance/dissimilarity
					if self.config.threshold is None or torchAdjMatrix[(pid1, pid2)] > self.config.threshold - self.config.margin:  # Make MST Edges smaller
						currMSTLoss = currMSTLoss + torchAdjMatrix[(pid1, pid2)]
						numMSTEdges += 1
				else: # Model outputs similarity
					if self.config.threshold is None:  # Make MST Edges larger
						currMSTLoss = currMSTLoss - torchAdjMatrix[(pid1, pid2)]
						numMSTEdges += 1
					elif torchAdjMatrix[(pid1, pid2)] < self.config.threshold + self.config.margin:  # Make MST edges larger as edges represent similarity
						currMSTLoss = currMSTLoss - torchAdjMatrix[(pid1, pid2)]
						numMSTEdges += 1
					
			
			totalMSTLoss = totalMSTLoss + currMSTLoss
			totalnumMSTEdges += numMSTEdges
		
		if self.config.normalizeLoss and totalnumMSTEdges > 0:
			totalMSTLoss = totalMSTLoss / totalnumMSTEdges
		
		return totalMSTLoss, totalnumMSTEdges
	
	def calcTripletLoss(self, canopy):
		
		totalWithinLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		totalWithinLoss = Variable(totalWithinLoss, requires_grad=True)
		
		totalAcrossLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		totalAcrossLoss = Variable(totalAcrossLoss, requires_grad=True)
		
		# If there is just 1 ground-truth cluster then we can not compute triplet loss
		# or if every point is a singleton cluster then return
		if len(canopy["clusterToPids"]) <= 1 or len(canopy["clusterToPids"]) == len(canopy["pidToCluster"]):
			return totalWithinLoss, totalAcrossLoss
		
		pidList = list(canopy["pidToCluster"].keys())
		negExamples = []
		posExamples = []
		numTriplets = self.config.numErrorTriplet*len(pidList)
		
		for i in range(numTriplets):
			pid1  = random.choice(pidList)
			
			# Get cluster for this point
			cid_pos 	= canopy["pidToCluster"][pid1]
			# Keep picking pid1 until you find a pid that does not belong to singleton cluster
			while len(canopy["clusterToPids"][cid_pos]) == 1:
				pid1  = random.choice(pidList)
				# Get cluster for this point
				cid_pos 	= canopy["pidToCluster"][pid1]
				
			# Pick a point from same cluster as pid1
			pid_pos 	= random.choice(canopy["clusterToPids"][cid_pos])
			while pid1 == pid_pos:  # pid1 and pid_pos should be different points
				pid_pos 	= random.choice(canopy["clusterToPids"][cid_pos])
			
			# Sample point from different cluster
			pid_neg  	= random.choice(pidList)
			cid_neg  	= canopy["pidToCluster"][pid_neg]
			while cid_neg == cid_pos:
				pid_neg  	= random.choice(pidList)
				cid_neg  	= canopy["pidToCluster"][pid_neg]
				
			
			# Make sure first pid is smaller than second one, if not then swap them
			p1,p2 = min(pid1,pid_pos), max(pid1, pid_pos)
			posExamples += [(p1,p2)]
			
			p1,p2 = min(pid1,pid_neg), max(pid1, pid_neg)
			negExamples += [(p1,p2)]
			
		pairFeatures 	= canopy["pairFeatures"]
		posPairFeatList = [pairFeatures[pair] for pair in posExamples]
		negPairFeatList = [pairFeatures[pair] for pair in negExamples]
		
		posPairWeights = self.model.pairBatchForward(posPairFeatList)
		negPairWeights = self.model.pairBatchForward(negPairFeatList)
		
		
		if not self.config.outDisSim:
			raise NotImplementedError
		
		# Treating weights as dissimilarities
		applyThreshold 	= torch.nn.Threshold(self.config.threshold - self.config.margin, 0)
		filteredEdges 	= applyThreshold(posPairWeights)  # Retain edges larger than thresh-margin to compute loss
		totalWithinLoss = totalWithinLoss + torch.sum(filteredEdges)
		
		
		applyThreshold 	= torch.nn.Threshold(-1*(self.config.threshold + self.config.margin), 0)
		filteredEdges 	= -1*applyThreshold(-1*negPairWeights)  # Retain edges smaller than thresh+margin to compute loss
		totalAcrossLoss = totalAcrossLoss - torch.sum(filteredEdges)  # Add -sum() because we want to make these edges larger
		
		if self.config.normalizeLoss:
			totalWithinLoss = totalWithinLoss/ numTriplets
			totalAcrossLoss = totalAcrossLoss/ numTriplets
		
		return totalWithinLoss, totalAcrossLoss
	
	def calcOverallHACLoss(self, canopy, linkAlpha):
		
		start = time.time()
		t1 = time.time()

		numPoints = len(canopy["pidToCluster"])
		self.logger.info("----------------------------------------------------------------")
		self.logger.info("\t\tNumPoints={}".format(numPoints))
		distMat_torch	= get_affinity_mat(model=self.model, pairFeatures=canopy["pairFeatures"], numPoints=numPoints, dType='dist', getNPMat=False)
		distMat_NP 		= distMat_torch.cpu().data.numpy()

		t2 = time.time()
		self.logger.info("\t\tTime taken to compute distance matrix:{:.4f} for numPoints={}".format(t2 -t1, numPoints))
		if linkAlpha == "auto":
			linkAlphaTorch = self.model.linkAlpha
			linkAlpha = self.model.linkAlpha.cpu().data.numpy()[0]
			linkAlpha = float(linkAlpha)
		else:
			linkAlphaTorch = linkAlpha

		t1 = time.time()
		posLinkages, negLinkages = runHAC_torch_allEdges(origDistMat=distMat_NP, origTorchDistMat=distMat_torch,
														 numPoints=numPoints, linkAlpha=linkAlpha,
														 linkAlphaTorch=linkAlphaTorch, pidToGtCluster=canopy["pidToCluster"],
														 scaleDist=self.config.scaleDist, getBestImpure=False)
		t2 = time.time()
		
		self.logger.info("\t\t\tTime taken to run HAC with softMax with vectors for posNeg links:{:.4f}".format(t2 - t1))
		posLinkLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		posLinkLoss = Variable(posLinkLoss, requires_grad=True)
		
		negLinkLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		negLinkLoss = Variable(negLinkLoss, requires_grad=True)
		
		t1 = time.time()
		posCtr, negCtr = 0, 0
		if self.config.threshold is None:
			posLinkLoss = posLinkLoss + torch.sum( posLinkages )
			posCtr += posLinkages.shape[0]
		else:
			applyThreshold = torch.nn.Threshold(self.config.threshold - self.config.margin, 0)
			filteredLinks = applyThreshold(posLinkages)
			posLinkLoss =  posLinkLoss + torch.sum( filteredLinks )
			posCtr += filteredLinks.nonzero().shape[0]
		
		if self.config.threshold is None:
			negLinkLoss = negLinkLoss + torch.sum(negLinkages)
			negCtr += negLinkages.shape[0]
		else:
			applyThreshold = torch.nn.Threshold( -1*(self.config.threshold + self.config.margin), 0)
			filteredLinks  = -1*applyThreshold(-1*negLinkages)
			negLinkLoss = negLinkLoss - torch.sum(filteredLinks)
			negCtr += filteredLinks.nonzero().shape[0]
			
		t2 = time.time()
		self.logger.info("\t\t\tTime taken to accumulate loss with softMax with vectors for posNeg links :{:.4f}".format(t2 - t1))
		end = time.time()
		if self.config.normExpLinkLoss:
			posLinkLoss = posLinkLoss / posCtr if posCtr != 0 else posLinkLoss
			negLinkLoss = negLinkLoss / negCtr if negCtr != 0 else negLinkLoss
		
		
		self.logger.info("\t\tPosLinkLoss = {} {}".format(posLinkLoss.cpu().data.numpy()[0], posCtr))
		self.logger.info("\t\tNegLinkLoss = {} {}".format(negLinkLoss.cpu().data.numpy()[0], negCtr))

		self.logger.info("----------------------------Time = {:.4f}------------------------------------".format(end-start))
		return posLinkLoss, negLinkLoss
	
	