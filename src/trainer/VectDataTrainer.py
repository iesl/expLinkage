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
import math
import random
import pprint
import numpy as np
import torch
from torch.autograd import Variable

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


from eval.evalVectData import eval_model_vect
from eval.finalEval import eval_all_data
from utils.basic_utils import  create_logger, load_vec_data_splits, calc_batch_size
from utils.plotting import plot_scores, write_scores_comb
from eval.threshold import choose_threshold

from models import create_new_model
from hier_clust.expLink import runHAC_torch_allEdges, runHAC_torch_allEdges_faces
from trainer.BaseTrainer import BaseTrainer

class VectDataTrainer(BaseTrainer):
	"""docstring for VectDataTrainer"""
	
	def __init__(self,config):
		super(VectDataTrainer, self).__init__(config)
		
		self.config = config
		self.logger = create_logger(config=config, logFile=config.logFile, currLogger=None)
		self.model 	= create_new_model(config)
		self.evalFunc = eval_model_vect
		
		self.trainCanopies, self.testCanopies, self.devCanopies = load_vec_data_splits(self.config)

		if self.config.useGPU:
			self.logger.info("Shifting model to cuda because GPUs are available!")
			self.model = self.model.cuda()
		
		self.optimizer = None
		self.resetOptimizer()
		
		self.logger.info("Successfully initialized model trainer...")
		self.logger.info(str(self))
	
	def __str__(self):
		printDict = {}
		printStr = ""
		printStr += "--" * 15 + "Model trainer parameters" + "--" * 15 + "\n"
		printStr += "resultDir::\t{}\n".format(self.config.resultDir)
		printStr += "inputDim::\t{}\n".format(self.config.inputDim)
		printStr += "trainFrac::\t{}\n".format(self.config.trainFrac)
		printStr += "numTrainPoints::\t{}\n".format([len(canopy["points"]) for canopy in self.trainCanopies])
		printStr += "numTrainCluster::\t{}\n".format([len(canopy["clusters"]) for canopy in self.trainCanopies])
		printStr += "testFrac::\t{}\n".format(self.config.testFrac)
		printStr += "numTestPoints::\t{}\n".format([len(canopy["points"]) for canopy in self.testCanopies])
		printStr += "numTestCluster::\t{}\n".format([len(canopy["clusters"]) for canopy in self.testCanopies])
		printStr += "devFrac::\t{}\n".format(self.config.devFrac)
		printStr += "numDevPoints::\t{}\n".format([len(canopy["points"]) for canopy in self.devCanopies])
		printStr += "numDevCluster::\t{}\n".format([len(canopy["clusters"]) for canopy in self.devCanopies])
		
		printDict["Model Params"]  = str(self.model)
		printDict["Optimizer"]	   = str(self.optimizer) + "\n" + str(self.optimizer.param_groups) + "\n"
		
		printDict["Config"] = pprint.pformat(self.config.to_json())
		
		return pprint.pformat(printDict)
	
	def printModelWeights(self):
		weightStr = self.model.getWeightStr()
		self.logger.info(weightStr)
		
	def train(self):
		
		if self.config.trainObj.startswith("triplet"):
			self.train_triplet()
		elif "Within" in self.config.trainObj and "Across" in self.config.trainObj:
			if self.config.threshold is None:
				if self.config.margin is not None:
					self.logger.info("Training in ranking style using margin={}".format(self.config.margin))
					self.train_rankingStyle()
				else:
					self.logger.info(
						"Training in using all edges as specified by trainObj (without any comparision to threshold)"
						"or any margin")
					self.train_general()
			else:
				assert self.config.margin is not None  # If threshold is not None then margin should not be None, it can be zero at least
				self.logger.info(
					"Training in using all edges as specified by trainObj by comparing them with threshold+-margin={}+-{}".format(
						self.config.threshold, self.config.margin))
				self.train_general()
		
		elif self.config.trainObj.startswith("linkage"):
			self.train_rankingStyle()
		else:
			self.logger.info("Invalid training objective::{}".format(self.config.trainObj))
			raise Exception("Invalid training objective::{}".format(self.config.trainObj))
	
	# This function implements different ways of calculating loss for within cluster and across cluster edges
	def train_general(self):
		
		allLosses = {"train": {}, "test": {}, "dev": {}}
		allScores = {"train": {}, "test": {}, "dev": {}}
		
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
			
			for canopy in self.trainCanopies:
				for cid in canopy["clusters"]:  # Iterate over each cluster and find loss for its within cluster and across cluster edges
					withinClusterLoss, acrossClusterLoss = self.calcLoss_general(canopy, cid)
					
					totalLoss = withinClusterLoss + acrossClusterLoss
					self.optimizer.zero_grad()
					totalLoss.backward()
					self.optimizer.step()
					
					totalWithinLoss = totalWithinLoss + withinClusterLoss
					totalAcrossLoss = totalAcrossLoss + acrossClusterLoss
			
			# self.logger.info("Processed cluster {} with {} points".format(cid, len(canopy["clusters"][cid])))
			# self.logger.info("Epoch:{}\tcid::{}\tminWithin\t\t{:.3f}".format(epoch, cid, withinClusterLoss.data[0]))
			# self.logger.info("\tcid::{}\tminDistAcross\t{:.3f}".format(cid, acrossClusterLoss.data[0]))
			
			totalWithinLoss = totalWithinLoss.cpu().data.numpy()[0]
			totalAcrossLoss = totalAcrossLoss.cpu().data.numpy()[0]
			
			allLosses["train"][epoch] = (totalWithinLoss, totalAcrossLoss)
			allLosses["dev"][epoch] = self.calcTotalLoss_general(self.devCanopies)
			if hasattr(self.model, "linkAlpha"):
				linkAlpha = self.model.linkAlpha
				allLosses["test"][epoch] = linkAlpha.cpu().data.numpy()[0]
			else:
				allLosses["test"][epoch] = 0.
			
			end = time.time()
			self.logger.info("\nEpoch:{}\tTime taken::\t{:.3f}".format(epoch, end - start))
			self.logger.info("\ttotalWithinLoss\t\t{:.3f}".format(totalWithinLoss))
			self.logger.info("\ttotalAcrossLoss\t{:.3f}".format(totalAcrossLoss))
			
			if (epoch + 1) % self.config.epochToEval == 0:
				threshDict = {}
				for method in self.config.inferenceMethods:
					if method == "connComp" or method.endswith("@t"):
						threshDict[method] = choose_threshold(trainer=self,infMethod=method, epoch=epoch)
					
				allScores["train"][epoch], allScores["test"][epoch], allScores["dev"][epoch] = eval_all_data(trainer=self,threshDict=threshDict)
				self.plotAndWriteData(allLosses=allLosses, allScores=allScores)
			
			if self.stopTraining(epoch=epoch, allLosses=allLosses):
				break
		
		self.plotAndWriteData(allLosses=allLosses, allScores=allScores)
	
	def train_triplet(self):
		
		allLosses = {"train": {}, "dev": {}, "test": {}}
		allScores = {"train": {}, "dev": {}, "test": {}}
		
		if self.config.evalBeforeTrain:
			threshDict = {}
			for method in self.config.inferenceMethods:
				if method == "connComp" or method.endswith("@t"):
					threshDict[method] = choose_threshold(trainer=self,infMethod=method, epoch=-1)

			allScores["train"][-1], allScores["test"][-1], allScores["dev"][-1] = eval_all_data(trainer=self,threshDict=threshDict)
		
		for epoch in range(self.config.numEpoch):
			start = time.time()
			
			totalNegSampleLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
			totalNegSampleLoss = Variable(totalNegSampleLoss, requires_grad=True)
			totalPosSampleLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
			totalPosSampleLoss = Variable(totalPosSampleLoss, requires_grad=True)
			overAllLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
			overAllLoss = Variable(overAllLoss, requires_grad=True)
			
			for canopy in self.trainCanopies:
				for cid in canopy["clusters"]:
					totalLoss, posSampleLoss, negSampleLoss = self.calcLoss_triplet(canopy, cid)
					if self.config.threshold is not None:  # Separate losses are calculated only when we have a threshold
						totalPosSampleLoss = totalPosSampleLoss + posSampleLoss
						totalNegSampleLoss = totalNegSampleLoss + negSampleLoss
					
					overAllLoss = overAllLoss + totalLoss
					overAllLoss = overAllLoss + totalLoss
					
					self.optimizer.zero_grad()
					totalLoss.backward()
					self.optimizer.step()
			
			# self.logger.info("Epoch:{}\tcid::{:<6}\ttotalLoss\t{:.3f}".format(epoch, cid, totalLoss.item()))
			
			totalPosSampleLoss = totalPosSampleLoss.data.cpu().numpy()[0]
			totalNegSampleLoss = totalNegSampleLoss.data.cpu().numpy()[0]
			overAllLoss 		= overAllLoss.data.cpu().numpy()[0]
			allDevLoss, allDevWithin, allDevAcross = self.calcTotalLoss_triplet(self.devCanopies)
			if self.config.threshold is None:
				allLosses["train"][epoch] = overAllLoss
				allLosses["dev"][epoch] = allDevLoss
			else:
				allLosses["train"][epoch] = (totalPosSampleLoss, totalNegSampleLoss)
				allLosses["dev"][epoch] = (allDevWithin, allDevAcross)

			
			end = time.time()
			self.logger.info("\nEpoch:{}\tTime taken::\t{:.3f}".format(epoch, end - start))
			self.logger.info("\tTotalNegSampleLoss\t{:.3f}".format(totalNegSampleLoss))
			self.logger.info("\tTotalPosSampleLoss\t{:.3f}".format(totalPosSampleLoss))
			self.logger.info("\toverAllLoss\t{:.3f}".format(overAllLoss))
			
			if (epoch + 1) % self.config.epochToEval == 0:
				threshDict = {}
				for method in self.config.inferenceMethods:
					if method == "connComp" or method.endswith("@t"):
						threshDict[method] = choose_threshold(trainer=self,infMethod=method, epoch=epoch)
					
				allScores["train"][epoch], allScores["test"][epoch], allScores["dev"][epoch] = eval_all_data(trainer=self,threshDict=threshDict)
				self.plotAndWriteData(allLosses=allLosses, allScores=allScores)
			
			
			if self.stopTraining(epoch=epoch, allLosses=allLosses):
				break
		
		self.plotAndWriteData(allLosses=allLosses, allScores=allScores)
	
	# This function implements different ways of calculating loss for within cluster and across cluster edges
	# (where it tries to rank within edges comparatively better than across edges)
	def train_rankingStyle(self):
		assert self.config.trainObj != "mstWithin_allAcross"  # These are not supported currently
		assert self.config.trainObj != "allWithin_allAcross"
		assert self.config.threshold is None or self.config.trainObj.startswith("linkage")  # This function should only be used when no threshold is given for training
		
		allLosses = {"train": {}, "dev": {}, "test": {}}
		allScores = {"train": {}, "dev": {}, "test": {}}
		
		if self.config.evalBeforeTrain:
			threshDict = {}
			for method in self.config.inferenceMethods:
				if method == "connComp" or method.endswith("@t"):
					threshDict[method] = choose_threshold(trainer=self,infMethod=method, epoch=-1)
	
			allScores["train"][-1], allScores["test"][-1], allScores["dev"][-1] = eval_all_data(trainer=self,threshDict=threshDict)
		
		for epoch in range(self.config.numEpoch):
			
			start = time.time()
			totalLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
			totalLoss = Variable(totalLoss, requires_grad=True)
			
			for canopy in self.trainCanopies:
				for currCid in canopy["clusters"]:  # Iterate over each cluster and find loss for it
					
				
					loss = self.calcLoss_rankingStyle(canopy, currCid)
					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()
					
					totalLoss = totalLoss + loss
					# self.logger.info("Epoch:{}\tcid::{:<6}\ttotalLoss\t{:.3f}".format(epoch, currCid, loss.data[0]))
					# self.logger.info("Epoch:{}\tcid::{:<6}\ttotalLoss\t{:.3f}".format(epoch, currCid, loss.item()))
					
					if self.config.trainObj.startswith("linkage"):  # Breaking because for linakge_{} objective, we do not compute loss for each cluster separately, rather we build tree over all points at once
						break
			
			allLosses["train"][epoch] = totalLoss
			allLosses["train"][epoch] = allLosses["train"][epoch].data.cpu().numpy()[0]
			self.logger.info("Calculating loss on dev set")
			allLosses["dev"][epoch] = self.calcTotalLoss_rankingStyle(self.devCanopies)
			if hasattr(self.model, "linkAlpha"):
				linkAlpha = self.model.linkAlpha
				allLosses["test"][epoch] = float(linkAlpha.data.cpu().numpy()[0])
			else:
				allLosses["test"][epoch] = 0.
			
			end = time.time()
			self.logger.info("\nEpoch:{}\tTime taken::\t{:.3f}".format(epoch, end - start))
			self.logger.info("\ttotalLoss\t\t{:.3f}".format(totalLoss.data[0]))
			self.logger.info("\tAlpha:{:.4f}".format(allLosses["test"][epoch]))
			# self.logger.info("\tAlpha:{}\t{}\t{}".format(allLosses["test"][epoch], allLosses["test"][epoch] == float(np.nan), type(allLosses["test"][epoch])))
			if "{}".format(allLosses["test"][epoch]) == "nan":
				self.logger.info("This run is wasted as linkAlpha has value nan")
				exit(0)
			
			if (epoch + 1) % self.config.epochToEval == 0:
				threshDict = {}
				for method in self.config.inferenceMethods:
					if method == "connComp" or method.endswith("@t"):
						threshDict[method] = choose_threshold(trainer=self,infMethod=method, epoch=epoch)
					
				allScores["train"][epoch], allScores["test"][epoch], allScores["dev"][epoch] = eval_all_data(trainer=self,threshDict=threshDict)
				self.plotAndWriteData(allLosses=allLosses, allScores=allScores)
			
			if self.stopTraining(epoch=epoch, allLosses=allLosses): break
		
		self.plotAndWriteData(allLosses=allLosses, allScores=allScores)
	
	def stopTraining(self, epoch, allLosses):
		
		numEpToAvg = self.config.numEpToAvg
		
		if epoch <= numEpToAvg:
			return False
		
		# Triplet
		dtype = "dev" if len(self.devCanopies) > 0 else "train"  # Which canopies to determine convergence...
		
		if self.config.threshold is None or self.config.trainObj.startswith("linkage"):  # Used when using ranking style training, or when trainObj is linkage_{}
			avgLoss = np.mean([allLosses[dtype][e] for e in range(epoch - 1)][-1 * numEpToAvg:])  # Average loss over last numEpToAvg epoch...
			
			if avgLoss == 0 or abs(allLosses[dtype][epoch] - avgLoss) / abs(avgLoss) < self.config.epsilon:
				self.logger.info("{} loss has saturated so stopping training...{}\t"
								 "{}".format(dtype, abs(allLosses[dtype][epoch] - avgLoss), abs(avgLoss)))
				return True
		
		else:  # Used when train_general function where within and across cluster losses are calculated separately
			avgLossWithin = np.mean([allLosses[dtype][e][0] for e in range(epoch - 1)][-1 * numEpToAvg:])  # Average loss over last numEpToAvg epoch...
			avgLossAcross = np.mean([allLosses[dtype][e][1] for e in range(epoch - 1)][-1 * numEpToAvg:])  # Average loss over last numEpToAvg epoch...
			
			if (avgLossWithin == 0 or abs(allLosses[dtype][epoch][0] - avgLossWithin) / abs(
					avgLossWithin) < self.config.epsilon) and \
					(avgLossAcross == 0 or abs(allLosses[dtype][epoch][1] - avgLossAcross) / abs(
						avgLossAcross) < self.config.epsilon):
				self.logger.info("{} within loss has saturated so stopping training...{}\t{}".format(dtype, abs(
					allLosses[dtype][epoch][0] - avgLossWithin), abs(avgLossWithin)))
				self.logger.info("{} across loss has saturated so stopping training...{}\t{}".format(dtype, abs(
					allLosses[dtype][epoch][1] - avgLossAcross), abs(avgLossAcross)))
				return True
		
		return False
	
	def plotAndWriteData(self, allLosses, allScores):
		
		if self.config.makeScorePlots:
			plot_scores(allLosses=allLosses, allScores=allScores, currResultDir=self.config.resultDir, xlabel="Epoch")
		else:
			# Avoid plotting scores, just plot losses
			plot_scores(allLosses=allLosses, allScores={"train":{}, "test":{}, "dev":{}}, currResultDir=self.config.resultDir, xlabel="Epoch")
			
		write_scores_comb(allLosses=allLosses, allScores=allScores, currResultDir=self.config.resultDir, xlabel="Epoch")
		
	# 	Returns tuple (withinLoss, acrossLoss) for loss functions other than triplet, and when not training using ranking style
	def calcTotalLoss_general(self, canopies):
		
		totalWithinLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		totalWithinLoss = Variable(totalWithinLoss, requires_grad=True)
		
		totalAcrossLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		totalAcrossLoss = Variable(totalAcrossLoss, requires_grad=True)
		
		for canopy in canopies:
			for cid in canopy[
				"clusters"]:  # Iterate over each cluster and find loss for its within cluster and across cluster edges
				withinClusterLoss, acrossClusterLoss = self.calcLoss_general(canopy, cid)
				totalWithinLoss = totalWithinLoss + withinClusterLoss
				totalAcrossLoss = totalAcrossLoss + acrossClusterLoss
		
		totalWithinLoss = totalWithinLoss.data.cpu().numpy()[0]
		totalAcrossLoss = totalAcrossLoss.data.cpu().numpy()[0]
		
		return (totalWithinLoss, totalAcrossLoss)
	
	# 	Return scalar tuple (overAllLoss, totalWithinLoss, totalAcrossLoss)
	def calcTotalLoss_triplet(self, canopies):
		
		overAllLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		overAllLoss = Variable(overAllLoss, requires_grad=True)
		totalNegSampleLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		totalNegSampleLoss = Variable(totalNegSampleLoss, requires_grad=True)
		totalPosSampleLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		totalPosSampleLoss = Variable(totalPosSampleLoss, requires_grad=True)
		
		for canopy in canopies:
			for cid in canopy["clusters"]:
				totalLoss, posSampleLoss, negSampleLoss = self.calcLoss_triplet(canopy, cid)
				if self.config.threshold is not None:  # Separate losses are calculated only when we have a threshold
					totalPosSampleLoss = totalPosSampleLoss + posSampleLoss
					totalNegSampleLoss = totalNegSampleLoss + negSampleLoss
				
				overAllLoss = overAllLoss + totalLoss
		
		totalPosSampleLoss = totalPosSampleLoss.data.cpu().numpy()[0]
		totalNegSampleLoss = totalNegSampleLoss.data.cpu().numpy()[0]
		overAllLoss = overAllLoss.data.cpu().numpy()[0]
		
		return overAllLoss, totalNegSampleLoss, totalPosSampleLoss
	
	# Returns scalar totalLoss when training in ranking style
	def calcTotalLoss_rankingStyle(self, canopies):
		totalLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		totalLoss = Variable(totalLoss, requires_grad=True)
		
		for canopy in canopies:
			for currCid in canopy["clusters"]:  # Iterate over each cluster and find loss for it
				totalLoss = totalLoss + self.calcLoss_rankingStyle(canopy, currCid)
				if self.config.trainObj.startswith("linkage"):  # Breaking because for linakge_{} objective, we do not compute loss for each cluster separately, rather we build tree over all points at once
					# self.logger.info("\nBreaking out of loop as linkage objective losses are calculated for all clusters at once\n")
					break
		
		totalLoss = totalLoss.data.cpu().numpy()[0]
		
		return totalLoss
	
	# Returns withinClusterLoss, acrossClusterLoss as torch variables for cluster=cid and given canopy
	# for general loss functions (other than triplet) when training with a threshold
	def calcLoss_general(self, canopy, cid):
		
		######################### Loss for within cluster edges  #################
		if self.config.trainObj.startswith("minWithin"):
			withinClusterLoss = self.calcMinWithinLoss(cluster=canopy["clusters"][cid])
		elif self.config.trainObj.startswith("maxWithin"):
			withinClusterLoss = self.calcMaxWithinLoss(cluster=canopy["clusters"][cid])
		elif self.config.trainObj.startswith("mstWithin"):
			withinClusterLoss = self.calcMSTLoss(cluster=canopy["clusters"][cid])
		elif self.config.trainObj.startswith("allWithin"):
			withinClusterLoss = self.calcAllPairWithinLoss(cluster=canopy["clusters"][cid])
		else:
			raise Exception("Invalid loss for within cluster edges:{}".format(self.config.trainObj))
		
		######################### Loss for across cluster edges  #################
		if self.config.trainObj.endswith("minAcross"):
			acrossClusterLoss = self.calcMinAcrossLoss(canopy=canopy, currCid=cid)
		elif self.config.trainObj.endswith("allAcross"):
			acrossClusterLoss = self.calcAllAcrossLoss(canopy=canopy, currCid=cid)
		elif self.config.trainObj.endswith("approxMinAcross"):
			acrossClusterLoss = self.calcApproxMinAcrossLoss(canopy=canopy, currCid=cid)
		else:
			raise Exception("Invalid loss for across cluster edges:{}".format(self.config.trainObj))
		
		return withinClusterLoss, acrossClusterLoss
	
	# Returns totalLoss, posSampleLoss, negSampleLoss as torch variables for cluster=cid and given canopy
	# using triplets to compute loss .
	# If threshold is None, then it returns (totalLoss,0,0) as posSample and negSampleLoss are not calculated separately in that case
	def calcLoss_triplet(self, canopy, cid):
		######################### Loss for Error Triplets #####################################
		numErrorTriplets = self.config.numErrorTriplet * len(canopy["clusters"][cid])
		points = []  # Anchor points
		posPoints = []  # Positive examples corresponding to each anchor point
		negPoints = []  # Negative examples corresponding to each anchor point
		
		for i in range(numErrorTriplets):
			point = random.choice(canopy["clusters"][cid])  # Add a point from this cluster, uniformly at random
			point_pos = random.choice(canopy["clusters"][cid])
			while point == point_pos and len(canopy["clusters"][cid]) > 1:  # point and point_pos have to be different
				point_pos = random.choice(canopy["clusters"][cid])
			
			# This is not uniformly sampling from all other points because we are sampling a cluster with equal probability, irrespective of number of points in that cluster
			# Sample a point from one of the other clusters
			otherCid = random.choice(list(canopy["clusters"].keys()))
			while otherCid == cid:
				otherCid = random.choice(list(canopy["clusters"].keys()))
			
			point_neg = random.choice(canopy["clusters"][otherCid])
			
			points.append(point)
			posPoints.append(point_pos)
			negPoints.append(point_neg)
		
		posDistances = self.model.batchForwardOneToOne(points, posPoints)
		negDistances = self.model.batchForwardOneToOne(points, negPoints)
		
		if self.config.threshold is None:
			posMinusNeg = posDistances - negDistances
			applyThreshold = torch.nn.Threshold(-1 * self.config.margin, 0)
			
			totalLoss = torch.sum(applyThreshold(posMinusNeg))  # Add all pos and neg edges to loss where pos edge > neg edge
			return totalLoss, 0, 0
		else:
			applyThreshold = torch.nn.Threshold(self.config.threshold - self.config.margin, 0)
			posDistances = applyThreshold(posDistances)  # Retain distances larger than threshold - margin
			
			applyThreshold = torch.nn.Threshold(-1 * (self.config.threshold + self.config.margin), 0)
			negDistances = -1 * applyThreshold(-1 * negDistances)  # Retain distances(edges) smaller than threshold + margin
			
			# Accumulate all positive and negative edges to compute loss
			posSampleLoss = torch.sum(posDistances)
			negSampleLoss = -1 * torch.sum(negDistances)
			
			totalLoss = posSampleLoss + negSampleLoss
			
			return totalLoss, posSampleLoss, negSampleLoss
	
	# Returns totalLoss as torch variables for cluster=cid and given canopy when training
	# without a threshold for general loss functions (other than triplet)
	def calcLoss_rankingStyle(self, canopy, currCid):
		
		if self.config.trainObj.startswith("linkage"):
			try:
				linkAlpha = float(self.config.trainObj.split("_")[-1])
			except ValueError:
				linkAlpha = self.config.trainObj.split("_")[-1]
			
			posLinkLoss, negLinkLoss = self.calcOverallHACLoss(canopy=canopy, linkAlpha=linkAlpha)
			return posLinkLoss + negLinkLoss
		else:
			
			currPoints = canopy["clusters"][currCid]
			numPoints = len(currPoints)
			acrossPoints = []
			
			for otherCid in canopy["clusters"]:  # TODO This can be sped up using intelligent slicing operations
				if otherCid != currCid:
					acrossPoints += canopy["clusters"][otherCid]
			
			acrossAdjMatrix = self.model.batchForwardAcross(currPoints, acrossPoints)
			if self.config.trainObj.endswith("minAcross"):
				acrossVector = torch.min(acrossAdjMatrix, dim=1)[0].view(-1, 1)
			elif self.config.trainObj.endswith("allAcross"):
				acrossVector = acrossAdjMatrix
			else:
				acrossVector = None
				raise Exception("Invalid loss for across cluster edges:{}".format(self.config.trainObj))
			
			if self.config.trainObj.startswith("mstWithin"):
				assert not self.config.trainObj.endswith(
					"allAcross")  # Loss for allAcross edges is not supported with mst edges
				loss = self.calcMSTLossRanking(cluster=currPoints, acrossVector=acrossVector)
			else:
				withinAdjMatrix = self.model.batchForwardWithin(currPoints)
				if self.config.trainObj.startswith("minWithin"):
					maxVal = torch.max(withinAdjMatrix)
					diag = torch.eye(numPoints)
					if self.config.useGPU: diag.data = diag.data.cuda()
					withinAdjMatrix = withinAdjMatrix + diag * maxVal  # Need to add large value to all diagonal so that it can be avoided when calculating min
					withinVector = torch.min(withinAdjMatrix, dim=1)[0].view(-1, 1)
				elif self.config.trainObj.startswith("maxWithin"):
					withinVector = torch.max(withinAdjMatrix, dim=1)[0].view(-1, 1)
				elif self.config.trainObj.startswith("allWithin"):
					withinVector = withinAdjMatrix
				else:
					withinVector = None
					raise Exception("Invalid loss for within cluster edges:{:.3f}".format(self.config.trainObj))
				
				# print("Time taken for computing within edges:{}".format(time.time() - t1))
				withinMinusAcross = withinVector - acrossVector
				applyMargin = torch.nn.Threshold(-1 * self.config.margin,
												 0)  # withinEdge should be less than acrossEdge - margin
				loss = torch.sum(applyMargin(withinMinusAcross))
			
			return loss
	
	############# Different Loss functions for across cluster edges ################################
	
	# Calculates (exact) minAcross cluster distance for all points in trainClusters
	def calcAllAcrossLoss(self, canopy, currCid):
		tempTensor = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		allAcrossLoss = Variable(tempTensor, requires_grad=True)
		
		acrossPoints = []
		for otherCid in canopy["clusters"]:
			if otherCid != currCid:
				acrossPoints += canopy["clusters"][otherCid]
		
		numPoints = len(canopy["clusters"][currCid])
		batchSize = calc_batch_size(len(acrossPoints), self.config.inputDim)
		numBatches = int(math.ceil(numPoints / batchSize))
		# print("BatchSize:{}\tNumBatches:{}\tnumPoints\t{}\tacrossPoints\t{}".format(batchSize, numBatches, numPoints, len(acrossPoints)))
		for batchNum in range(numBatches):
			startIdx = batchNum * batchSize
			endIdx = min(numPoints, (batchNum + 1) * batchSize)
			currPoints = canopy["clusters"][currCid][startIdx: endIdx]
			
			adjMatrix = self.model.batchForwardAcross(currPoints, acrossPoints)
			
			if self.config.threshold is None:
				allAcrossLoss = allAcrossLoss + torch.sum(adjMatrix)  # Add all edges to loss
			else:
				applyThreshold = torch.nn.Threshold(-1 * (self.config.threshold + self.config.margin), 0)
				adjMatrix = -1 * applyThreshold(-1 * adjMatrix)  # Retain edges smaller than threshold, acrossEdges larger than threshold should not be used to compute loss
				allAcrossLoss = allAcrossLoss + torch.sum(adjMatrix)
		
		return -1 * allAcrossLoss  # Multiplying -1 because we want to increase across edges as optimizer tries to minimize loss
	
	# Calculates (exact) minAcross cluster distance for all points in trainClusters
	def calcMinAcrossLoss(self, canopy, currCid):
		tempTensor = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		minDistAcrossLoss = Variable(tempTensor, requires_grad=True)
		
		acrossPoints = []
		for otherCid in canopy["clusters"]:
			if otherCid != currCid:
				acrossPoints += canopy["clusters"][otherCid]
		
		numPoints = len(canopy["clusters"][currCid])
		batchSize = calc_batch_size(len(acrossPoints), self.config.inputDim)
		numBatches = int(math.ceil(numPoints / batchSize))
		# print("BatchSize:{}\tNumBatches:{}\tnumPoints\t{}\tacrossPoints\t{}".format(batchSize, numBatches, numPoints, len(acrossPoints)))
		for batchNum in range(numBatches):
			startIdx = batchNum * batchSize
			endIdx = min(numPoints, (batchNum + 1) * batchSize)
			currPoints = canopy["clusters"][currCid][startIdx: endIdx]
			
			adjMatrix = self.model.batchForwardAcross(currPoints, acrossPoints)
			minDistVector, _ = torch.min(adjMatrix, dim=1)
			
			if self.config.threshold is None:
				minDistAcrossLoss = minDistAcrossLoss + torch.sum(minDistVector)  # Add all edges to loss
			else:
				applyThreshold = torch.nn.Threshold(-1 * (self.config.threshold + self.config.margin), 0)
				minDistVector = -1 * applyThreshold(-1 * minDistVector)  # Retain edges smaller than thres+margin, acrossEdges larger than thresh+margin should not be used to compute loss
				minDistAcrossLoss = minDistAcrossLoss + torch.sum(minDistVector)
		
		return -1 * minDistAcrossLoss  # Multiplying -1 because we want to increase across edges as optimizer tries to minimize loss
	
	# Calculates (exact) minAcross cluster distance for all points in trainClusters
	def calcMinAcrossLoss_slow(self, canopy, currCid):
		tempTensor = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		minDistAcrossLoss = Variable(tempTensor, requires_grad=True)
		
		acrossPoints = []
		for otherCid in canopy["clusters"]:
			if otherCid != currCid:
				acrossPoints += canopy["clusters"][otherCid]
		
		for point in canopy["clusters"][currCid]:
			adjVector = self.model.batchForwardAcross([point], acrossPoints)
			minDistAcross = torch.min(adjVector)
			
			if (self.config.threshold is None) or (minDistAcross < self.config.threshold + self.config.margin):
				minDistAcrossLoss = minDistAcrossLoss + minDistAcross
		
		return -1 * minDistAcrossLoss  # Multiplying -1 because we want to increase across edges and optimizer tries to minimize loss
	
	def calcApproxMinAcrossLoss(self, canopy, currCid):
		tempTensor = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		approxMinAcrossLoss = Variable(tempTensor, requires_grad=True)
		
		for point in canopy["clusters"][currCid]:  # TODO Speedup using batch computation
			# Choose a different cluster and find distance to closest point in this cluster(Approximation comes due to sampling this cluster)
			otherCid = random.choice(list(canopy["clusters"].keys()))
			while otherCid == currCid:
				otherCid = random.choice(list(canopy["clusters"].keys()))
			
			adjVector = self.model.batchForwardAcross([point], canopy["clusters"][otherCid])
			minDistAcross = torch.min(adjVector)
			
			if (self.config.threshold is None) or (minDistAcross < self.config.threshold + self.config.margin):
				approxMinAcrossLoss = approxMinAcrossLoss + minDistAcross
		
		return -1 * approxMinAcrossLoss  # Multiplying -1 because we want to increase across edges and optimizer tries to minimize loss
	
	############# Different Loss functions for within cluster edges ################################
	
	# Calculates (exact) minWithin cluster distance for each point in  trainClusters
	def calcMinWithinLoss(self, cluster):
		
		adjMatrix = self.model.batchForwardWithin(cluster)
		maxVal = torch.max(adjMatrix)
		diag = torch.eye(len(cluster))
		if self.config.useGPU: diag.data = diag.data.cuda()
		adjMatrix = adjMatrix + diag * maxVal  # Need to add large value to all diagonal so that it can be avoided when calculating min
		
		minVector = torch.min(adjMatrix, dim=1)[0]
		if self.config.threshold is not None:
			applyThreshold = torch.nn.Threshold(self.config.threshold - self.config.margin, 0)  # Remove edges which are smaller than thresh-margin, they should not be used to compute loss
			minVector = applyThreshold(minVector)
		
		minDistWithinLoss = torch.sum(minVector)
		
		return minDistWithinLoss
	
	# Calculates (exact) maxWithin cluster distance for each point in cluster
	def calcMaxWithinLoss(self, cluster):
		adjMatrix = self.model.batchForwardWithin(cluster)
		maxVector = torch.max(adjMatrix, dim=1)[0]
		if self.config.threshold is not None:
			applyThreshold = torch.nn.Threshold(self.config.threshold - self.config.margin, 0)
			maxVector = applyThreshold(maxVector)  # Remove edges which are smaller than thresh-margin, do not use them to compute loss
		maxDistWithinLoss = torch.sum(maxVector)
		
		return maxDistWithinLoss
	
	# Calculates (exact) minWithin cluster distance for each point in cluster
	def calcAllPairWithinLoss(self, cluster):
		
		adjMatrix = self.model.batchForwardWithin(cluster)
		if self.config.threshold is not None:
			applyThreshold = torch.nn.Threshold(self.config.threshold - self.config.margin, 0)
			adjMatrix = applyThreshold(adjMatrix)  # Remove edges which are smaller than threshold - margin, do not use them to compute loss
		allPairWithinLoss = torch.sum(adjMatrix)
		
		return allPairWithinLoss
	
	# Creates an MST over points in cluster(which is a list of points) and returns loss for mst edges which are greater than a threshold
	def calcMSTLoss(self, cluster):
		clusterAdjMatrix = self.model.batchForwardWithin(cluster)
		if self.config.useGPU:
			clusterAdjMatrix_NP = clusterAdjMatrix.cpu().data.numpy()
		else:
			clusterAdjMatrix_NP = clusterAdjMatrix.data.numpy()
		
		mst = minimum_spanning_tree(csr_matrix(clusterAdjMatrix_NP))
		mst = mst.todok()  # Get edges in minimum spanning tree
		
		mstLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		mstLoss = Variable(mstLoss, requires_grad=True)
		
		# Iterate over edges in MST and add loss for edges that were not predicted
		for (pid1, pid2) in mst.keys():
			# assert (clusterAdjMatrix[pid1][pid2].data == mst[(pid1, pid2)]) # Useful for debugging
			if (self.config.threshold is None) or (clusterAdjMatrix_NP[pid1][pid2] > self.config.threshold - self.config.margin):
				mstLoss = mstLoss + clusterAdjMatrix[pid1][pid2]
		
		return mstLoss
	
	def calcMSTLossRanking(self, cluster, acrossVector):
		assert self.config.threshold is None  # This function should only be called when using ranking style loss and threshold is None
		
		clusterAdjMatrix = self.model.batchForwardWithin(cluster)
		# print(clusterAdjMatrix.shape)
		# print(clusterAdjMatrix)
		clusterAdjMatrix_NP = clusterAdjMatrix.cpu().data.numpy()
		mst = minimum_spanning_tree(csr_matrix(clusterAdjMatrix_NP)).todok()  # Get edges in minimum spanning tree
		
		totalLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		totalLoss = Variable(totalLoss, requires_grad=True)
		
		# Iterate over edges in MST and add loss for edges that were not predicted
		for (pid1, pid2) in mst.keys():
			assert clusterAdjMatrix[pid1][pid2] == mst[(pid1, pid2)]
			if clusterAdjMatrix[pid1][pid2] - acrossVector[
				pid1] > -1 * self.config.margin:  # If MST edge (p1,p2) worse than bestAcross edge for p1 - margin
				totalLoss = totalLoss + clusterAdjMatrix[pid1][pid2] - acrossVector[pid1]
			# print("{}\t{} - {} = {}".format(pid1,clusterAdjMatrix[pid1][pid2], acrossVector[pid1], clusterAdjMatrix[pid1][pid2]  - acrossVector[pid1]))
			
			if clusterAdjMatrix[pid1][pid2] - acrossVector[
				pid2] > -1 * self.config.margin:  # If MST edge (p1,p2) worse than bestAcross edge for p2 - margin
				totalLoss = totalLoss + clusterAdjMatrix[pid1][pid2] - acrossVector[pid2]
		# print("{}\t{} - {} = {}\n".format(pid1, clusterAdjMatrix[pid1][pid2], acrossVector[pid2], clusterAdjMatrix[pid1][pid2] - acrossVector[pid2]))
		
		return totalLoss
	
	def calcOverallHACLoss(self, canopy, linkAlpha):
		
		start = time.time()
		allPoints = []
		pidToGtCluster = {}
		for pid in canopy["points"]:  # TODO: Precompute this if this takes time
			point, cid = canopy["points"][pid]
			allPoints += [point]
			pidToGtCluster[pid] = cid
		
		numPoints = len(allPoints)
		print("NumPoints:{}".format(numPoints))
		torchDistMat = self.model.batchForwardWithin(allPoints)
		distMat_NP = torchDistMat.cpu().data.numpy()
		
		if linkAlpha == "auto":
			linkAlphaTorch = self.model.linkAlpha
			linkAlpha = self.model.linkAlpha.data.cpu().numpy()[0]
			linkAlpha = float(linkAlpha)
		else:
			linkAlphaTorch = linkAlpha
		
		posLinkages, negLinkages = runHAC_torch_allEdges_faces(origDistMat=distMat_NP, origTorchDistMat=torchDistMat,
														 numPoints=numPoints, linkAlpha=linkAlpha,
														 linkAlphaTorch=linkAlphaTorch, pidToGtCluster=pidToGtCluster,
														 scaleDist=self.config.scaleDist, getBestImpure=False)
		
		posLinkLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		posLinkLoss = Variable(posLinkLoss, requires_grad=True)
		
		negLinkLoss = torch.cuda.FloatTensor([0.]) if self.config.useGPU else torch.FloatTensor([0.])
		negLinkLoss = Variable(negLinkLoss, requires_grad=True)
		
		t1 = time.time()
		posCtr, negCtr = 0, 0
		if self.config.threshold is None:
			posLinkLoss = torch.sum( posLinkages ) # Add all posLinkages to loss
			posCtr += posLinkages.shape[0]
		else:
			applyThreshold = torch.nn.Threshold(self.config.threshold - self.config.margin, 0) # Keep only those posLinkages that are greater than thresh - margin
			filteredLinks = applyThreshold(posLinkages)
			posLinkLoss =  posLinkLoss + torch.sum( filteredLinks )
			posCtr += filteredLinks.nonzero().shape[0]
		
		if (self.config.threshold is None):
			negLinkLoss = negLinkLoss + torch.sum(negLinkages)  # Add all negLinkages to loss
			negCtr += negLinkages.shape[0]
		else:
			applyThreshold = torch.nn.Threshold( -1*(self.config.threshold + self.config.margin), 0)
			filteredLinks  = -1*applyThreshold(-1*negLinkages) # Keep only those negLinkages that are smaller than thresh + margin
			negLinkLoss = negLinkLoss - torch.sum(filteredLinks)
			negCtr += filteredLinks.nonzero().shape[0]
		
		posLinkLoss = posLinkLoss / posCtr if posCtr != 0 else posLinkLoss
		negLinkLoss = negLinkLoss / negCtr if negCtr != 0 else negLinkLoss
		
		self.logger.info("\t\tPosLinkLoss = {} {}".format(posLinkLoss.cpu().data.numpy()[0], posCtr))
		self.logger.info("\t\tNegLinkLoss = {} {}".format(negLinkLoss.cpu().data.numpy()[0], negCtr))
		
		end = time.time()
		self.logger.info("\t\tTime taken in calcOverallHACLoss = {:.4f}".format(end - start))
		
		return posLinkLoss, negLinkLoss
	