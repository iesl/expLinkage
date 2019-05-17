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

import torch
from torch.autograd import Variable
import numpy as np


class MahalanobisDist(torch.nn.Module):
	
	def __init__(self, config):
	
		super(MahalanobisDist, self).__init__()
		self.config 	= config
		self.inputDim 	= config.inputDim
		self.outputDim 	= self.inputDim
	
		self.seqModel = torch.nn.Sequential(
          torch.nn.Linear(self.inputDim,self.outputDim, bias=False)
        )

		if config.idenInit: # Initialize with Identity Matrix
			self.seqModel[0].weight.requires_grad = False
			self.seqModel[0].weight.data = torch.eye(self.config.inputDim)
			self.seqModel[0].weight.requires_grad = True
			
	def __str__(self):
		printStr = ""
		printStr += "-----------------Mahalanobis Distance Learner Parameters-----------------------------" + "\n"
		printStr += "inputDim::\t" + str(self.inputDim) + "\n"
		printStr += "Layers::" + str(self.seqModel) + "\n"
		printStr += "Parameters::" + str(list(self.parameters())) + "\n"
		printStr += "-------------------------------------------------------------------"
		return printStr
	
	def getWeightStr(self):
		weightStr = "Weight::{}".format(self.seqModel[0].weight)
		return weightStr
		
	# Returns numpy array after transforming every point according to Mahalanobis distance matrix
	def transformPoints(self, pointList):

		if self.config.useGPU:
			pointList = torch.cuda.FloatTensor(pointList)
		else:
			pointList = torch.Tensor(pointList)
		transformedPointList = self.seqModel(pointList)
		if self.config.useGPU:
			transformedPointList   = transformedPointList.cpu().data.numpy()
		else:
			transformedPointList = transformedPointList.data.numpy()

		return transformedPointList
	
	def pairForward(self, pairFeature):
		if self.config.useGPU:
			pairFeature = Variable(
				torch.cuda.FloatTensor(pairFeature))  # take difference of two vectors to send as input
		else:
			pairFeature = Variable(torch.Tensor(pairFeature))  # take difference of two vectors to send as input
		
		prediction = torch.norm(self.seqModel(pairFeature),p=2).view(1)
		return prediction
	
	def pairBatchForward(self, pairFeatureList):
		listLen = len(pairFeatureList)
		if self.config.useGPU:
			pairFeatureList = Variable(
				torch.cuda.FloatTensor(pairFeatureList))  # take difference of two vectors to send as input
		else:
			pairFeatureList = Variable(torch.Tensor(pairFeatureList))  # take difference of two vectors to send as input
		
		prediction = torch.norm(self.seqModel(pairFeatureList),dim=1,p=2).view(listLen,1)
		assert prediction.shape == torch.Size([listLen,1])
		return prediction
	
	def forward(self, point1, point2):

		if self.config.useGPU:
			p1 = torch.cuda.FloatTensor(point1)
			p2 = torch.cuda.FloatTensor(point2)
		else:
			p1 = torch.Tensor(point1)
			p2 = torch.Tensor(point2)

		embed1 = self.seqModel(p1)
		embed2 = self.seqModel(p2)
		distance = torch.norm(embed1 - embed2,p=2)
		return distance

	# This function does not return a pytorch Variable.
	# Just the Mahalabonis distance between point1 and point2
	def forwardPlain(self, point1, point2):

		distance = self.forward(point1, point2)
		if self.config.useGPU:
			distance = distance.cpu().data.numpy()
		else:
			distance = distance.data.numpy()
		return distance

	# Takes list of points and returns an adjacency matrix for it of size n x n
	def batchForwardWithin(self, points):
		numPoints = len(points)
		if self.config.useGPU:
			pointList1 = torch.cuda.FloatTensor(points)
			pointList2 = torch.cuda.FloatTensor(points)
		else:
			pointList1 = torch.Tensor(points)
			pointList2 = torch.Tensor(points)

		embedList1 = self.seqModel(pointList1).view(numPoints, 1, self.outputDim)
		embedList2 = self.seqModel(pointList2).view(1, numPoints, self.outputDim)

		# Use broadcasting feature to get nXn matrix where (i,j) contains ||p_i - p_j||_2
		distMatrix = torch.norm(embedList1 - embedList2, p=2, dim=2).view(numPoints, numPoints)

		return distMatrix

	# Takes list of 2 points and returns an adjacency matrix for them of size n1 x n2
	def batchForwardAcross(self, pointList1, pointList2):
		numPoint1 = len(pointList1)
		numPoint2 = len(pointList2)
		if self.config.useGPU:
			pointList1 = torch.cuda.FloatTensor(pointList1)
			pointList2 = torch.cuda.FloatTensor(pointList2)
		else:
			pointList1 = torch.Tensor(pointList1)
			pointList2 = torch.Tensor(pointList2)

		embedList1 = self.seqModel(pointList1).view(numPoint1, 1, self.outputDim)
		embedList2 = self.seqModel(pointList2).view(1, numPoint2, self.outputDim)

		# Use broadcasting feature to get nXn matrix where (i,j) contains ||p_i - p_j||_2
		distMatrix = torch.norm(embedList1 - embedList2, p=2, dim=2).view(numPoint1, numPoint2)
		return distMatrix

	# Returns distance between corresponding points in list 1 and list 2
	def batchForwardOneToOne(self, pointList1, pointList2):
		assert (len(pointList1) == len(pointList2))
		numPoints = len(pointList1)
		if self.config.useGPU:
			pointList1 = torch.cuda.FloatTensor(pointList1).view(numPoints, self.inputDim)
			pointList2 = torch.cuda.FloatTensor(pointList2).view(numPoints, self.inputDim)
		else:
			pointList1 = torch.Tensor(pointList1).view(numPoints, self.inputDim)
			pointList2 = torch.Tensor(pointList2).view(numPoints, self.inputDim)

		embedList1 = self.seqModel(pointList1)
		embedList2 = self.seqModel(pointList2)

		distMatrix = (torch.norm(embedList1 - embedList2, p=2, dim=1)).view(numPoints, 1)
		return distMatrix

class GenLinkMahalanobis(MahalanobisDist):
	
	def __init__(self, config):
		super(GenLinkMahalanobis, self).__init__(config)
		
		tempAlphaVal = np.random.normal(self.config.alphaInitMu, self.config.alphaInitSigma, 1)[0]
		if self.config.useGPU:
			self.linkAlpha = Variable(torch.cuda.FloatTensor([tempAlphaVal]), requires_grad=True)
		else:
			self.linkAlpha = Variable(torch.FloatTensor([tempAlphaVal]), requires_grad=True)
		
	def __str__(self):
		printStr = ""
		printStr += "-----------------General Linkage with Mahalanobis Distance Matrix: Parameters-----------------------------" + "\n"
		printStr += "linkAlpha::\t" + str(self.linkAlpha) + "\n"
		printStr += "inputDim::\t" + str(self.inputDim) + "\n"
		printStr += "Layers::" + str(self.seqModel) + "\n"
		printStr += "Parameters::" + str(list(self.parameters())) + "\n"
		printStr += "-------------------------------------------------------------------\n"
		return printStr
	