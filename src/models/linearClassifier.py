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
from utils.Config import Config

class LinearClassifier(torch.nn.Module):
	"""docstring for Linear Classifier"""
	
	def __init__(self, config):
		super(LinearClassifier, self).__init__()
		assert isinstance(config, Config)
		self.config 		= config
		self.inputDim 		= config.inputDim # Dimension of vector for each point
		self.outputDim 		= 1
		
		self.seqModel = torch.nn.Sequential(
          torch.nn.Linear(self.inputDim,self.outputDim)
        )
		
		tempAlphaVal = np.random.normal(self.config.alphaInitMu, self.config.alphaInitSigma, 1)[0]
		if self.config.useGPU:
			self.linkAlpha = Variable(torch.cuda.FloatTensor([tempAlphaVal]), requires_grad=True)
		else:
			self.linkAlpha = Variable(torch.FloatTensor([tempAlphaVal]), requires_grad=True)
	
		
	def __str__(self):
		printStr = ""
		printStr += "-----------------Linear Classifier Parameters----------------------" + "\n"
		printStr += "linkAlpha:" + str(self.linkAlpha) + "\n"
		printStr += "inputDim::" + str(self.inputDim) + "\n"
		printStr += "output dissimilarity\t" + str(self.config.outDisSim) + "\n"
		printStr += "Layers::" + str(self.seqModel) + "\n"
		printStr += self.getWeightStr()

		printStr += "-------------------------------------------------------------------"
		return printStr
	
	def getWeightStr(self):
		weightStr = ""
		weightStr += "Weight::{}".format(self.seqModel[0].weight) + "\n"
		weightStr += "Bias::{}".format(self.seqModel[0].bias) + "\n"
		return weightStr
	
	def pairForward(self,pairFeature):
		if self.config.useGPU:
			pairFeature = Variable(torch.cuda.FloatTensor(pairFeature))
		else:
			pairFeature = Variable(torch.Tensor(pairFeature))

		prediction = self.seqModel(pairFeature)
		return prediction

	def pairBatchForward(self,pairFeatureList):
		if self.config.useGPU:
			pairFeatureList = Variable(torch.cuda.FloatTensor(pairFeatureList))
		else:
			pairFeatureList = Variable(torch.Tensor(pairFeatureList))
		
		prediction = self.seqModel(pairFeatureList)
		return prediction

class AvgLinearClassifier(LinearClassifier):
	
	def __init__(self, config):
		super(AvgLinearClassifier, self).__init__(config)
		biasPresent = self.seqModel[0].bias is not None
		self.updateNum = 0
		self.avgWeights = torch.nn.Linear(self.inputDim, self.outputDim, bias=biasPresent)
		
	def __str__(self):
		printStr = ""
		printStr += "-----------------Average Linear Classifier Parameters-----------------------------" + "\n"
		printStr += "linkAlpha::\t" + str(self.linkAlpha) + "\n"
		printStr += "inputDim::\t" + str(self.inputDim) + "\n"
		printStr += "output dissimilarity\t" + str(self.config.outDisSim) + "\n"
		printStr += "updateNum" + str(self.updateNum) + "\n"
		printStr += "Layers::" + str(self.seqModel) + "\n"
		printStr += self.getWeightStr()
		printStr += "-------------------------------------------------------------------"
		return printStr
	
	def getWeightStr(self):
		weightStr = ""
		weightStr += "Weight::{}".format(self.seqModel[0].weight) + "\n"
		weightStr += "Bias::{}".format(self.seqModel[0].bias) + "\n"
		
		weightStr += "Avg Weight::{}".format(self.avgWeights.weight.data) + "\n"
		weightStr += "Avg Bias::{}".format(self.avgWeights.bias.data)+ "\n"
		return weightStr
		
	# Average weights after making gradient update
	def updateAvgWeights(self):
		
		self.avgWeights.weight.data = self.updateNum * self.avgWeights.weight.data + self.seqModel[0].weight.data
		if self.avgWeights.bias is not None:
			self.avgWeights.bias.data = self.updateNum * self.avgWeights.bias.data + self.seqModel[0].bias.data
		
		self.updateNum += 1
		self.avgWeights.weight.data = self.avgWeights.weight.data / self.updateNum
		if self.avgWeights.bias is not None:
			self.avgWeights.bias.data = self.avgWeights.bias.data / self.updateNum
	
	def pairAvgBatchForward(self, pairFeatureList):
		if self.config.useGPU:
			pairFeatureList = Variable(torch.cuda.FloatTensor(pairFeatureList))
		else:
			pairFeatureList = Variable(torch.Tensor(pairFeatureList))
		
		prediction = self.avgWeights(pairFeatureList)
		return prediction

	def pairAvgForward(self,pairFeature):
		if self.config.useGPU:
			pairFeature = Variable(torch.cuda.FloatTensor(pairFeature))
		else:
			pairFeature = Variable(torch.Tensor(pairFeature))

		prediction = self.avgWeights(pairFeature)
		return prediction
	
