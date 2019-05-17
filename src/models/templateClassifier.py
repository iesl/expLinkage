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
import numpy as np

class Classifier(torch.nn.Module):
	
	def __init__(self,config):
		super(Classifier, self).__init__()
		
		self.config = config
		self.clf = None
		self.seqModel = torch.nn.Sequential(
          torch.nn.Linear(self.config.inputDim,self.config.inputDim)
        )
		
	def __str__(self):
		printStr = ""
		printStr += "-----------------Classifier Parameters-----------------------------" + "\n"
		printStr += str(self.clf)
		printStr += "-------------------------------------------------------------------"
		return printStr
	
	def getWeightStr(self):
		return "\n\nNo parameters\n\n"
	
	def pairForward(self, pairFeature):
		raise NotImplementedError
		# prediction = self.clf.predict(pairFeature)
		# return torch.autograd.Variable(torch.FloatTensor(prediction),requires_grad=False)
	
	def pairBatchForward(self, pairFeatureList):
		prediction = self.clf.predict(pairFeatureList)
		prediction = torch.FloatTensor(prediction).view(-1,1)
		return torch.autograd.Variable(prediction, requires_grad=False)
	
	def forward(self, point1, point2):
		raise NotImplementedError
	
	# This function does not return a pytorch Variable.
	# Just the distance between point1 and point2 as per current model
	def forwardPlain(self, point1, point2):
		raise NotImplementedError
	
	# Takes list of points and returns an adjacency matrix for it of size n x n
	def batchForwardWithin(self, points):
		raise NotImplementedError
	
	# Takes list of 2 points and returns an adjacency matrix for them of size n1 x n2
	def batchForwardAcross(self, pointList1, pointList2):
		raise NotImplementedError
	
	def batchForwardOneToOne(self, pointList1, pointList2):
		raise NotImplementedError
		


if __name__ == '__main__':
	torch.manual_seed(2)
	np.random.seed(1)
	print("There is no code to run here...")