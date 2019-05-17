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

from utils.Config import Config
import torch
import os

class BaseTrainer(object):
	"""docstring for Base Trainer Class"""

	def __init__(self, config):
		super(BaseTrainer, self).__init__()
		
		assert isinstance(config, Config)
		self.config 	= config
		self.logger 	= None
		self.optimizer 	= None
		self.trainCanopies 	= {}
		self.testCanopies 	= {}
		self.devCanopies 	= {}
	
	def __str__(self):
		return "Base Trainer Class"
	
	def train(self):
		raise NotImplementedError
	
	def loadModel(self):
		# Load model and reset optimizer to have parameters of the loaded model
		if os.path.isfile(self.config.bestModel):
			self.model = torch.load(self.config.bestModel)
			self.logger.info("Loading model from:{}".format(self.config.bestModel))
		else:
			bestModel = os.path.join(self.config.resultDir, self.config.bestModel)
			if os.path.isfile(bestModel):
				self.model = torch.load(bestModel)
				self.logger.info("Loading model from:{}".format(bestModel))
			else:
				try:
					bestModel = os.path.join(self.config.resultDir, "model_alpha.torch")
					self.model = torch.load(bestModel)
					self.logger.info("Loading model from:{}".format(bestModel))
				except:
					bestModel = os.path.join(self.config.resultDir, "model.torch")
					self.model = torch.load(bestModel)
					self.logger.info("Loading model from:{}".format(bestModel))
				
		self.resetOptimizer()
		
	def resetOptimizer(self):
		
		if self.config.trainObj == "linkage_auto":
			assert self.config.trainModel and self.config.trainAlpha
		
		if self.config.trainModel and self.config.trainAlpha : # Add model.seqModel parameters and linkAlpha to the  optimizer
			assert self.config.trainObj == "linkage_auto"
			self.optimizer = torch.optim.Adam([{'params': self.model.seqModel.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.l2Alpha}])
			self.optimizer.add_param_group({'params': self.model.linkAlpha, 'lr': self.config.alphaLr})
			
		elif (not self.config.trainModel) and self.config.trainAlpha: # Add linkAlpha to the  optimizer
			self.optimizer = torch.optim.Adam([{'params': self.model.linkAlpha, "lr": self.config.alphaLr}])
			
		elif self.config.trainModel and (not self.config.trainAlpha): # Add model.seqModel parameters to optimizer
			assert self.config.trainObj != "linkage_auto"
			self.optimizer = torch.optim.Adam([{'params': self.model.seqModel.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.l2Alpha}])
		
		else:
			self.optimizer = torch.optim.Adam()
		
	