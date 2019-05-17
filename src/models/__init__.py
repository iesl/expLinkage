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
from models.linearClassifier import LinearClassifier, AvgLinearClassifier
from models.templateClassifier import Classifier
from models.mahalabonis import GenLinkMahalanobis

def create_new_model(config):
	""" Create a new model object based on the modelType field in the config

    :param config:
    :return: New created object
    """
	assert isinstance(config,Config)
	if config.modelType == "linear": # Learn a pairwise classifier
		model = LinearClassifier(config)
	elif config.modelType == "avgLinear":  # Learn a pairwise classifier and uses avgWeights at the end of training
		model = AvgLinearClassifier(config)
	elif config.modelType == "maha":
		model = GenLinkMahalanobis(config)
	elif config.modelType == "template":
		model = Classifier(config)  # This class is just a template to use with skLearn classifiers with current code setup
	else:
		raise Exception("Unknown Model: {}".format(config.modelType))

	return model
	
# def  load_model(config):
# 	""" Load model object using the bestModel field in the config
#
#     :param config:
#     :return:Loaded Model Object
#     """
# 	assert isinstance(config,Config)
# 	if config.modelType == "linear": # Learn a pairwise classifier
# 		model = LinearClassifier.load(config.bestModel)
# 	elif config.modelType == "avgLinear":  # Learn a pairwise classifier and uses avgWeights at the end of training
# 		model = AvgLinearClassifier.load(config.bestModel)
# 	elif config.modelType == "template":
# 		model = Classifier()  # This class is just a template to use with skLearn classifiers with current code setup
# 	else:
# 		raise Exception("Unknown Model: {}".format(config.modelType))
#
# 	return model
#