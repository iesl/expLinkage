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


from eval.evalPairFeat import eval_model_pair_feat_per_canopy
from models.mahalabonis import MahalanobisDist
from models.linearClassifier import AvgLinearClassifier, LinearClassifier
from eval.threshold import choose_threshold
from utils.plotting import write_scores_comb,write_scores_separate, plot_scores_per_canopy, plot_scores

# Perform final evaluation of model
def run_final_eval(trainer):
	
	# assert isinstance(trainer, VectDataTrainer) or isinstance(trainer, PairFeatureTrainer)
	
	trainer.logger.info("Choosing best threshold for evaluation in the end...")
	if isinstance(trainer.model, AvgLinearClassifier):
		trainer.logger.info("Loading average weights")
		trainer.model.seqModel[0].weight.data =  trainer.model.avgWeights.weight.data
		if trainer.model.seqModel[0].bias is not None:
			trainer.model.seqModel[0].bias.data =  trainer.model.avgWeights.bias.data
	
	trainer.logger.info("Weights being used for performing evaluation...")
	trainer.printModelWeights()

	trainer.config.threshold = None  # Uncomment this line if you want to chooseThreshold
	############################### Choose threshold based on dev canopy######################################
	threshDict = {}
	for method in trainer.config.inferenceMethods:
		threshDict[method] = choose_threshold(trainer, infMethod=method,  epoch="END_BestDev")
	
	trainer.logger.info("Using dev thresholdVals:{}".format(threshDict))
	eval_all_data(trainer, threshDict,  "/BestDevThresh")
	###########################################################################################################
	
	############################### Choose threshold based on test canopy######################################
	if len(trainer.testCanopies) > 0 and trainer.config.evalOnTestThresh:
		threshDict = {}
		for method in trainer.config.inferenceMethods:
			threshDict[method] = choose_threshold(trainer, infMethod=method, epoch="END_BestTest", canopies=trainer.testCanopies)
		
		trainer.logger.info("Using test thresholdVals:{}".format(threshDict))
		eval_all_data(trainer, threshDict, "/BestTestThresh")
	###########################################################################################################
	
	##################################### Choose threshold based on train canopy##############################
	if trainer.config.evalOnTrainThresh:
		threshDict = {}
		for method in trainer.config.inferenceMethods:
			threshDict[method] = choose_threshold(trainer, infMethod=method, epoch="END_BestTrain", canopies=trainer.trainCanopies)
		
		trainer.logger.info("Using train thresholdVals:{}".format(threshDict))
		eval_all_data(trainer, threshDict, "/BestTrainThresh")
	###########################################################################################################
	pass
	
def eval_all_data(trainer, threshDict, relResultDir = None):
	allScores = {"train": {}, "test": {}, "dev": {}}

	# Not using config.infertenceMethods as sometimes we want to just evaluate on just 1 inference methods during training
	infMethods  = [method for method in threshDict.keys()]
	
	allScores["test"][0]  = trainer.evalFunc(config=trainer.config,model=trainer.model, canopies=trainer.testCanopies, threshDict=threshDict,
									  inferenceMethods=infMethods, metricsForEval=trainer.config.metricsForEval)
	
	allScores["dev"][0]   = trainer.evalFunc(config=trainer.config,model=trainer.model, canopies=trainer.devCanopies, threshDict=threshDict,
									 inferenceMethods=infMethods, metricsForEval=trainer.config.metricsForEval)
	
	allScores["train"][0] = trainer.evalFunc(config=trainer.config,model=trainer.model, canopies=trainer.trainCanopies, threshDict=threshDict,
									   inferenceMethods=infMethods, metricsForEval=trainer.config.metricsForEval)
	
	if relResultDir is not None:
		if trainer.config.makeScorePlots:
			plot_scores(allLosses={"train":{}, "test":{}, "dev":{}}, allScores=allScores,
					currResultDir=trainer.config.resultDir + relResultDir, xlabel="Threshold")
		
		write_scores_comb(allLosses={"train": {}, "test": {}, "dev": {}}, allScores=allScores,
						  currResultDir=trainer.config.resultDir + relResultDir, xlabel="Threshold")
		
		write_scores_separate(allLosses={"train": {}, "test": {}, "dev": {}}, allScores=allScores,
							  currResultDir=trainer.config.resultDir + relResultDir, xlabel="Threshold")
	
	return allScores["train"][0], allScores["test"][0], allScores["dev"][0]

def run_final_eval_per_canopy(trainer):
	
	from trainer.PairFeatureTrainer import PairFeatureTrainer
	assert isinstance(trainer, PairFeatureTrainer)

	trainer.logger.info("Choosing optimal threshold and running model with average weights for that...".format())

	if isinstance(trainer.model, AvgLinearClassifier):
		trainer.model.seqModel[0].weight.data =  trainer.model.avgWeights.weight.data
		if trainer.model.seqModel[0].bias is not None:
			trainer.model.seqModel[0].bias.data =  trainer.model.avgWeights.bias.data

	if isinstance(trainer.model, MahalanobisDist) or isinstance(trainer.model, LinearClassifier):
		trainer.logger.info("Weights being used for performing evalutaion...")
		trainer.logger.info("Weight::{}".format(trainer.model.seqModel[0].weight))
		trainer.logger.info("Bias::{}".format(trainer.model.seqModel[0].bias))

	trainer.logger.info("Choosing best threshold for evaluation in the end...")

	trainer.config.threshold = None  # Uncomment this line if you want to chooseThreshold

	############################### Choose threshold based on dev canopy######################################
	threshDict = {}
	for method in trainer.config.inferenceMethods:
		threshDict[method] = choose_threshold(trainer, infMethod=method, epoch="END_BestDev")

	trainer.logger.info("Using dev thresholdVals:{}".format(threshDict))
	eval_all_data_per_canopy(trainer, threshDict, "/BestDevThresh")
	###########################################################################################################


	############################### Choose threshold based on test canopy######################################
	if len(trainer.testCanopies) > 0:
		threshDict = {}
		for method in trainer.config.inferenceMethods:
			threshDict[method] = choose_threshold(trainer, infMethod=method, epoch="END_BestTest",
												  canopies=trainer.testCanopies)

		trainer.logger.info("Using test thresholdVals:{}".format(threshDict))
		eval_all_data_per_canopy(trainer, threshDict, "/BestTestThresh")
	###########################################################################################################

	####################################3# Choose threshold based on train canopy##############################
	threshDict = {}
	for method in trainer.config.inferenceMethods:
		threshDict[method] = choose_threshold(trainer, infMethod=method, epoch="END_BestTrain",
											  canopies=trainer.trainCanopies)

	trainer.logger.info("Using train thresholdVals:{}".format(threshDict))
	eval_all_data_per_canopy(trainer, threshDict, "/BestTrainThresh")
	###########################################################################################################
	pass

def eval_all_data_per_canopy(trainer, threshDict, relResultDir):
	allScores = {}
	
	# def eval_model_pair_feat_per_canopy(model, canopies, inferenceMethods, threshDict, logger, metricsForEval)
	
	allScores["test"] = eval_model_pair_feat_per_canopy(model=trainer.model, canopies=trainer.testCanopies, logger=trainer.logger,
												 threshDict=threshDict, inferenceMethods=trainer.config.inferenceMethods, metricsForEval=trainer.config.metricsForEval)

	allScores["dev"] = eval_model_pair_feat_per_canopy(model=trainer.model, canopies=trainer.devCanopies, logger=trainer.logger,
												threshDict=threshDict, inferenceMethods=trainer.config.inferenceMethods, metricsForEval=trainer.config.metricsForEval)

	allScores["train"] = eval_model_pair_feat_per_canopy(model=trainer.model, canopies=trainer.trainCanopies, logger=trainer.logger,
												  threshDict=threshDict, inferenceMethods=trainer.config.inferenceMethods, metricsForEval=trainer.config.metricsForEval)

	plot_scores_per_canopy(allScores=allScores, currResultDir=trainer.config.resultDir + relResultDir)
