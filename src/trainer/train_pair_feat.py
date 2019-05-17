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


import argparse, time, sys, os
from pathlib import Path
import torch


from utils.Config import Config
from utils.basic_utils import create_logger
from eval.finalEval import run_final_eval

from models.linearClassifier import LinearClassifier
from trainer.PairFeatureTrainer import PairFeatureTrainer

def trainExpLinkOnly(trainer):
	
	assert isinstance(trainer, PairFeatureTrainer)
	
	if trainer.config.trainObj == "linkage_auto":
		trainer.logger.info("Not training linkageAlpha separately because if trainObj is linakge_auto then it must be trained already...")
	elif (trainer.config.modelType == "avgLinear" or trainer.config.modelType == "linear"):
		
		if trainer.config.modelType == "avgLinear":
			newModel = LinearClassifier(trainer.config)
			newModel.seqModel[0].weight.data =  trainer.model.avgWeights.weight.data
			if trainer.model.seqModel[0].bias is not None:
				newModel.seqModel[0].bias.data = trainer.model.avgWeights.bias.data
			
			trainer.model = newModel
		elif trainer.config.modelType == "linear":
			newModel = LinearClassifier(trainer.config)
			newModel.seqModel[0].weight.data =  trainer.model.seqModel[0].weight.data
			if trainer.model.seqModel[0].bias is not None:
				newModel.seqModel[0].bias.data =  trainer.model.seqModel[0].bias.data
			
			trainer.model = newModel
		else:
			raise Exception("Invalid modelType..{}".format(trainer.config.modelType))
		
		if trainer.config.useGPU:
			trainer.logger.info("Shifting model to cuda because GPUs are available!")
			trainer.model = trainer.model.cuda()
		
		trainer.config.trainAlpha = True
		trainer.config.trainModel = False
		trainer.resetOptimizer()
		
		if "linkage_auto" not in trainer.config.inferenceMethods:
			trainer.config.inferenceMethods += ["linkage_auto"]
		if "linkage_auto@t" not in trainer.config.inferenceMethods:
			trainer.config.inferenceMethods += ["linkage_auto@t"]
		
		origCSVFile = "{}/origTraining/results.csv"
		fileCheck = Path(origCSVFile.format(trainer.config.resultDir))
		if not fileCheck.is_file():
			print("File does not exist:{}".format(origCSVFile))
			command = "cd {} && mkdir -p origTraining && cp *.csv origTraining/ && cp *.png origTraining/".format(trainer.config.resultDir)
			os.system(command)
			
		trainer.config.trainObj = "linkage_auto"
		trainer.logger.info("Training alpha parameter of expLink ...\n\n\n")
		trainer.logger.info(trainer.model)
		
		trainT1 = time.time()
		success = trainer.train()
		if success is not None and (not success):
			try:
				trainer.config.inferenceMethods.remove("linkage_auto@t")
				trainer.config.inferenceMethods.remove("linkage_auto")
			except:
				pass
		
		trainer.printModelWeights()
		
		trainer.config.bestModel = os.path.join(trainer.config.resultDir, "model_alpha.torch")
		torch.save(trainer.model, trainer.config.bestModel )
		trainer.config.save_config(trainer.config.resultDir, "config_expLink.json")
		
		trainT2 = time.time()
		trainer.logger.info("Training alpha parameter of expLink linkage ends in time={:.3f} = {:.3f} min = {:.3f} hr \n\n\n".format(trainT2 - trainT1,(trainT2 - trainT1)/60, (trainT2 - trainT1)/3600))
	else:
		trainer.logger.info("Not training linkageAlpha separately because if modelType is not linear or avgLinear... ")

def runMain(config):
	assert isinstance(config,Config)
	
	command = sys.argv
	start = time.time()
	
	if config.mode == "train":
		trainer = PairFeatureTrainer(config)
		trainer.logger.info(command)
		
		trainer.logger.info("Inital Weights of the model...")
		trainer.printModelWeights()
		
		
		trainT1 = time.time()
		trainer.train()
		trainT2 = time.time()
		trainer.logger.info("Training ends in time={:.3f} = {:.3f} min = {:.3f} hr Saving model".format(trainT2 - trainT1,(trainT2 - trainT1)/60,(trainT2 - trainT1)/3600))
		
		trainer.logger.info("Weights that the model converged to...")
		trainer.printModelWeights()
		
		trainer.config.bestModel = os.path.join(trainer.config.resultDir, "model.torch")
		torch.save(trainer.model, trainer.config.bestModel )
		trainer.config.save_config(trainer.config.resultDir)
		trainer.logger.info("Saved model...")
		
		if config.trainExpLink:
			trainExpLinkOnly(trainer)
		
	elif config.mode == "trainExpLink":
		trainer = PairFeatureTrainer(config)
		trainer.logger.info(command)
		
		# Load model and reset optimizer to have parameters of the loaded model
		trainer.loadModel()
		
		# Update output directory
		trainer.config.resultDir = trainer.config.resultDir + args.newDirSuffix
		Path(trainer.config.resultDir).mkdir(parents=True, exist_ok=True)  # Create resultDir directory if not already present
		
		# Update logger object
		trainer.logger = create_logger(config=config, logFile="logFile_trainExpLink.txt", currLogger=trainer.logger)
		
		trainer.logger.info(trainer)
		trainExpLinkOnly(trainer)
		
	elif config.mode == "test":
		trainer = PairFeatureTrainer(config)
		trainer.logger.info(command)
		
		# Load model and reset optimizer to have parameters of the loaded model
		trainer.loadModel()
		
		# Update output directory
		trainer.config.resultDir = trainer.config.resultDir + args.newDirSuffix
		Path(trainer.config.resultDir).mkdir(parents=True, exist_ok=True)  # Create resultDir directory if not already present
		
		# Update logger object
		trainer.logger = create_logger(config=config, logFile="logFile_retest.txt", currLogger=trainer.logger)
		
	else:
		raise Exception("Invalid mode = {}. Choose one from: test, train or trainExpLink".format(config.mode))
	
	
	t1 = time.time()
	run_final_eval(trainer)
	t2 = time.time()
	trainer.logger.info(" Total time taken for final evaluation = {:.4f} = {:.4f} min = {:.4f} hours".format(t2 - t1, (t2 - t1)/60, (t2 - t1)/3600))
	
	trainer.logger.info(trainer)
	trainer.logger.info(command)
	end = time.time()
	trainer.logger.info(" Total time taken  = {:.4f} = {:.4f} min = {:.4f} hours".format(end - start, (end - start)/60, (end - start)/3600))
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser( description='Supervised clustering training with features given on every pair of points')
	
	temp_config = Config()
	parser.add_argument('--config', type=str, help="Config file")
	################################## OPTIONAL ARGUMENTS TO OVERWRITE CONFIG FILE ARGS###################################################
	for config_arg in temp_config.__dict__:
		def_val = temp_config.__getattribute__(config_arg)
		arg_type = type(def_val) if def_val is not None else str
		parser.add_argument('--{}'.format(config_arg), type=arg_type, default=None, help='If not specified then value from config file will be used')
	#########################################################################################################

	args = parser.parse_args()
	
	assert args.config is not None
	config = Config(args.config)
	for config_arg in temp_config.__dict__:
		def_val = getattr(args, config_arg)
		if def_val is not None:
			
			old_val = config.__dict__[config_arg]
			config.__dict__.update({config_arg:def_val})
			new_val =config.__dict__[config_arg]
			print("Updating Config.{} from {} to {} using arg_val={}".format(config_arg, old_val, new_val, def_val))
	
	# Update result directory if there are any parameters passed through command line that are different from those in config file
	if args.resultDir is None:
		config.updateResultDir("auto")
	else:
		config.updateResultDir(args.resultDir)
	
	Path(config.resultDir).mkdir(parents=True, exist_ok=True)  # Create resultDir directory if not already present
	config.useGPU 		= config.cuda and torch.cuda.is_available()
	config.updateRandomSeeds(config.seed)
	config.save_config(config.resultDir, "orig_config.json")
	
	runMain(config)
	