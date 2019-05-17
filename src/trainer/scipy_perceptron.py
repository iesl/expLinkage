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


import numpy as np,argparse
import torch
from pathlib import Path

from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from eval.evalPairFeat import get_conn_comp_pair_feat
from utils.Config import Config
from utils.plotting import plot_clusters_w_edges, plot_clusters
from eval.threshold import choose_threshold

from models.linearClassifier import LinearClassifier
from models.templateClassifier import Classifier
from PairFeatureTrainer import PairFeatureTrainer

def getBestClassifier(modelType,seed,X,Y):
	classifiers = {}
	np.random.seed(seed)
	for i in range(10):
		if modelType == "SVMLinear":
			clf = SGDClassifier(loss="hinge", penalty="l2", tol=1e-9, alpha=0.01, max_iter=1000)  # Linear SVM
		elif modelType == "SVMRbf":
			clf = SVC(gamma='auto', tol=1e-9, )
		elif modelType == "Perceptron":
			clf = SGDClassifier(loss="perceptron", penalty="l2", tol=1e-9, alpha=0.01, max_iter=1000)
		elif modelType == "AvgPerceptron":
			clf = SGDClassifier(loss="perceptron", penalty="l2", tol=1e-9, alpha=0.01, max_iter=1000,average=True)
		elif modelType == "MST":
			clf = Perceptron(random_state=config.seed, penalty="l2", max_iter=1000, alpha=0.01, tol=1e-5, warm_start=True,shuffle=True)
			clf.fit(X, Y) # Doint this to just to get other class variable initialized
			# Optimal parameters as learnt by MST objective
			clf.coef_ = np.array([[-0.092749, -0.076006]])
			clf.intercept_ = np.array([0.3871])
		else:
			raise Exception("Invalid Model:{}".format(modelType))
		
		# Initializing parameters. Need to set warm_start to True for this purpose. If shuffle is False then we get
		# same results for every random_state but if shuffle is True then we get different parameters because data is shuffled
		# at every iteration
		# clf = Perceptron(random_state=args.seed, penalty="l2", max_iter=1000, alpha=0.01, tol=1e-5, warm_start=True,shuffle=True)
		# clf.coef_ = np.array([[1,1]])
		# clf.intercept_ = np.array([0])
		# clf.fit(X, Y)
		# Optimal parameters as learnt by MST objective
		# clf.coef_ = np.array([[-0.092749, -0.076006]])
		# clf.intercept_ = np.array([0.3871])
		
		if modelType != "MST":
			clf.fit(X, Y)
		score = clf.score(X, Y)
		print("Accuracy on train data:{:.3f}".format(score))
		classifiers[i] = (clf,score)
	
	bestClf = None
	bestScore  = 0
	for i in classifiers.keys():
		if bestClf is None or bestScore < classifiers[i][1]:
			bestClf   = classifiers[i][0]
			bestScore = classifiers[i][1]
	
	print("Model with best Accuracy on train data:{:.3f}".format(bestScore))
	return bestClf

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser("Run Scipy perceptron on pairwise data(synthetic points in R2)")
	parser.add_argument('--config', type=str, help="Config file")
	
	temp_config = Config()
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
	
	trainer 	= PairFeatureTrainer(config)
	resultDir 	= trainer.config.resultDir
	
	X,Y = [],[]
	for canopyId in trainer.trainCanopies:
		canopy = trainer.trainCanopies[canopyId]
		for (p1,p2) in canopy["pairFeatures"]:
			X.append(canopy["pairFeatures"][(p1,p2)])
			label = 1 if canopy["pidToCluster"][p1] == canopy["pidToCluster"][p2] else 0
			Y.append(label)
	
	X, Y = np.array(X), np.array(Y)
	clf = getBestClassifier(config.model, config.seed, X, Y)
	
	if "spiral" in trainer.config.dataDir:
		pidToPoint = {}
		with open("{}/1/pidToPoint.txt".format(trainer.config.dataDir)) as f:
			for line in f:
				lineV = line.strip().split()
				pid, x1, x2 = int(lineV[0]), float(lineV[1]), float(lineV[2])
				pidToPoint[pid] = (x1, x2)
		
	
		if hasattr(clf,"coef_"):
			b 		= clf.intercept_[0]
			m1,m2 	= clf.coef_[0][0],clf.coef_[0][1]
			
			assert isinstance(trainer.model, LinearClassifier)
			trainer.model.seqModel[0].weight.data 	= torch.cuda.FloatTensor([[m1, m2]]) if config.useGPU else torch.FloatTensor([[m1, m2]])
			trainer.model.seqModel[0].bias.data 	= torch.cuda.FloatTensor([b]) if config.useGPU else torch.FloatTensor([b])
			optThresh = choose_threshold(trainer,"connComp", "1", trainer.trainCanopies)

			model 		= (m1, m2, b)
			optModel 	= (m1, m2, b - optThresh)
			plot_clusters_w_edges(trainer.trainCanopies, model, "{}/boundary_{}.pdf".format(resultDir, config.seed))
			plot_clusters_w_edges(trainer.trainCanopies, optModel, "{}/boundaryOpt_{}.pdf".format(resultDir, config.seed))
			# plotClustersEdges(trainer.trainCanopies, optModel, "{}/boundaryOptWithBase_{}.pdf".format(resultDir, config.seed), baseModel=model)
			plot_clusters_w_edges(trainer.trainCanopies, model, "{}/boundaryOptWithBase_{}.pdf".format(resultDir, config.seed), baseModel=optModel)
		elif isinstance(clf,SVC):
			trainer.model 		= Classifier(config)
			trainer.model.clf 	= clf
			optThresh = choose_threshold(trainer,"connComp", "1", trainer.trainCanopies)
			print("Opt threshold = {}".format(optThresh))
			
			plot_clusters_w_edges(trainer.trainCanopies, clf, "{}/boundary_{}.png".format(resultDir, config.seed))
		else:
			raise Exception("Invalid model:{}",clf)
	
		for canopyId in trainer.trainCanopies:
			canopy = trainer.trainCanopies[canopyId]
			pidToPredCluster = get_conn_comp_pair_feat(model=trainer.model, pairFeatures=canopy["pairFeatures"],
													   pidToCluster=canopy["pidToCluster"], threshold=optThresh)
			pointToPredCluster = {}
			pointToTrueCluster = {}
			for pid in pidToPredCluster:
				point = pidToPoint[pid]
				pointToPredCluster[point] = pidToPredCluster[pid]
				pointToTrueCluster[point] = canopy["pidToCluster"][pid]
			
			
			plot_clusters(pointToCluster=pointToPredCluster, filename=trainer.config.resultDir + "/predClusterOptThresh_{}.pdf".format(config.seed))
			plot_clusters(pointToCluster=pointToTrueCluster, filename=trainer.config.resultDir + "/trueCluster.pdf")
			
			pidToPredCluster = get_conn_comp_pair_feat(model=trainer.model, pairFeatures=canopy["pairFeatures"],
													   pidToCluster=canopy["pidToCluster"], threshold=0)
			pointToPredCluster = {}
			pointToTrueCluster = {}
			for pid in pidToPredCluster:
				point = pidToPoint[pid]
				pointToPredCluster[point] = pidToPredCluster[pid]
				pointToTrueCluster[point] = canopy["pidToCluster"][pid]
			plot_clusters(pointToCluster=pointToPredCluster, filename=trainer.config.resultDir + "/predClusterLearnt.pdf".format(config.seed))
			