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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# Recursive search
# TODO: Change to just work for F1. Remove f1ToEval argument
def choose_threshold(trainer, infMethod,  epoch="0", canopies=None):
	
	f1ToEval="f1"
	if infMethod != "connComp" and (not infMethod.endswith("@t")):
		trainer.logger.info("Can not choose threshold for infMethod = {}".format(infMethod))
		return 0.
	
	printLog = True
	if canopies is None:
		if len(trainer.devCanopies) != 0:
			canopies = trainer.devCanopies
		else:
			canopies = trainer.trainCanopies
	else:
		pass
	
	# Precison , recall and f1 to use when finding bestTHreshold. Alaternatively, we could use "connComp_muc_precision" etc
	if f1ToEval == "muc_f1":
		recallToUse = "{}_muc_recall".format(infMethod)
		precisionToUse = "{}_muc_precision".format(infMethod)
	elif f1ToEval == "f1":
		recallToUse = "{}_recall".format(infMethod)
		precisionToUse = "{}_precision".format(infMethod)
	else:
		recallToUse = None
		precisionToUse = None
		raise Exception("Invalid f1ToUse={} to choose threshold".format(f1ToEval))
	
	start = time.time()
	trainer.logger.info("==" * 20 + "Beginning choosing threshold for method ={}".format(infMethod))
	
	currThreshold = 0.128
	precision = 0
	allScores = {"{}_{}".format(infMethod, metric): {} for metric in  ["precision","recall", "f1", "muc_precision","muc_recall", "muc_f1"]}
	f1Metric = "{}_{}".format(infMethod, f1ToEval)
	
	while precision != 1:
		scores = trainer.evalFunc(config=trainer.config, model=trainer.model, canopies=canopies,
						  threshDict={infMethod:currThreshold}, inferenceMethods=[infMethod], metricsForEval=f1ToEval)
		
		precision = scores[precisionToUse][0]
		for metric in allScores:
			allScores[metric][currThreshold] = scores[metric][0] if metric in scores else 0
		
		if printLog: trainer.logger.info("Precision:{}\t Threshold:{:.3f}".format(precision, currThreshold))
		if trainer.config.outDisSim: # Decreasing threshold to get better precision as model outputs distance
			if currThreshold < 0:
				currThreshold *= 2 # It is a negative number and making it smaller by multiplying it by 2
			elif currThreshold > 0.0001:
				currThreshold /= 2 # It is a positive number and making it smaller by dividing it by 2
			elif currThreshold == 0.0:
				currThreshold = -0.128 # Assign a small negative value to move away from zero
			else:# Switch over from very small positive to very small negative to continue making threshold smaller
				currThreshold = 0.
		
		else:  # Increasing threshold to get better precision as model outputs similarity
			if currThreshold > 0: # It is already positive and make it larger by multiplying by 2
				currThreshold *= 2
			elif currThreshold < -0.0001: # If it is negative then make it larger by dividing by 2
				currThreshold /= 2
			elif currThreshold == 0.0:
				currThreshold = 0.128 # Assign a small positive value to move away from zero
			else:  # Switch over from very small negative to very small positive to continue making threshold larger
				currThreshold = 0.
				
		
	
	bestRecall = -1
	theshForBestRecall = None
	for threshold in allScores[recallToUse]:
		if allScores[recallToUse][threshold] > bestRecall:
			bestRecall = allScores[recallToUse][threshold]
			theshForBestRecall = threshold
			
	if printLog: trainer.logger.info(" Best Recall:{:.3f}\t Threshold:{:.3f}".format(bestRecall, theshForBestRecall))
	
	if bestRecall != 1:
		currRecall = bestRecall
		currThreshold = theshForBestRecall
		while currRecall != 1:
			scores = trainer.evalFunc(config=trainer.config,model=trainer.model, canopies=canopies, threshDict={infMethod:currThreshold},
										  inferenceMethods=[infMethod], metricsForEval=f1ToEval)
			for metric in allScores:
				allScores[metric][currThreshold] = scores[metric][0] if metric in scores else 0
			
			currRecall = allScores[recallToUse][currThreshold]
			if printLog: trainer.logger.info("Recall:{:.3f}\t Threshold:{:.3f}".format(currRecall, currThreshold))
			
			if trainer.config.outDisSim: # Increasing threshold to get better recall as model outputs distance
				if currThreshold > 0:  # If positive already, the multiply by 2 to make it larger
					currThreshold *= 2
				elif currThreshold < -0.0001: # It negative then divide by 2 to make it larger
					currThreshold /= 2
				elif currThreshold == 0: # Assign a small positive value to move away from zero
					currThreshold = 0.128
				else: # It too small negative, then switch over from negative to positive to continue making it larger
					currThreshold = 0.
			else:  # Decrease threshold as it gives better recall
				if currThreshold < 0: # If negative, then making threshold smaller by n[multiplying it by 2
					currThreshold *= 2
				elif currThreshold > 0.0001:  # If positive, then making threshold smaller by dividing it by 2
					currThreshold /= 2
				elif currThreshold == 0.0: # Assign a small negative value to move away from zero
					currThreshold = -0.128
				else: # It too small positive, then switch over from positive to negative to continue making it smaller
					currThreshold = 0.
					
	''' Each time, I find threshold values between which the f1 score peaks. Then I try threshold values between those bounds
	and repeat the same procedure until: F1 at t1,t2 and (t1+t2)/2 is all same when rounded by 2 decimals or I have done this
	recursive search for more than 4 times
	'''
	
	bestF1 = -1
	threshForBestF1 = None
	for threshold in allScores[f1Metric]:
		if allScores[f1Metric][threshold] > bestF1:
			bestF1 = allScores[f1Metric][threshold]
			threshForBestF1 = threshold
	
	if threshForBestF1 == 1:
		return threshForBestF1
	
	# Try random values between thresh for best recall and thresh for best precision
	if printLog: trainer.logger.info("AllScores:{}".format(allScores))
	
	t1 = sorted(allScores[f1Metric].keys())[0]
	t2 = sorted(allScores[f1Metric].keys())[-1]
	numIntermediateThresh = 50
	thresholdValsToTry = np.arange(t1, t2, (t2  - t1) / numIntermediateThresh)
	if printLog: trainer.logger.info("Trying some additional threshold values between largest and smallest tried so far:{}".format(thresholdValsToTry))
	for thresh in thresholdValsToTry:
		scores = trainer.evalFunc(config=trainer.config,model=trainer.model, canopies=canopies, threshDict={infMethod:thresh},
									  inferenceMethods=[infMethod], metricsForEval=f1ToEval)
		for metric in allScores:
			allScores[metric][thresh] = scores[metric][0] if metric in scores else 0
		
	numRecSearch = 0
	while numRecSearch <= 6:
		numRecSearch += 1
		thresholdVals = sorted(list(allScores[f1Metric].keys()))
		if len(thresholdVals) == 1:
			if printLog: trainer.logger.info("Best threshold found in just 1 attempt:{}\t{}".format(thresholdVals[0], allScores[f1Metric]))
			break
		
		assert len(thresholdVals) >= 2
		
		bestThreshold = thresholdVals[0]
		for threshTried in thresholdVals:  # Choose threshold that gave best F1 on dev set
			if printLog: trainer.logger.info("{}\tThreshold:{:.3f}\tF1:{:.6f}".format(numRecSearch, threshTried, allScores[f1Metric][threshTried]))
			if allScores[f1Metric][threshTried] >= allScores[f1Metric][bestThreshold]:
				bestThreshold = threshTried
		
		lowerThreshold = thresholdVals[0]
		upperThreshold = thresholdVals[-1]
		
		prevThreshold = None
		for threshTried in thresholdVals:
			if prevThreshold == bestThreshold:
				upperThreshold = threshTried  # Threshold immediately AFTER the threshold that gives best F1
			
			if threshTried == bestThreshold:
				# Threshold immediately BEFORE the threshold that gives best F1
				lowerThreshold = prevThreshold if prevThreshold is not None else bestThreshold
			
			prevThreshold = threshTried
		
		# Push upperThreshold to as large as possible such that it still stays immediately next to best F1
		thresholdVals = sorted(thresholdVals)
		for ctr, threshold in enumerate(thresholdVals):
			if threshold < upperThreshold: continue
			if allScores[f1Metric][upperThreshold] == allScores[f1Metric][bestThreshold]:
				if ctr < len(thresholdVals) - 1:
					upperThreshold = thresholdVals[ctr + 1]
			else:
				break
		
		# Push lowerThreshold to as large as possible such that it still stays immediately before best F1
		# thresholdVals = sorted(thresholdVals, reverse=True)
		# for ctr,threshold in enumerate(thresholdVals):
		# 	if threshold > lowerThreshold: continue
		# 	if allScores[f1ToUse][lowerThreshold] == allScores[f1ToUse][bestThreshold]:
		# 		if ctr < len(thresholdVals)-1:
		# 			lowerThreshold = thresholdVals[ctr+1]
		# 	else:
		# 		break
		
		if printLog: trainer.logger.info("Upper Threshold:{:.3f} Lower Threshold:{:.3f}".format(upperThreshold, lowerThreshold))
		
		numIntermediateThresh = int(20 / numRecSearch)
		thresholdValsToTry = np.arange(lowerThreshold, upperThreshold,
									   (upperThreshold - lowerThreshold) / numIntermediateThresh)
		if printLog: trainer.logger.info("Threshold Values to try:{}".format(["{:.3f}".format(x) for x in thresholdValsToTry]))
		for currThreshold in thresholdValsToTry:
			scores = trainer.evalFunc(config=trainer.config,model=trainer.model, canopies=canopies, threshDict={infMethod:currThreshold},
										  inferenceMethods=[infMethod], metricsForEval=f1ToEval)
			for metric in allScores:
				allScores[metric][currThreshold] = scores[metric][0] if metric in scores else 0
			if currThreshold not in allScores[f1Metric]:
				pass
		
		midThreshold = (lowerThreshold + upperThreshold) / 2
		if printLog: trainer.logger.info("Mid Threshold:{:.3f}".format(midThreshold))
		if midThreshold not in allScores[f1Metric]:
			scores = trainer.evalFunc(config=trainer.config, model=trainer.model, canopies=canopies, threshDict={infMethod:midThreshold},
										  inferenceMethods=[infMethod], metricsForEval=f1ToEval)
			for metric in allScores:
				allScores[metric][midThreshold] = scores[metric][0] if metric in scores else 0
		
		
		if (round(allScores[f1Metric][upperThreshold], 3) == round(allScores[f1Metric][lowerThreshold], 3)) \
				and (round(allScores[f1Metric][midThreshold], 3) == round(allScores[f1Metric][lowerThreshold], 3)):
			trainer.logger.info("Stopping as F1 at upperThreshold, lowerThreshold and midThreshold is same upto 3 decimal places")
			break
	
	# Choose bestThreshold from all the threshold values tried so far
	thresholdVals = sorted(list(allScores[f1Metric].keys()))
	bestThreshold = thresholdVals[0]
	for threshTried in allScores[f1Metric]:  # Choose threshold that gave best F1 on dev set
		if allScores[f1Metric][threshTried] >= allScores[f1Metric][bestThreshold]:
			bestThreshold = threshTried
	
	end = time.time()
	threshTried = sorted(list(allScores[f1Metric].keys()))
	if printLog: trainer.logger.info("Tried {} threshold values. Threshold tried:{}".format(len(allScores[f1Metric]), ",".join(["{:.3f}\t{:.6f}\n".format(x, allScores[f1Metric][x]) for x in threshTried])))
	trainer.logger.info("Time taken for choosing threshold={:.3f} with {} = {:.4f} is {:.3f}".format(bestThreshold, f1Metric, allScores[f1Metric][bestThreshold], end - start))
	trainer.logger.info("==" * 20 + "\n")
	
	Path(trainer.config.resultDir + "/chooseThresh").mkdir(parents=True, exist_ok=True)  # Create resultDir directory if not already present
	
	for metric in [f1Metric]:
		plt.clf()
		X = sorted(allScores[metric].keys())
		Y = [allScores[metric][x] for x in X]
		plt.plot(X, Y, 'ro-')
		plt.plot([bestThreshold], [allScores[metric][bestThreshold]], 'b*')
		plt.xlabel("Threshold")
		plt.ylabel("{} {}".format(infMethod, metric))
		plt.grid()
		plt.title("{} vs Threshold".format(metric))
		plt.savefig(trainer.config.resultDir + "/chooseThresh/{}_{}_{}.png".format(infMethod, metric, epoch))
		plt.close()
	
	return bestThreshold
