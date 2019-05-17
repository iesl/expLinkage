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
import os
import itertools

# y_true & y_pred is list of labels for each point. Recall, Precision, F1 is for predicted edges on underlying points
def comp_prec_rec_f1_fast(y_true, y_pred):
	assert (len(y_true) == len(y_pred))
	t1 = time.time()
	with open("predicted.tsv", "w") as predicted:
		for id, val in enumerate(y_pred):
			predicted.write(str(id) + "\t" + str(val) + "\n")
	
	with open("goldFile.tsv", "w") as goldFile:
		for id, val in enumerate(y_true):
			goldFile.write(str(id) + "\t" + str(val) + "\n")
	
	filenum = time.time()
	command = "cd $XCLUSTER_ROOT  && source bin/setup.sh &&"
	command += "sh bin/util/score_pairwise.sh ../singleLinkage/predicted.tsv ../singleLinkage/goldFile.tsv algo data None > tempResult_{}".format(filenum)
	print("executing command::\n{}\n".format(command))
	os.system(command)
	precision, recall, f1 = 0, 1, 0
	XCLUSTER_ROOT = os.getenv("XCLUSTER_ROOT")
	
	with open("{}/tempResult_{}".format(XCLUSTER_ROOT,filenum), "r") as results:
		for line in results:
			algo, data, precision, recall, f1 = line.split()
			precision = float(precision)
			recall = float(recall)
			f1 = float(f1)
	
	command = "rm {}/tempResult_{}".format(XCLUSTER_ROOT, filenum)
	print("executing command::\n{}\n".format(command))
	os.system(command)
	t2 = time.time()
	print("Time taken = {:.3f}".format(t2 - t1))
	return {"precision": precision, "recall": recall, "f1": f1}

# y_true & y_pred is list of labels for each point. Recall, Precision, F1 is for predicted edges on underlying points
def comp_prec_rec_f1(y_true, y_pred):  # TODO Optimize this, we do not need to calculate trueNeg and that is a large fraction of all edges
	assert (len(y_true) == len(y_pred))
	truePos = 0
	falseNeg = 0
	
	trueNeg = 0
	falsePos = 0
	numPoints = len(y_true)

	for pid1, pid2 in itertools.combinations(range(numPoints), 2):
		if y_pred[pid1] == y_pred[pid2]:
			if y_true[pid1] == y_true[pid2]:
				truePos += 1  # TP
			else:
				falsePos += 1  # FP
		else:
			if y_true[pid1] == y_true[pid2]:
				falseNeg += 1  # FN
			else:
				trueNeg += 1  # TN
	
	precision 	= truePos / (truePos + falsePos) if (truePos + falsePos) > 0 else 1.
	recall 		= truePos / (truePos + falseNeg) if (truePos + falseNeg) > 0 else 1.
	f1 			= 2 * precision * recall / (precision + recall) if precision + recall > 0. else 0.
	
	return {"precision": precision, "recall": recall, "f1": f1,
			"recall_num":truePos, "recall_den":truePos + falseNeg,
			"precision_num": truePos, "precision_den": truePos + falsePos}
 