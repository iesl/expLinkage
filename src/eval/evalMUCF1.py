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

import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def calc_muc_score(pidToCluster_true, pidToCluster_pred):
	# For each predicted cluster, accumulate points in that cluster
	predClusterToPids = {}
	for idx, predCluster in enumerate(pidToCluster_pred):
		try:
			predClusterToPids[predCluster].append(idx)
		except:
			predClusterToPids[predCluster] = [idx]
	
	precNumerator, precDenominator = 0, 0
	for predCid in predClusterToPids:  # Compute precision for each predicted cluster. Find connected component in each predicted cluster
		pidList = predClusterToPids[predCid]
		if len(pidList) <= 1:
			continue
		
		data, rows, cols = [], [], []
		for p1, p2 in itertools.combinations(pidList, 2):
			if pidToCluster_true[p1] == pidToCluster_true[p2]:
				data += [1]
				rows += [p1]
				cols += [p2]
				
				data += [1]
				rows += [p2]
				cols += [p1]
		idMapping = {p: idx for idx, p in enumerate(pidList)}
		rows = [idMapping[p] for p in rows]
		cols = [idMapping[p] for p in cols]
		numPointInCluster = len(pidList)
		predClusterSparseMatrix = csr_matrix((data, (rows, cols)), shape=(numPointInCluster, numPointInCluster))
		
		numConnComp = connected_components(predClusterSparseMatrix)[0]
		precNumerator += numPointInCluster - numConnComp
		precDenominator += numPointInCluster - 1
		# print("Points in predCluster:{}\t{}\n{}/{}".format(predCid, pidList, numPointInCluster - numConnComp, numPointInCluster-1))
	
	precision = precNumerator / precDenominator if precDenominator > 0 else 1
	
	trueClusterToPids = {}
	for idx, trueCluster in enumerate(pidToCluster_true):
		try:
			trueClusterToPids[trueCluster].append(idx)
		except:
			trueClusterToPids[trueCluster] = [idx]
	
	recallNumerator, recallDenominator = 0, 0
	for trueCid in trueClusterToPids:
		pidList = trueClusterToPids[trueCid]
		if len(pidList) <= 1:
			continue
		
		data, rows, cols = [], [], []
		for p1, p2 in itertools.combinations(pidList, 2):
			if pidToCluster_pred[p1] == pidToCluster_pred[p2]:
				data += [1]
				rows += [p1]
				cols += [p2]
				
				data += [1]
				rows += [p2]
				cols += [p1]
		
		idMapping = {p: idx for idx, p in enumerate(pidList)}
		rows = [idMapping[p] for p in rows]
		cols = [idMapping[p] for p in cols]
		numPointInCluster = len(pidList)
		
		trueClusterSparseMatrix = csr_matrix((data, (rows, cols)), shape=(numPointInCluster, numPointInCluster))
		numConnComp = connected_components(trueClusterSparseMatrix)[0]
		recallNumerator += numPointInCluster - numConnComp
		recallDenominator += numPointInCluster - 1
		# print("Points in trueCluster:{}\t{}\n{}/{}".format(trueCid, pidList, numPointInCluster - numConnComp, numPointInCluster - 1))
		
	recall = recallNumerator / recallDenominator if recallDenominator > 0 else 1
	f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
	
	return {"muc_precision": precision, "muc_recall": recall, "muc_f1": f1,
			"muc_precision_num": precNumerator, "muc_precision_den": precDenominator,
			"muc_recall_num": recallNumerator, "muc_recall_den": recallDenominator}
