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

import numpy as np
import time
from hier_clust.expLink import getPidToPredClusters, computeDendPurity

		
def run_random_split(pidToCluster, k=None):

	numPoints  = len(pidToCluster)
	activeClusters = [pid for pid in range(numPoints)]
	newCid          = numPoints

	pidToParent = {}
	children    = {pid:None for pid in activeClusters}

	y_pred = None
	while len(activeClusters) > 1:

		# Find clusters to merge
		
		cs = np.random.choice(activeClusters, 2, replace=False) # Random clusters to merge
		c1 = cs[0]
		c2 = cs[1]
		
		# Remove merged clusters for list
		activeClusters.remove(c1)
		activeClusters.remove(c2)
		
		# Update distances of the merged cluster with all remaining clusters
		activeClusters.append(newCid)
		
		children[newCid] 	= (c1,c2)
		pidToParent[c1] 	= newCid
		pidToParent[c2] 	= newCid

		if k is not None and len(activeClusters) == k: # Get flat clusters such that there are k clusters
			pidToPredCluster_k = getPidToPredClusters(numPoints=numPoints, pidToParent=pidToParent)
			y_pred = [pidToPredCluster_k[ pid ] for pid in range(numPoints)]

		newCid += 1
	
	if y_pred is None: # This is triggered when while loop terminated without forming flat clusters. it means that all points are put in 1 cluster
		y_pred = [1 for x in range(numPoints)]
	
	if pidToCluster is None:
		dendPurity = 0
	else:
		dendPurity = computeDendPurity(pidToCluster=pidToCluster, children=children, pidToParent=pidToParent)
	
	
	return y_pred, dendPurity

