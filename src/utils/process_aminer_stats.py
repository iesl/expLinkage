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

import math
import numpy as np
from utils.basic_utils import read_canopy_data
import json
from collections import defaultdict
import itertools,csv

def run(dataDir):

	canopyData = read_canopy_data(dataDir)
	
	all_measures= defaultdict(dict)
	for canopyId in canopyData:
		canopy = canopyData[canopyId]
		numEnts  = len(canopy["clusterToPids"])
		numMents = len(canopy["pidToCluster"])
		avgMents = np.mean( [len(canopy["clusterToPids"][c]) for c in canopy["clusterToPids"]] )
		stdMents = np.std( [len(canopy["clusterToPids"][c]) for c in canopy["clusterToPids"]] )
		numSingletons = sum([1 for c in canopy["clusterToPids"] if len(canopy["clusterToPids"][c]) == 1 ])
		all_measures["numEnts"][canopyId] 		= numEnts
		all_measures["numMents"][canopyId] 		= numMents
		all_measures["avgMents"][canopyId] 		= avgMents
		all_measures["stdMents"][canopyId] 		= stdMents
		all_measures["numSingletons"][canopyId] = numSingletons
	
	for measure in all_measures:
		json.dump(all_measures[measure], open("resources/aminer/aminer_{}.json".format(measure),"w"))
	
	all_measures["origin"] = json.load(open("resources/aminer/aminer_origin.json","r"))
	
	corrCoeff = {}
	
	for m1,m2 in itertools.combinations_with_replacement(all_measures,2):
		canopies = list(all_measures[m1].keys())
		X_1 = [all_measures[m1][c] for c in canopies]
		X_2 = [all_measures[m2][c] for c in canopies]
		corrCoeff[m1,m2] = np.corrcoef(X_1, X_2)[0, 1]
		corrCoeff[m2,m1] = np.corrcoef(X_1, X_2)[1, 0]
	
	mlist = list(all_measures.keys())
	with open("resources/aminer/aminer_correlation.csv","w") as f:
		f = csv.DictWriter( f,["Method"]+ mlist )
		f.writeheader()
		for m1 in mlist:
			row = {"Method":m1}
			for m2 in mlist:
				row[m2] = "{:.3f}".format(corrCoeff[m1,m2])
		
			f.writerow(row)
		
		
		
		


if __name__ == "__main__":
	run("../data/authorCoref")