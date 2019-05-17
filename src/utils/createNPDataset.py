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

import os
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from utils.fixNPCorefDataFormat import fixDataFormat

def createDataset(dataDir,outDir):
	
	for dataType in ["train","test"]:
		processPairwiseData(dataDir, dataType,outDir)
	
# dataType= "train" or "test"
def processPairwiseData(dataDir,dataType,outDir):
	with open(dataDir+"/"+dataType,"r") as f:
		fileList = f.read().split()
	
	Path(outDir).mkdir(parents=True, exist_ok=True)
	
	
	for file in fileList:
		Path("{}/{}".format(outDir, file)).mkdir(parents=True, exist_ok=True)
		featureFile = open("{}/{}/pairFeatures.csv".format(outDir, file), "w")
		rows, cols, data = [], [], []
		uniquePts = {}
		with open("{}/{}/features.development/features.arff".format(dataDir, file), "r") as f:
			for line in f:
				if line.startswith("@"): continue
				if len(line.split(",")) < 2: continue
				
				featureFile.write(line)
				lineV = line.strip().split(",")
				docNum, id1, id2 = int(lineV[0]), int(lineV[1]), int(lineV[2])

				uniquePts[id1] = 1
				uniquePts[id2] = 1
				if lineV[-1] == "+":
					# Accumulate data to create sparse matrix and then run connected components to retrieve gt clusters
					rows += [id1]
					cols += [id2]
					data += [1]
					
					rows += [id2]
					cols += [id1]
					data += [1]
				elif lineV[-1] == "-":
					pass
				else:
					print(lineV)
					raise Exception("Invalid end token")
		featureFile.close()
		
		numPoints = len(uniquePts)
		sparseMatrix = csr_matrix((data, (rows, cols)), shape=(numPoints, numPoints))
		connComp = connected_components(sparseMatrix)
		if file == "2":
			print(file, numPoints, connComp)
	
		with open("{}/{}/gtClusters.tsv".format(outDir,file),"w") as f:
			for id in range(numPoints):
				f.write("{}\t{}\n".format(id, connComp[1][id]))
		

if __name__ == '__main__':
	tempOutDir = "../data/NP_Coref_temp"
	createDataset(dataDir="../data/reconcile/uw-corpus",outDir=tempOutDir)


	newDir = "../data/NP_Coref"

	# Remove docNum from this temporary dataset
	fixDataFormat(origDir=tempOutDir, newDir=newDir)
	
	os.system("rm -r {}".format(tempOutDir))