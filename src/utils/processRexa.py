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

import csv
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Read feature vectors from dataDir and write them to outDir after processing, one canopy at a time
def processRexa(dataDir, outDir):
	
	folderList = [str(f) for f in Path(dataDir).glob("*") if f.is_dir()]
	
	for ctr, folder in enumerate(folderList):
		canopyId = folder.split("/")[-1]
		pairFeatures = {}
		mentToId = {}
		pidToCluster = {}
		rows,cols,data = [],[],[]
		with open("{}/pair_vecs.tsv".format(folder),"r") as f:
			reader = csv.reader(f,delimiter="\t")
			for line in reader:
				m1, m2 = line[0], line[1]
				featureVec = line[3:-1]
				
				pairFeatures[(m1, m2)] = featureVec
				mentToId[m1] = 1
				mentToId[m2] = 1
				if line[2] == "1":
					# Accumulate data to create sparse matrix and then run connected components to retrieve gt clusters
					rows += [m1]
					cols += [m2]
					data += [1]

					rows += [m2]
					cols += [m1]
					data += [1]
				elif line[2] == "0":
					pass
				else:
					print(line[2])
					raise Exception("Invalid end token")
			
			mentToId = {ment:ctr for ctr,ment in enumerate(mentToId)} # Assign unique id to each point
			
			# Find out ground-truth cluster after running connected components
			rows = [mentToId[ment] for ment in rows]
			cols = [mentToId[ment] for ment in cols]
			numPoints = len(mentToId)
			sparseMatrix = csr_matrix((data, (rows, cols)), shape=(numPoints, numPoints))
			connComp = connected_components(sparseMatrix)
			
			for pid in range(numPoints):
				pidToCluster[pid] = connComp[1][pid]
				
			Path("{}/{}".format(outDir, canopyId)).mkdir(parents=True, exist_ok=True)
			with open("{}/{}/gtClusters.tsv".format(outDir, canopyId), "w") as f:
				for pid in pidToCluster:
					f.write("{}\t{}\n".format(pid, pidToCluster[pid]))
					
			with open("{}/{}/pairFeatures.csv".format(outDir, canopyId), "w") as f:
				writer = csv.writer(f)
				for m1,m2 in pairFeatures:
					line = [ mentToId[m1], mentToId[m2] ] + pairFeatures[(m1,m2)]
					
					if pidToCluster[mentToId[m1]] == pidToCluster[mentToId[m2]]:
						line.append(1)
					else:
						line.append(0)
						
					writer.writerow(line)
					
if __name__ == "__main__":

	# dataDir = "../data/data/rexa/canopy"
	dataDir = "../data/data_rexa_all/nick-rexa/rexa/canopy"
	outDir  = "../data/rexa_new"
	processRexa(dataDir=dataDir, outDir=outDir)