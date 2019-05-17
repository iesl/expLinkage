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

import csv, itertools
from pathlib import Path

# Read feature vectors from dataDir and write them to outDir after processing, one canopy at a time
def processADANA(dataDir, outDir):
	authorList = [str(f) for f in Path(dataDir).glob("*.xml") if f.is_file()]
	authorList = [authorFile[:-4] for authorFile in authorList]
	print("Author list:{}".format(authorList))
	for authorFile in authorList:
		
		authorName = authorFile.split("/")[-1]
		if authorFile.endswith("Wei Wang"):
			print("Skipping {} because it does not have {}_ans.txt".format(authorFile,authorFile))
			continue
			
		pairFeatures = {}
		pidToCluster = {}
		
		with open("{}_ans.txt".format(authorFile),"r") as f:
			for line in f:
				line  =line.split()
				paperId, clusterId = int(line[0]),int(line[1])
				pidToCluster[paperId] = clusterId
		
		# Initialize pairFeatures to empty list
		pidList = sorted(pidToCluster)
		for p1,p2 in itertools.combinations(pidList,2):
			pairFeatures[(p1,p2)] = []
			
		with open("{}.txt".format(authorFile),"r") as f:
			numPapers = int(f.readline().strip())
			for featNum in range(8):
				for i in range(numPapers-1):
					line = f.readline()
					line = [float(x) for x in line.strip().split()]
					for j,val in enumerate(line):
						pairFeatures[(i,i+j+1)].append(val)
				
				line = f.readline() # Read empty line between two feature matrices
		
		print("Writing down data for author:{}".format(authorFile))
		Path("{}/{}".format(outDir, authorName)).mkdir(parents=True, exist_ok=True)
		with open("{}/{}/gtClusters.tsv".format(outDir, authorName), "w") as f:
			for pid in pidToCluster:
				f.write("{}\t{}\n".format(pid, pidToCluster[pid]))
		
		with open("{}/{}/pairFeatures.csv".format(outDir, authorName), "w") as f:
			writer = csv.writer(f)
			for p1, p2 in pairFeatures:
				line = [p1, p2] + pairFeatures[(p1, p2)]
				if pidToCluster[p1] == pidToCluster[p2]:
					line.append(1)
				else:
					line.append(0)
				
				writer.writerow(line)


if __name__ == "__main__":
	
	dataDir = "../data/rich-author-disambiguation-data/experimental-results"
	outDir = "../data/authorCoref"
	processADANA(dataDir=dataDir, outDir=outDir)