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


import  csv,os,argparse,copy,itertools,scipy
import numpy as np
from pathlib import Path
from utils.plotting import plotMetricsFromCSV

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc
plt.rcParams.update({'font.size': 8,'font.family': 'Serif'})
rc('font',**{'family':'serif','serif':['Times']})

def readAvgFile(filename):
	data= {}
	with open(filename,"r") as f:
		csvReader = csv.DictReader(f)
		ctr = 0
		for row in csvReader:
			data = row
			ctr += 1
		if ctr > 1:
			print("There is more than one row in this file:{}\n In case of more than one row,I am just using the last row".format(filename))

	return data

def readResultFile(filename):
	data = {}
	with open(filename, "r") as f:
		csvReader = csv.DictReader(f)
		ctr = 0
		for row in csvReader:
			data[ctr] = {}
			for k in row:
				try:
					data[ctr][k] = float(row[k])
				except :
					print("Error converting value to float:{}".format(row[k]))
					data[ctr][k] = row[k]
			ctr += 1

	return data

def createTable(baseResDir, outDirPrefix, xlabel, baseTemplate, parameters, varyParam ):
	"""
	# This is put together data from all avgResults files for different variation of methods, eg for different objectives
	
	:param baseResDir:
	:param outDirPrefix:
	:param xlabel:
	:param baseTemplate:
	:param parameters:
	:param varyParam: Parameter that will vary across different columns of combined results table, eg training objective
	:return:
	"""

	os.chdir(baseResDir)
	baseTemplate = "{outDirPrefix}_xlabel={xlabel}/" +baseTemplate

	tempParam 		= copy.deepcopy(parameters)
	tempParam[varyParam] = "{" + varyParam + "}"

	currResultDir 	= baseTemplate.format(outDirPrefix=outDirPrefix,xlabel=xlabel,**tempParam)
	Path(currResultDir).mkdir(parents=True, exist_ok=True)  # Create resultDir directory if not already present
	Path("{}/combTables".format(currResultDir)).mkdir(parents=True, exist_ok=True)  # Create resultDir directory if not already present
	print("\nCurrResultDir:{}\n".format(currResultDir))
	
	
	baseTemplate = baseTemplate + "/avgOfBestResults.csv"
	filenameTemplate = baseTemplate.format(outDirPrefix=outDirPrefix,xlabel=xlabel,**tempParam)

	data = {}
	for paramVal in parameters[varyParam]: # Read csv file for each paramVal
		currFile = filenameTemplate.format(**{varyParam:paramVal})
		data[paramVal] = readAvgFile(currFile)

	for dataType in ["train","test","dev"]:
		for metType  in ["f1","randIndex","dendPurity","nmi"]:
			if "face" in baseResDir:
				metrics = ["Method"]
				for infMethod in [ "singleLink", "singleLink@t", "avgLink", "avgLink@t", "compLink", "compLink@t", "linkage_auto", "linkage_auto@t"]:
					metrics += ["{}_{}_{}_mean".format(dataType, infMethod, metType)]
					metrics += ["{}_{}_{}_std".format(dataType, infMethod, metType)]
	
			elif "NP_Coref" in baseResDir or "rexa" in baseResDir or  "authorCoref" in baseResDir:
				metrics = ["Method"]
				for infMethod in ["connComp", "singleLink", "singleLink@t", "avgLink", "avgLink@t", "compLink", "compLink@t", "linkage_auto", "linkage_auto@t"]:
					metrics += ["{}_{}_{}_mean".format(dataType, infMethod, metType)]
					metrics += ["{}_{}_{}_std".format(dataType, infMethod, metType)]
			else:
				metrics = ["Method"]
				for paramVal in data.keys():
					metrics += list(data[paramVal].keys())
					break
	
			with open("{}/combTables/{}_{}.csv".format(currResultDir,metType, dataType),"w") as f:
				csvWriter = csv.DictWriter(f, fieldnames=metrics)
				csvWriter.writeheader()
	
				for paramVal in data.keys():
					# tempDict = copy.deepcopy(data[paramVal])
					tempMetric = [m for m in  metrics if m!= "Method" and m in data[paramVal]]
					tempDict= {"Method": paramVal}
					for m in tempMetric:
						if isinstance(data[paramVal][m], float):
							tempDict[m] = "{:.4f}".format(data[paramVal][m])
						else:
							tempDict[m] = "{}".format(data[paramVal][m])
	
					csvWriter.writerow(tempDict)

def createWinLossMatrix_matching(baseResDir, outDirPrefix, xlabel, baseTemplate, parameters, varyParam ):
 
	os.chdir(baseResDir)
	baseTemplate = "{outDirPrefix}_xlabel={xlabel}/" + baseTemplate

	tempParam = copy.deepcopy(parameters)
	tempParam[varyParam] = "{" + varyParam + "}"

	currResultDir = baseTemplate.format(outDirPrefix=outDirPrefix, xlabel=xlabel, **tempParam)
	Path(currResultDir).mkdir(parents=True, exist_ok=True)  # Create resultDir directory if not already present
	print("\nCurrResultDir:{}\n".format(currResultDir))

	newBaseTemplate = baseTemplate + "/results.csv"
	filenameTemplate = newBaseTemplate.format(outDirPrefix=outDirPrefix, xlabel=xlabel, **tempParam)
	data = {}
	for paramVal in parameters[varyParam]:  # Read csv file for each paramVal
		currFile = filenameTemplate.format(**{varyParam: paramVal})
		data[paramVal] = readResultFile(currFile)

	newBaseTemplate = baseTemplate + "/avgOfBestResults.csv"
	filenameTemplate = newBaseTemplate.format(outDirPrefix=outDirPrefix, xlabel=xlabel, **tempParam)
	avgData = {}
	for paramVal in parameters[varyParam]:  # Read csv file for each paramVal
		currFile = filenameTemplate.format(**{varyParam: paramVal})
		avgData[paramVal] = readAvgFile(currFile)
	
	resType = "test"
	for evalMetric in ["f1", "dendPurity","dendPurityTopDown"]:
	# for evalMetric in ["dendPurityTopDown"]:
		bestMethodForMetric = {}
		metrics = []
		if evalMetric == "f1":
			for infMethod in [ "singleLink@t", "avgLink@t", "compLink@t", "linkage_auto@t"]:
				metric = "{}_{}_{}_mean".format(resType, infMethod, evalMetric)
				metrics += [metric]
					
		elif evalMetric == "dendPurity":
			for infMethod in [ "singleLink", "avgLink", "compLink", "linkage_auto"]:
				metric = "{}_{}_{}_mean".format(resType, infMethod, evalMetric)
				metrics += [metric]
				
		elif evalMetric == "dendPurityTopDown":
			for infMethod in [ "recSparsest","random"]:
				metric = "{}_{}_{}_mean".format(resType, infMethod, "dendPurity")
				metrics += [metric]
			
		assert varyParam == "obj"
		trainMethods = parameters[varyParam]
		
		methodToIdx  = {method:ctr for ctr,method in enumerate(trainMethods)}
		metricToIdx  = {metric:ctr for ctr,metric in enumerate(metrics)}
		
		winLoss = {} # Relative to bestMethod for each metric
		winLossMatrix = np.zeros((len(trainMethods),len(metrics)))
		
		for metric in metrics:
			
			# Find how many times one method wins/loses over other
			bestScore = 0
			bestMethod = 0
			for method in trainMethods:
				try:
					if float(avgData[method][metric]) > bestScore:
						bestMethod = method
						bestScore =  float(avgData[method][metric])
				except:
					print(avgData[method][metric])
					exit(0)
					
			bestMethodForMetric[metric] = bestMethod
			
			winLoss[metric] = {}
			for method in trainMethods:
				
				winLoss[metric][(bestMethod, method)] = []
				numFiles = len(data[method])
				
				assert len(data[bestMethod]) == len(data[method])
				
				for ctr in range(numFiles):
					if metric in data[bestMethod][ctr] and metric in data[method][ctr]:
						winLoss[metric][(bestMethod, method)] += [data[method][ctr][metric] - data[bestMethod][ctr][metric]]
					else:
						pass
				
				avgPerf 	= np.mean( winLoss[metric][(bestMethod, method)] )
				stdDevPerf 	= np.std( winLoss[metric][(bestMethod, method)] )
				
				numWins = sum([1 for x in winLoss[metric][(bestMethod, method)] if x>0 ])
				# winAvg  = np.mean([x for x in winLoss[(bestMethod, method)] if x>0 ]) if numWins > 0 else 0
				# winStd  = np.std([x for x in winLoss[(bestMethod, method)] if x>0 ])  if numWins > 0 else 0

				avgDiff = np.mean([x for x in winLoss[metric][(bestMethod, method)]])

				error  		= np.sum([(x-avgDiff)**2 for x in winLoss[metric][(bestMethod, method)]])
				error  		= np.sqrt(error/(numFiles-1))
				t_statistic = np.abs(avgDiff*np.sqrt(numFiles)/error)
				p_value 	= 2*(1 - scipy.stats.t.cdf(t_statistic, numFiles-1))
				
				winLoss[metric][(bestMethod,method)] = (100*avgPerf, 100*stdDevPerf,p_value)
				i = methodToIdx[method]
				j = metricToIdx[metric]
				winLossMatrix[i][j] = 100*avgPerf

					
		# Write Data for all metric in 1 file instead of writing them in separate files
		
		f1 = open("{}/relativeScores_{}.csv".format(currResultDir,evalMetric), "w")
		f2 = open("{}/absoluteScores_{}.csv".format(currResultDir,evalMetric), "w")
		
		header = ["TrainMethod"]
		for ctr,metric in enumerate(metrics):
			header += ["amp{}".format(ctr), metric]
		
		csvWriter1 = csv.DictWriter(f1, fieldnames=header)
		csvWriter1.writeheader()
		
		csvWriter2 = csv.DictWriter(f2, fieldnames=header)
		csvWriter2.writeheader()
		
		for method in trainMethods:
			tempDict1 = {"TrainMethod": method}
			tempDict2 = {"TrainMethod": method}
			for ctr,metric in enumerate(metrics):
				tempDict1["amp{}".format(ctr)] = "&"
				tempDict2["amp{}".format(ctr)] = "&"
				bestMethod = bestMethodForMetric[metric]
				try:
					avgDiff = winLoss[metric][(bestMethod, method)][0]
					p_value = winLoss[metric][(bestMethod, method)][2]
					absoluteVal = 100*float(avgData[method][metric])
					
					if p_value < 0.01:
						# tempDict[metric]  = "{:.1f}".format(avg) + "$^{**}$"
						temp1 = r"\underline{\underline{" +  "{:.1f}".format(avgDiff) + "}}"
						tempDict1[metric]  =  "{:40}".format(temp1)
						
						
						temp2 = r"\underline{\underline{" +  "{:.1f}".format(absoluteVal) + "}}"
						tempDict2[metric]  =  "{:40}".format(temp2)
					elif p_value < 0.05:
						# tempDict[metric]  = "{:.1f}".format(avg) + "$^{*}$"
						temp1 = r"\underline{" +  "{:.1f}".format(avgDiff) + "}"
						tempDict1[metric]  =  "{:40}".format(temp1)
						
						temp2 = r"\underline{" +  "{:.1f}".format(absoluteVal) + "}"
						tempDict2[metric]  =  "{:40}".format(temp2)
					else:
						
						if method == bestMethod:
							temp1 = r"\textbf{" + "{:.1f}".format( 100*float(avgData[method][metric])) + "}"
							tempDict1[metric]  =  "{:40}".format(temp1)
							
							temp2 =   r"\textbf{" + "{:.1f}".format(absoluteVal) + "}"
							tempDict2[metric]  =  "{:40}".format(temp2)
						else:
							temp1 = "{:.1f}".format( avgDiff )
							tempDict1[metric]  =  "{:40}".format(temp1)
						
							temp2 =   "{:.1f}".format(absoluteVal)
							tempDict2[metric]  =  "{:40}".format(temp2)
						
				except Exception as e:
					print(winLoss[metric])
					raise e
			
			csvWriter1.writerow(tempDict1)
			csvWriter2.writerow(tempDict2)
		
		f1.close()
		f2.close()
		
		
		
if __name__ == "__main__":
	
	combDiffObj = True
	
	parser = argparse.ArgumentParser(description='Combine results from methods into one file, python -m scripts.compareMethods --baseResDir=../results/c\=NP_Coref --outDirPrefix=BestF1_AvgW --xlabel=Threshold --threshold=0.0 --margin=5 --trainObj allWithin_allAcross allWithin_bestAcross bestWithin_allAcross bestWithin_bestAcross mstWithin_allAcross mstWithin_bestAcross mstAll_False_False mstAll_False_True --trainFrac=0.7 --testFrac=0.3 --modelType=avgLinear --seed 1 2 3 4 5 6 7 8 9 10')

	parser.add_argument('--seed', nargs='+', required=True, type=int, help="seed for random number generator")
	parser.add_argument('--xlabel', type=str, required=True, help='X-Label')
	parser.add_argument('--baseResDir', type=str, required=True, help='Directory where all result folders are stored')
	parser.add_argument('--outDirPrefix', type=str, required=True, help='Prefix to be used for directory where results will be stored')

	if combDiffObj:
		parser.add_argument('--trainObj', type=str, required=True, help='training Objective',nargs="+") # Used when comparing different objectives
		parser.add_argument('--suffix', type=str, default="", help="Suffix at end of each directory")
	else:
		parser.add_argument('--trainObj', type=str, required=True, help='training Objective') # Used when comparing different hyper-parameter/implementaions for same objective
		parser.add_argument('--suffix', type=str, help="Suffix at end of each directory", nargs="+")
		
	args = parser.parse_args()
	parameters = {}
	parameters["s"] = args.seed
	
	if combDiffObj:
		parameters["obj"] = [x for x in args.trainObj] # Used when comparing different objectives
		if args.suffix == "":
			baseTemplate = "obj={obj}_s={s}"
		else:
			parameters["suff"] = [args.suffix]
			baseTemplate = "obj={obj}_s={s}{suff}"
	else:
		parameters["obj"] = [args.trainObj]
		parameters["suff"] = [x for x in args.suffix]
		baseTemplate = "obj={obj}_s={s}{suff}"
		
	xlabel = args.xlabel
	pwd = os.getcwd()
	
	if combDiffObj:
		createTable(baseResDir=args.baseResDir, outDirPrefix=args.outDirPrefix, xlabel=args.xlabel,baseTemplate=baseTemplate,
					parameters=parameters,varyParam="obj")

		os.chdir(pwd)
		createWinLossMatrix_matching(baseResDir=args.baseResDir, outDirPrefix=args.outDirPrefix, xlabel=args.xlabel,baseTemplate=baseTemplate,
							parameters=parameters, varyParam="obj")
	else:
		createTable(baseResDir=args.baseResDir, outDirPrefix=args.outDirPrefix, xlabel=args.xlabel,baseTemplate=baseTemplate,
					parameters=parameters,varyParam="suff")
	