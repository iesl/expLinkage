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

import  csv,os,argparse,copy
import numpy as np
from pathlib import Path

from utils.plotting import plotMetricsFromCSV
from utils.basic_utils import get_filename_list
from utils.Config import Config

def combineResults(parameters, xlabel, currResultDir, template):
	"""
	Put together results from all files into 1 file
	:param parameters: Dictionary with key as parameter names and value as a list of parameter values which need to be combined
	:param xlabel: Varying dimension
	:param currResultDir:
	:param template:  template for folder name where results are read from for combining
	:return:
	"""
	
	filenameList = get_filename_list(parameters, template)
	data = {}
	header = None
	numFiles  = 0
	# Read data from all files
	for filenum,filename in enumerate(filenameList):
		data[filenum] = {}
		
		fileCheck = Path(filename)
		if not fileCheck.is_file():
			print("pwd:{}".format(os.getcwd()))
			print("File does not exist:{}".format(filename))
			continue
		
		numFiles +=1
		with open(filename, "r") as f: # Read data from this file into a dictionary
			csvReader = csv.DictReader(f)
			for row in csvReader:
				if header is None: header = list(row.keys())  # Get header names
				
				for col in header:  # Convert all row values to float if they can be else assign 0 value(these columns must be empty)
					try:
						row[col] = float(row[col])
					except ValueError:  # ValueError because these col must be empty
						assert row[col] == ""
						row[col] = 0
				
				# xlabelValue = None
				# for col in header:  # Find value of xDim for this row
				# 	if col == xlabel:
				# 		xlabelValue = row[col]
				# 		break
				xlabelValue = row[xlabel] if xlabel in row else None
				
				assert xlabelValue is not None
				data[filenum][xlabelValue] = {}
				for col in header:  # Add data for all col in data dictionary as list of values
					if col == xlabel: continue
					data[filenum][xlabelValue][col] = row[col]
		
		assert len(data[filenum]) == 1
					
	# Compute best result for each file
	finalData = {}
	for filenum in data:
		if len(data[filenum].keys()) == 0:
			print("Ignoring file:{}\n".format(filenum))
			bestxDimValue = None
			continue
		else:
			assert len(data[filenum]) == 1
			bestxDimValue = list(data[filenum].keys())[0]
			
		bestRow = {}
		for col in data[filenum][bestxDimValue]:
			bestRow[col] = data[filenum][bestxDimValue][col]
		finalData[filenum] = (bestxDimValue,copy.deepcopy(bestRow))
		
	# Write csv file containing best results from all files
	with open(currResultDir + "/results.csv", "w") as f:
		csvWriter = csv.DictWriter(f, fieldnames=header+["FileNum"])
		csvWriter.writeheader()
		
		for filenum in range(numFiles):
			if filenum in finalData:
				tempDict = copy.deepcopy(finalData[filenum][1])
				tempDict[xlabel] = finalData[filenum][0]  # Add xDim to dictionary when writing data
				tempDict["FileNum"] = filenum
				
				csvWriter.writerow(tempDict)
			else:
				pass
				# print("Filenum not included in best result, possibly because choosing a threshold failed for this file:{}".format(filenum))
	
	print("\nIgnoring orginal standard deviations when computing avg of best results")
	print("File will have standard deviation of best mean scores\n")
	# Write csv file containing avg of best results from all files
	with open(currResultDir + "/avgOfBestResults.csv", "w") as f:
		csvWriter = csv.DictWriter(f, fieldnames=header)
		csvWriter.writeheader()

		avgData = {col:[] for col in header}
		
		for col in header:
			if col.endswith("_std"): # If commenting this, also comment computation of std deviation  of best scores
				continue
				# print("Ignoring deviation of best scores, just reporting average of best scores, and average of their std deviations")
				
			for filenum in range(numFiles):
				if filenum in finalData:
					if col == xlabel:
						avgData[col] += [finalData[filenum][0]]
					else:
						avgData[col] += [finalData[filenum][1][col]]
				else:
					pass
					# print("Filenum not included in best result, possibly because choosing a threshold failed for this file:{}".format(filenum))
				
			if col.endswith("_mean"): # Computing std deviation of best scores
				avgData[col[:-5]+"_std"] = np.std(avgData[col])
			avgData[col] = np.mean(avgData[col])
		
		for col in avgData:
			if isinstance(avgData[col], float):
				avgData[col] = "{:0.4f}".format(avgData[col])
				
		csvWriter.writerow(avgData)

def run_combineResults(baseResDir, outDirPrefix, xlabel, baseTemplate, relResultDir, parameters):
	"""

	:param baseResDir:
	:param outDirPrefix: Prefix to be used for directory where results will be stored
	:param xlabel: Dimension along which best rows have to be found. eg=Threshold, Epoch
	:param baseTemplate: Template structure for folder where results are stored
	:param relResultDir: Folder where result.csv file is present,(relative to result directory where training results are stored)
	:param parameters: Dictionary with key as parameter names and value as a list of parameter values which need to be combined
	:return:
	"""
	origWorkDir = os.getcwd()
	os.chdir(baseResDir)
	
	currResultDir = "{outDirPrefix}_xlabel={xlabel}/{base}".format(outDirPrefix=outDirPrefix, xlabel=xlabel, base=baseTemplate)
	currResultDir = currResultDir.format(**parameters)
	Path(currResultDir).mkdir(parents=True, exist_ok=True)  # Create resultDir directory if not already present
	print("CurrResultDir:{}".format(currResultDir))
	
	############### Combine Results ############################
	template = baseTemplate + "/{}/results.csv".format(relResultDir)
	combineResults(parameters, xlabel, currResultDir, template)
	
	################ Plot Results ##############################
	# os.chdir(origWorkDir)
	# currResultDir = baseResDir + "/" + currResultDir
	# plotMetricsFromCSV(currResultDir=currResultDir, xlabel="FileNum")


if __name__ == "__main__":
	

	parser = argparse.ArgumentParser(description='Combine results from different runs Ex: python -m scripts.combineResults --outDirPrefix=BestF1_AvgW --baseResDir=../results/c=NP_Coref --relResultDir=varyThresAvgWeights_f1 --xlabel=Threshold --trainObj=allWithin_allAcross  --threshold=0.0 --margin=5 --modelType=avgLinear --trainFrac=0.6 --testFrac=0.3 --devFrac=0.1 --seed 1 2 3 4 5 6 7 8 9 10')
	
	# ################################## OPTIONAL ARGUMENTS TO OVERWRITE CONFIG FILE ARGS###################################################
	# temp_config = Config()
	# for config_arg in temp_config.__dict__:
	# 	if config_arg == "seed": continue
	# 	def_val = temp_config.__getattribute__(config_arg)
	# 	arg_type = type(def_val) if def_val is not None else str
	# 	parser.add_argument('--{}'.format(config_arg), type=arg_type, default=None, help='If not specified then value from config file will be used')
	# #########################################################################################################

	parser.add_argument('--config', type=str,required=True, help='Config file')
	parser.add_argument('--seed', nargs='+',required=True, type=int, help="seed for random number generator")
	parser.add_argument('--xlabel', type=str,required=True, help='X-Label')
	parser.add_argument('--baseResDir', type=str, required=True,help='Directory where all result folders are stored')
	parser.add_argument('--suffix', type=str, default="", help="Suffix at end of each directory")
	parser.add_argument('--relResultDir', type=str,required=True, help='Name of folder where results.csv file is present(relative to folder where training results are stored')
	parser.add_argument('--outDirPrefix', type=str,required=True, help='Prefix to be used for directory where results will be stored')
	
	args = parser.parse_args()
	config = Config(args.config)
	
	parameters = {}
	parameters["d"] 	= config.dataDir.split("/")[-1]
	parameters["obj"] 	= config.trainObj
	parameters["s"] 	= args.seed
	xlabel 				= args.xlabel

	if args.suffix != "":
		parameters["suff"] 	= [args.suffix]
		baseTemplate = "obj={obj}_s={s}{suff}"
	else:
		baseTemplate = "obj={obj}_s={s}"
	
	run_combineResults(baseResDir=args.baseResDir, outDirPrefix=args.outDirPrefix, xlabel=args.xlabel,
					   baseTemplate=baseTemplate, relResultDir=args.relResultDir, parameters=parameters)
