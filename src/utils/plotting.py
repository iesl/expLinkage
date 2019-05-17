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
import json
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from models.linearClassifier import LinearClassifier,AvgLinearClassifier

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns


cmap = [('firebrick', 'lightsalmon'), ('green', 'yellowgreen'), ('navy', 'skyblue'), ('darkorange', 'gold'),
		('darkviolet', 'orchid')
	, ('sienna', 'tan'), ('olive', 'y'), ('gray', 'silver'), ('deeppink', 'violet'), ('deepskyblue', 'lightskyblue')]

# Plots clusters with points in R2. pointToCluster is a dictionary with points as keys and their cluster as value
def plot_clusters(pointToCluster, filename='cluster', edgeList= []):

	clusters = {pointToCluster[point]:1 for point in pointToCluster}
	
	plt.clf()
	for edge in edgeList:
		x1,y1,x2,y2 = edge
		plt.plot([x1,x2],[y1,y2],'k-')

	dataToPlot = []
	for cluster in clusters:
		currClusterPoints = [point for point in pointToCluster if pointToCluster[point] == cluster]
		X = [point[0] for point in currClusterPoints]
		Y = [point[1] for point in currClusterPoints]
		
		dataToPlot.append((X, Y))

	dataToPlot.sort(key=lambda pair: len(pair[0]), reverse=True)
	# shapes = "s","^"
	shapes = "."
	for ctr,(X, Y) in enumerate(dataToPlot):
		plt.scatter(X, Y,100, marker=shapes[ctr%len(shapes)] )
	
	plt.axis('off')
	plt.savefig(filename)
	plt.close()

# Takes canopy as input which contains list of pairwise features for each (p1,p2), where p1 \in R2
# Plots these edges and also decision boundary if it is a linear classifier.
def plot_clusters_w_edges(canopy, model, filename, baseModel=None):
	
	matplotlib.rcParams.update({'font.size': 16})
	
	canopyId = list(canopy.keys())[0]
	canopy = canopy[canopyId]
	print("Filename:{}".format(filename))
	
	plt.clf()
	fig, ax = plt.subplots()
	posEdges = []
	negEdges = []
	for p1, p2 in canopy["pairFeatures"]:
		if len(canopy["pairFeatures"][(p1, p2)]) > 2:
			print("Currently, cannot handle more than 2 dimensions\n")
			return
		
		if canopy["pidToCluster"][p1] == canopy["pidToCluster"][p2]:
			posEdges.append(canopy["pairFeatures"][(p1, p2)])
		else:
			negEdges.append(canopy["pairFeatures"][(p1, p2)])
	
	X = [point[0] for point in posEdges]
	Y = [point[1] for point in posEdges]
	plt.scatter(X, Y, 50, "g", "+")
	
	X = [point[0] for point in negEdges]
	Y = [point[1] for point in negEdges]
	plt.scatter(X, Y, 50, "r", "o")
	
	xmin, xmax = matplotlib.pyplot.xlim()
	ymin, ymax = matplotlib.pyplot.ylim()


	if isinstance(model, LinearClassifier):
		x_vals = np.arange(xmin, xmax, (xmax - xmin) / 100)
		
		
		m1 = model.seqModel[0].weight.data[0][0].cpu().data.numpy()
		m2 = model.seqModel[0].weight.data[0][1].cpu().data.numpy()
		b = model.seqModel[0].bias.cpu().data.numpy()[0]
	
		print("m1:{:.4f},m2:{:.4f},b:{:.4f}".format(m1, m2, b))
		X, Y = [], []
		for x in x_vals:
			y = (-1 * b - m1 * x) / m2
			
			Y += [y]
			X += [x]
		
		plt.plot(X, Y, 'b-', label="final1")
	
	if isinstance(model, AvgLinearClassifier):
		
		m1 = model.avgWeights.weight.data[0][0].cpu().data.numpy()
		m2 = model.avgWeights.weight.data[0][1].cpu().data.numpy()
		b = model.avgWeights.bias.cpu().data.numpy()[0]
		
		print("m1:{:.4f},m2{:.4f},b:{:.4f}".format(m1, m2, b))
		
		X, Y = [], []
		for x in x_vals:
			y = (-1 * b - m1 * x) / m2
			Y += [y]
			X += [x]
		
		plt.plot(X, Y, 'k-',label="final2")
	
	# Linear model with parameters given as (m1,m2,b)
	if isinstance(model, tuple):
		x_vals = np.arange(xmin, xmax, (xmax - xmin) / 100)
		
		m1,m2,b = model
		print("m1:{:.4f},m2:{:.4f},b:{:.4f}".format(m1, m2, b))
		X, Y = [], []
		for x in x_vals:
			y = (-1 * b - m1 * x) / m2
			
			Y += [y]
			X += [x]
		
		plt.plot(X, Y, 'b-', label="All Pairs")

	
	if isinstance(model, SVC):
		h = (xmax - xmin)/50
		xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
							 np.arange(ymin, ymax, h))
		
		Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		ax.contourf(xx, yy, Z, alpha=0.4,cmap=plt.cm.RdYlGn)
		# out = ax.contourf(xx, yy, Z, alpha=0.9,colors=["red","black","black","green"])
		pass



		# Linear model with parameters given as (m1,m2,b)
	if isinstance(baseModel, tuple):
		x_vals = np.arange(xmin, xmax, (xmax - xmin) / 100)
		
		m1,m2,b = baseModel
		print("m1:{:.4f},m2:{:.4f},b:{:.4f}".format(m1, m2, b))
		X, Y = [], []
		for x in x_vals:
			y = (-1 * b - m1 * x) / m2
			
			Y += [y]
			X += [x]
		
		plt.plot(X, Y, 'b--', label="All Pairs(OptThresh)")

	modelMST = (-0.028434,-0.024187,0.1186)
	# Linear model with parameters given as (m1,m2,b)
	if isinstance(modelMST, tuple):
		x_vals = np.arange(xmin, xmax, (xmax - xmin) / 100)

		m1, m2, b = modelMST
		print("MST m1:{:.4f},m2:{:.4f},b:{:.4f}".format(m1, m2, b))
		X, Y = [], []
		for x in x_vals:
			y = (-1 * b - m1 * x) / m2

			Y += [y]
			X += [x]

		plt.plot(X, Y, 'k-', label="Optimal")

	# plt.legend()
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	
	plt.axis('off')
	plt.legend()
	plt.savefig(filename)
	plt.close()

def plot_scores_per_canopy(allScores, currResultDir):
	trainScores, devScores, testScores = allScores["train"], allScores["dev"], allScores["test"]
	Path(currResultDir).mkdir(parents=True, exist_ok=True)  # Create resultDir directory if not already present
	
	metricList = list(set(list(trainScores.keys()) + list(testScores.keys()) + list(devScores.keys())))
	for metric in metricList:
		print("Plotting metric={}".format(metric))
		X,Y = [],[]
			
		if metric in trainScores:
			temp = [(x,trainScores[metric][x]) for x in trainScores[metric]]
			temp = sorted(temp,key=lambda x:x[1])
			X_train = [x for x,y in temp]
			X += X_train
			Y += [y for x,y in temp]
			
		if metric in testScores:
			temp = [(x,testScores[metric][x]) for x in testScores[metric]]
			temp = sorted(temp,key=lambda x:x[1])
			X_test = [x for x,y in temp]
			X += X_test
			Y += [y for x,y in temp]
			
		if metric in devScores:
			temp = [(x,devScores[metric][x]) for x in devScores[metric]]
			temp = sorted(temp,key=lambda x:x[1])
			X_dev = [x for x,y in temp]
			X += X_dev
			Y += [y for x,y in temp]
		
		
		if len(X) == 0: continue
		
		
		
		plt.clf()
		plt.rcParams.update({'font.size': 40})
		fig, all_ax = plt.subplots(2, 3, sharex=True, sharey=True,figsize=(80,60))
		# for ax in all_ax.flat:
			# ax.set(xlabel='Canopies', ylabel=metric)
		
		for m_ctr,measure in enumerate(["avgMents","numMents","numSingletons","stdMents","origin"]):
			print("Plotting for measure={}".format(measure))
			canopy_stats = 	json.load(open("resources/aminer/aminer_{}.json".format(measure),"r"))
			
			p_x,p_y = int(m_ctr/3),m_ctr%3
			ax = all_ax[p_x,p_y]
			
			barlist = ax.bar(X, Y)
			
			vals  = sorted(list(set([canopy_stats[key] for key in canopy_stats])))
			minVal = min([v for v in vals if v > 0])
			maxVal = max(vals)
			numVals = len(vals)
			numColors= min(100,numVals)
			cmap =	sns.color_palette("YlOrRd", numColors)


			if measure in ["avgMents","numSingletons","stdMents","numMents"]:
				bins = np.geomspace(minVal, maxVal, numColors)
			else:
				bins = np.arange(minVal, maxVal,  (maxVal-minVal)/numColors)
			
			valToIdx = {val:i for i,val in enumerate(vals)}
			for i, val in enumerate(vals):
				valToIdx[val] = 0
				for j in range(len(bins)):
					if  bins[j] <= val:
						valToIdx[val] = j
						
			for idx,x in enumerate(X):
				if x in canopy_stats:
					val = canopy_stats[x]
					color = cmap[valToIdx[val]]
					barlist[idx].set_color(color)
				else:
					barlist[idx].set_color('r')
		
			Z = [canopy_stats[x] for x in X]
			corrCoeff  = np.corrcoef(Y, Z)[0, 1]
			plt.title("{}".format(metric))
			ax.set_title("{}\t{:.3f}".format(measure,corrCoeff))
			ax.set_xticks([])
			
		plt.savefig(currResultDir + "/{}.png".format(metric))
		plt.close()
		
		
	for m_ctr,measure in enumerate(["avgMents","numMents","numSingletons","stdMents","origin"]):
		print("Plotting for measure={}".format(measure))
		canopy_stats = 	json.load(open("resources/aminer/aminer_{}.json".format(measure),"r"))
		
		plt.clf()
		matplotlib.rcdefaults()
		plt.figure(figsize=(16,9))
		Z = [canopy_stats[x] for x in canopy_stats]
		
		plt.hist(Z,bins=len(set(Z)))
		plt.savefig(currResultDir + "/distribution_{}.png".format(measure))
		plt.close()
	
def plot_scores(allLosses, allScores, xlabel, currResultDir):
	
	# assert isinstance(config, Config)
	trainScores, devScores, testScores = allScores["train"], allScores["dev"], allScores["test"]
	
	Path(currResultDir).mkdir(parents=True, exist_ok=True)  # Create resultDir directory if not already present
	line_styles = json.load(open("resources/line_styles.json"))
	for lossType in allLosses:
		loss = allLosses[lossType]
		X = sorted(loss.keys())
		try:
			plt.clf()
			Y_within = [loss[x][0] for x in X]
			Y_across = [loss[x][1] for x in X]
			Y_total  = [y1+y2 for (y1,y2) in zip(Y_within,Y_across)]

			plt.plot(X, Y_within, 'go-', label='withinEdgeLoss')
			plt.plot(X, Y_across, 'r*-', label='acrossEdgeLoss')
			plt.plot(X, Y_total,  'bo-', label='totalLoss')
		except:
			plt.clf()
			Y = [loss[x] for x in X]
			plt.plot(X, Y, 'go-', label='totalLoss')

		plt.title("{} loss".format(lossType))
		plt.xlabel(xlabel)
		plt.ylabel("Loss")
		plt.legend()
		plt.grid()
		plt.savefig(currResultDir + "/{}Loss.png".format(lossType))
		plt.close()

	metricList = []
	for x in trainScores.keys():
		metricList += trainScores[x].keys()
		break
	for x in devScores.keys():
		metricList += devScores[x].keys()
		break
	for x in testScores.keys():
		metricList += testScores[x].keys()
		break
	
	metricList = list(set(metricList))
	for metric in metricList:
		if "euclid" in metric: continue
		X = sorted(list(set( list(trainScores.keys()) + list(testScores.keys()) + list(devScores.keys()) )))
		# X = sorted(list(trainScores.keys()))
		if len(X) == 0: continue
		# if metric not in trainScores[X[0]]: continue
		
		Y_train_mean  = np.array([trainScores[x][metric][0] if x in trainScores and metric in trainScores[x] else 0.
								  for x in X  ])
		Y_train_error = np.array([trainScores[x][metric][1] if x in trainScores and metric in trainScores[x] else 0.
								  for x in X ])

		Y_dev_mean 	= np.array([devScores[x][metric][0] if x in devScores and metric in devScores[x] else 0.
							   for x in X])
		Y_dev_error = np.array([devScores[x][metric][1] if x in devScores and metric in devScores[x] else 0.
								for x in X])

		Y_test_mean  = np.array([testScores[x][metric][0] if x in testScores and metric in testScores[x] else 0.
								for x in X])
		Y_test_error = np.array([testScores[x][metric][1] if x in testScores and metric in testScores[x] else 0.
								 for x in X ])

		plt.clf()
		plt.errorbar(X, Y_train_mean, Y_train_error, **line_styles["train"]["style"])
		plt.fill_between(X, Y_train_mean - Y_train_error, Y_train_mean + Y_train_error, **line_styles["train"]["fill style"])

		plt.errorbar(X, Y_dev_mean, Y_dev_error, **line_styles["dev"]["style"])
		plt.fill_between(X, Y_dev_mean - Y_dev_error, Y_dev_mean + Y_dev_error, **line_styles["dev"]["fill style"])

		plt.errorbar(X, Y_test_mean, Y_test_error, **line_styles["test"]["style"])
		plt.fill_between(X, Y_test_mean - Y_test_error, Y_test_mean + Y_test_error, **line_styles["test"]["fill style"])

		plt.title("{}".format(metric))
		plt.xlabel(xlabel)
		plt.ylabel(metric)
		plt.legend()
		plt.grid()
		plt.savefig(currResultDir + "/" + metric + ".png")
		plt.close()

def write_scores_separate(allLosses, allScores, currResultDir, xlabel):
	trainScores, devScores, testScores = allScores["train"], allScores["dev"], allScores["test"]
	
	Path(currResultDir).mkdir(parents=True, exist_ok=True)  # Create resultDir directory if not already present

	for lossType in allLosses:
		loss = allLosses[lossType]
		X = sorted(loss.keys())
		try:
			with open("{}/{}Loss.csv".format(currResultDir, lossType),"w") as f:
				header = [xlabel, "WithinEdgeLoss", "AcrossEdgeLoss","TotalLoss"]
				csvWriter = csv.DictWriter(f, fieldnames=header)
				for x in X:
					row = {xlabel:x, "WithinEdgeLoss":loss[x][0], "AcrossEdgeLoss":loss[x][1], "TotalLoss":loss[x][0]+loss[x][1]}
					csvWriter.writerow(rowdict=row)
				
		except:
			with open("{}/{}Loss.csv".format(currResultDir, lossType),"w") as f:
				header = [xlabel, "TotalLoss"]
				csvWriter = csv.DictWriter(f, fieldnames=header)
				for x in X:
					row = {xlabel: x, "TotalLoss": loss[x]}
					csvWriter.writerow(rowdict=row)
	
	metricList = []
	for x in trainScores.keys():
		metricList += trainScores[x].keys()
		break
	for x in devScores.keys():
		metricList += devScores[x].keys()
		break
	for x in testScores.keys():
		metricList += testScores[x].keys()
		break
	
	metricList = list(set(metricList))
	for metric in metricList:
		if "euclid" in metric: continue
		X = sorted(list(trainScores.keys()))
		if len(X) == 0: continue
		if metric not in trainScores[X[0]]: continue
		
		Y_train_mean  = [trainScores[x][metric][0] if x in trainScores and metric in trainScores[x] else 0.
						 for x in X  ]
		Y_train_error = [trainScores[x][metric][1] if x in trainScores and metric in trainScores[x] else 0.
						 for x in X ]

		Y_dev_mean 	= [devScores[x][metric][0] if x in devScores and metric in devScores[x] else 0.
						 for x in X]
		Y_dev_error = [devScores[x][metric][1] if x in devScores and metric in devScores[x] else 0.
					   for x in X]

		Y_test_mean  = [testScores[x][metric][0] if x in testScores and metric in testScores[x] else 0.
						for x in X]
		Y_test_error = [testScores[x][metric][1] if x in testScores and metric in testScores[x] else 0.
						for x in X ]

		with open("{}/{}.csv".format(currResultDir, metric),"w") as f:
			header = [xlabel, "train_{}_mean".format(metric), "train_{}_std".format(metric), "test_{}_mean".format(metric), "test_{}_std".format(metric),
					  "dev_{}_mean".format(metric),"dev_{}_std".format(metric)]
			csvWriter = csv.DictWriter(f=f,fieldnames=header)
			csvWriter.writeheader()
			for ctr,x in enumerate(X):
				row = {xlabel:x,
					   "train_{}_mean".format(metric)	: Y_train_mean[ctr],
					   "train_{}_std".format(metric)	: Y_train_error[ctr],
					   "test_{}_mean".format(metric)	: Y_test_mean[ctr],
					   "test_{}_std".format(metric)		: Y_test_error[ctr],
					   "dev_{}_mean".format(metric)		: Y_dev_mean[ctr],
					   "dev_{}_std".format(metric)		: Y_dev_error[ctr]}
				csvWriter.writerow(rowdict=row)
			
def write_scores_comb(allLosses, allScores, currResultDir, xlabel):
	trainLoss, devLoss, testLoss = allLosses["train"],allLosses["dev"],allLosses["test"]
	trainScores, devScores, testScores = allScores["train"],allScores["dev"],allScores["test"]
	Path(currResultDir).mkdir(parents=True, exist_ok=True)  # Create resultDir directory if not already present

	metricList = []
	for x in trainScores.keys():
		metricList += trainScores[x].keys()
		break
	for x in devScores.keys():
		metricList += devScores[x].keys()
		break
	for x in testScores.keys():
		metricList += testScores[x].keys()
		break
	
	metricList = list(set(metricList))
	with open(currResultDir + "/results.csv", "w") as f:
		
		header = [xlabel, "TrainLoss","DevLoss","TestLoss"]
		for metric in metricList:
			header += ["test_" + metric + "_mean", "test_" + metric + "_std"]
		
		for metric in metricList:
			header += ["train_" + metric + "_mean", "train_" + metric + "_std"]
		
		for metric in metricList:
			header += ["dev_" + metric + "_mean", "dev_" + metric + "_std"]
		
		
		csvWriter = csv.DictWriter(f, fieldnames=header)
		csvWriter.writeheader()
		
		xDimValList = list(trainScores.keys()) + list(testScores.keys()) + list(devScores.keys()) + list(trainLoss.keys())
		xDimValList = sorted(list(set(xDimValList))) # Get unique sorted xdim-values
		
		for xDim in xDimValList:
			row = {}
			row[xlabel] = xDim
			row["TrainLoss"] = trainLoss[xDim] if xDim in trainLoss else 0.
			row["TestLoss"] = testLoss[xDim] if xDim in testLoss else 0.
			row["DevLoss"] 	= devLoss[xDim] if xDim in devLoss else 0.

			for metric in metricList:
				row["train_" + metric + "_mean"] = trainScores[xDim][metric][0] if (xDim in trainScores) and (metric in trainScores[xDim]) else ""
				row["train_" + metric + "_std"]  = trainScores[xDim][metric][1] if (xDim in trainScores) and (metric in trainScores[xDim]) else ""
				
				row["test_" + metric + "_mean"] = testScores[xDim][metric][0] if (xDim in testScores) and (metric in testScores[xDim]) else ""
				row["test_" + metric + "_std"]  = testScores[xDim][metric][1] if (xDim in testScores) and (metric in testScores[xDim]) else ""
			
				row["dev_" + metric + "_mean"] = devScores[xDim][metric][0] if (xDim in devScores) and (metric in devScores[xDim]) else ""
				row["dev_" + metric + "_std"]  = devScores[xDim][metric][1] if (xDim in devScores) and (metric in devScores[xDim]) else ""
			
			# print("Writing row...{}".format(row))
			csvWriter.writerow(row)
			
def readCSV(currResultDir,varyDim):
	
	raise NotImplementedError
	header = None
	allScores = {"train":{}, "test":{}, "dev":{}}
	allLosses = {"train":{}, "test":{}, "dev":{}}
	filename  = currResultDir + "/results.csv"
	with open(filename,"r") as f:
		csvReader = csv.DictReader(f)
		for row in csvReader:
			if header is None:
				header = row.keys()
			
			for col in header:
				try:
					row[col]= float(row[col])
				except ValueError:
					row[col]=0
					
			varyDimValue = None
			for col in header:
				if col == varyDim:
					varyDimValue = row[col]
					break
			

			for col in header:
				assert varyDimValue is not None
				if col == "TrainLoss": continue
				if col == "TestLoss": continue
				if col == "DevLoss": continue
				if col == varyDim: continue
				if col.startswith("train"):
					try:
						allScores["train"][varyDimValue][col] = row[col]
					except:
						allScores["train"][varyDimValue] = {}
						allScores["train"][varyDimValue][col] = row[col]
						
				elif col.startswith("test"):
					try:
						allScores["test"][varyDimValue][col] = row[col]
					except:
						allScores["test"][varyDimValue] = {}
						allScores["test"][varyDimValue][col] = row[col]
				elif col.startswith("dev"):
					try:
						allScores["dev"][varyDimValue][col] = row[col]
					except:
						allScores["dev"][varyDimValue] = {}
						allScores["dev"][varyDimValue][col] = row[col]
					
		
			allLosses["train"][varyDimValue] = row["TrainLoss"]
			allLosses["test"][varyDimValue] = row["TestLoss"]
			allLosses["dev"][varyDimValue] = row["DevLoss"]


	inferenceMethods = {}
	additionalMetrics = {}

	finalAllScores = {"train":{},"test":{},"dev":{}}
	for scoreType in allScores:
		for x in allScores["train"].keys():
			finalAllScores[scoreType][x] = {"allAcrossEdges":[1],"allWithinEdges":[1]}

			reducedCols = {} # Column names with _mean and _std stripped off them
			for col in allScores[scoreType][x].keys():
				if col.endswith("allAcrossEdges") or col.endswith("allWithinEdges"): continue
				if col.endswith("_mean"):
					redCol = col[:-5]
				elif col.endswith("_std"):
					redCol = col[:-4]
				else:
					redCol = col

				reducedCols[redCol] = 1

				if col.endswith("f1_mean"):
					method = col[6:-8].split("_")[0]
					inferenceMethods[method] = 1
				if col.endswith("_mean") and len(col.split("_"))==3:
					additionalMetrics[col[6:-5]] = 1

			# 	For each column add its mean and std as tuple in finalAllScores
			for col in reducedCols:
				if col.startswith("train"):
					finalCol = col[6:]
				elif col.startswith("test"):
					finalCol = col[5:]
				elif col.startswith("dev"):
					finalCol = col[4:]
				else:
					finalCol = col

				try:
					finalAllScores[scoreType][x][finalCol] = allScores[scoreType][x][col+"_mean"],allScores[scoreType][x][col+"_std"]
				except KeyError:
					finalAllScores[scoreType][x][finalCol] = allScores[scoreType][x][col]


	
	inferenceMethods = list(inferenceMethods.keys())
	additionalMetrics = list(additionalMetrics.keys())

	result = {"allLosses":allLosses,
			  "allScores":finalAllScores,
			  "inferenceMethods":inferenceMethods,
			  "additionalMetrics":additionalMetrics
			  }
	return result

def plotMetricsFromCSV(currResultDir, xlabel):
	
	raise NotImplementedError
	resultDict = readCSV(currResultDir,xlabel)
	
	resultDict["currResultDir"] = currResultDir
	resultDict["xlabel"] = xlabel
	resultDict["skipEdges"] = True
	plot_scores(**resultDict)

if __name__ == "__main__":
	pass
	# parser = argparse.ArgumentParser("For Plotting data from csv file")
	#
	# parser.add_argument('--resultDir', type=str, default=None,help="Directory where csv file is present")
	# parser.add_argument('--varyDim', type=str, default=None, help='X-Axis dimension name')
	#
	# args = parser.parse_args()
	# resultDir = args.resultDir
	# varyDim = args.varyDim
	# plotMetricsFromCSV(currResultDir=resultDir, xlabel=varyDim)

