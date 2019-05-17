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
import time

def write_tree(children_, Y, X_labels, filename):
	"""

	The children of each non-leaf node. Values less than n_samples correspond to leaves of the tree which are the
	original samples. A node i greater than or equal to n_samples is a non-leaf node and has children
	 children_[i - n_samples]. Alternatively at the i-th iteration, children[i][0] and children[i][1]
	 are merged to form node n_samples + i



	Args:
		children_:
		Y:
		fn:

	Returns:

	"""
	num_samples = len(Y)
	with open(filename, 'w') as fout:
		for i in range(0, len(children_)):
			node_i_id = "id_" + str(i + num_samples)

			if children_[i][0] < num_samples:
				child_0_node_id = str(X_labels[int(children_[i][0])])
				child_0_label = str(Y[int(children_[i][0])])
			else:
				child_0_node_id = "id_" + str(int(children_[i][0]))
				child_0_label = "None"

			if children_[i][1] < num_samples:
				child_1_node_id = str(X_labels[int(children_[i][1])])
				child_1_label = str(Y[int(children_[i][1])])
			else:
				child_1_node_id = "id_" + str(int(children_[i][1]))
				child_1_label = "None"

			fout.write("{}\t{}\t{}\n".format(child_0_node_id, node_i_id, child_0_label))
			fout.write("{}\t{}\t{}\n".format(child_1_node_id, node_i_id, child_1_label))
		root_ = "id_" + str(len(children_) + num_samples - 1)
		fout.write("{}\tNone\tNone\n".format(root_))

def calc_dend_purity(linkTree, pidList, y_true):
	dendPurity = 0
	XCLUSTER_ROOT = os.getenv("XCLUSTER_ROOT")
	filenum = time.time()
	treeFileName = "{}/perchTree_{}.tree".format(XCLUSTER_ROOT, filenum)
	
	while os.path.isfile(treeFileName):
		filenum = time.time()
		treeFileName = "{}/perchTree_{}.tree".format(XCLUSTER_ROOT, filenum)
	
	
	if isinstance(linkTree, str): # If linkTree is already a formatted string then just write it
		with open(treeFileName, "w") as f:
			f.write(linkTree)
	else:
		write_tree(linkTree, y_true, pidList, treeFileName)
	
	
	assert os.path.isfile(treeFileName)
	
	command = "cd $XCLUSTER_ROOT && source bin/setup.sh && pwd && "
	command += "sh bin/util/score_tree.sh {} algo data 24 None > treeResult_{}".format(treeFileName, filenum)
	os.system(command)
	
	resultFileName = "{}/treeResult_{}".format(XCLUSTER_ROOT, filenum)
	with open(resultFileName, "r") as reader:
		for line in reader:
			algo, data, dendPurity = line.split()
			dendPurity = float(dendPurity)
			break
			
	command = "rm {} && rm {}".format(treeFileName, resultFileName)
	# print("Removing files:{}".format(command))
	os.system(command)
	assert not os.path.isfile(treeFileName)
	assert not os.path.isfile(resultFileName)
	return dendPurity
