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


"""Run hierarchical sparsest cut."""
import argparse
import datetime
import numpy as np
import uuid
import os
import sys

from itertools import combinations

from sklearn.cluster import SpectralClustering

def log_exp_minus_dist(x, y):
    # return -((x - y).norm(2, 1)).unsqueeze(1)
    return np.linalg.norm(-(x - y))


def log_1_by_1p_dist(x, y):
    # return - torch.log1p(np.sqrt((x - y).norm(2, 1))).unsqueeze(1)
    return - np.log1p(np.linalg.norm(np.sqrt(x - y)))

def sparsest_cut(sims):
    if len(sims) == 2:
        return [0],[1]
    else:
        spectral = SpectralClustering(n_clusters=2,n_jobs=-1,affinity='precomputed')
        labels = spectral.fit_predict(sims)
        # print("SC gives: ")
        # print(labels)
        left = np.where(labels==0)[0].astype(np.int)
        # print("left")
        # print(left)
        right = np.where(labels==1)[0].astype(np.int)
        # print("right")
        # print(right)
        return left,right

def run(sim_file,label_file,out_file):
    sims = np.load(sim_file)
    labels = np.load(label_file)

    # (Node id, parent id, label, mat, objs)
    output = ''
    frontier = [(uuid.uuid4(), 'None', 'None', sims, np.arange(labels.shape[0]))]
    num_done = 0
    while frontier:
        # print("Splits on frontier: {}. Completed {}".format(len(frontier), num_done))
        nid, pid, label, mat, obs = frontier.pop(0)
        output += '%s\t%s\t%s\n' % (nid, pid, label)
        if obs.shape[0] > 1:
            l, r = sparsest_cut(mat)
            # Sometimes, this sparsest cut will not split the nodes. If this is
            # the case, we need to manually split them.
            if np.size(l) == 0:
                raise Exception('bad case...')
                l = [0]
                r = list(range(1, len(obs)))
            if np.size(r) == 0:
                raise Exception('bad case...')
                r = [0]
                l = list(range(1, len(obs)))

            if np.size(l) > 1:
                l_nid = uuid.uuid4()
                l_label = 'None'
            else:
                assert (np.size(l) == 1)
                l_nid = obs[l[0]]
                l_label = labels[obs[l[0]]]

            if np.size(r) > 1:
                r_nid = uuid.uuid4()
                r_label = 'None'
            else:
                assert (np.size(r) == 1)
                r_nid = obs[r[0]]
                r_label = labels[obs[r[0]]]

            # print(obs)
            l_obs = np.array([obs[i] for i in l])
            # print(l_obs)
            r_obs = np.array([obs[i] for i in r])
            # print(r_obs)
            frontier.append((l_nid, nid, l_label, mat[l, :][:, l], l_obs))
            frontier.append((r_nid, nid, r_label, mat[r, :][:, r], r_obs))
            # print(num_done)

    now = datetime.datetime.now()
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month,
                                                            now.day, now.hour,
                                                            now.minute,
                                                            now.second)
    out_dir = os.path.basename(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(out_file, 'w') as fout:
        fout.write(output)


def run_sparsest_cut(sims, labels ):
    
    # sims = np.load(sim_file)
    from scipy.spatial.distance import cdist
    # sims = cdist(transformedPointList,transformedPointList)
    # labels = np.array([pidToGtCluster[i] for i in range(len(pidToGtCluster))])
    
    # (Node id, parent id, label, mat, objs)
    output = ''
    frontier = [(uuid.uuid4(), 'None', 'None', sims, np.arange(labels.shape[0]))]
    num_done = 0
    while frontier:
        # print("Splits on frontier: {}. Completed {}".format(len(frontier), num_done))
        nid, pid, label, mat, obs = frontier.pop(0)
        output += '%s\t%s\t%s\n' % (nid, pid, label)
        if obs.shape[0] > 1:
            l, r = sparsest_cut(mat)
            # Sometimes, this sparsest cut will not split the nodes. If this is
            # the case, we need to manually split them.
            if np.size(l) == 0:
                raise Exception('bad case...')
                l = [0]
                r = list(range(1, len(obs)))
            if np.size(r) == 0:
                raise Exception('bad case...')
                r = [0]
                l = list(range(1, len(obs)))

            if np.size(l) > 1:
                l_nid = uuid.uuid4()
                l_label = 'None'
            else:
                assert (np.size(l) == 1)
                l_nid = obs[l[0]]
                l_label = labels[obs[l[0]]]

            if np.size(r) > 1:
                r_nid = uuid.uuid4()
                r_label = 'None'
            else:
                assert (np.size(r) == 1)
                r_nid = obs[r[0]]
                r_label = labels[obs[r[0]]]

            # print(obs)
            l_obs = np.array([obs[i] for i in l])
            # print(l_obs)
            r_obs = np.array([obs[i] for i in r])
            # print(r_obs)
            frontier.append((l_nid, nid, l_label, mat[l, :][:, l], l_obs))
            frontier.append((r_nid, nid, r_label, mat[r, :][:, r], r_obs))
            # print(num_done)

    return output
    

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3])
