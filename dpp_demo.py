#!/bin/env python2.7
#coding:utf-8

import numpy as np
import time


def testData():
    scores = []
    dssms = []
    uuids = []
    with open('./data.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            d = line.split('\t')
            uuid = d[0]
            score = float(d[1])
            dssm = [float(x) for x in d[2].split(',')]
            if not uuid: continue
            scores.append(score)
            dssms.append(dssm)
            uuids.append(uuid)
    rank_score = np.array(scores)
    item_embedding = np.array(dssms)
    uuids = np.array(uuids)

    return uuids, rank_score, item_embedding

class DPPModel(object):
    def __init__(self, **kwargs):
        self.item_count = kwargs['item_count']
        self.item_embed_size = kwargs['item_embed_size']
        self.max_iter = kwargs['max_iter']
        self.epsilon = kwargs['epsilon']
 
    def build_kernel_matrix(self):
        uuids, rank_score, item_embedding = testData()
        rank_score = rank_score[:self.item_count]
        item_embedding = item_embedding[:self.item_count]
        sim_matrix = np.dot(item_embedding, item_embedding.T)  # item之间的相似度矩阵
        self.kernel_matrix = rank_score.reshape((self.item_count, 1)) * sim_matrix * rank_score.reshape((1, self.item_count))
        self.uuids = uuids

 
    def dpp(self):
        c = np.zeros((self.max_iter, self.item_count))
        d = np.copy(np.diag(self.kernel_matrix))
        j = np.argmax(d)
        Yg = [j]
        iter = 0
        Z = list(range(self.item_count))
        while len(Yg) < self.max_iter:
            Z_Y = set(Z).difference(set(Yg))
            for i in Z_Y:
                if iter == 0:
                    ei = self.kernel_matrix[j, i] / np.sqrt(d[j])
                else:
                    ei = (self.kernel_matrix[j, i] - np.dot(c[:iter, j], c[:iter, i])) / np.sqrt(d[j])
                c[iter, i] = ei
                d[i] = d[i] - ei * ei
            d[j] = 0
            j = np.argmax(d)
            if d[j] < self.epsilon:
                break
            Yg.append(j)
            iter += 1
        # print(Yg)
        # print(self.uuids[Yg])
        return Yg.sort()
 
if __name__ == "__main__":
    kwargs = {
        'item_count': 20,
        'item_embed_size': 32,
        'max_iter': 6,
        'epsilon': 0.01
    }
    dpp_model = DPPModel(**kwargs)
    dpp_model.build_kernel_matrix()
    start_time = time.time()
    for i in range(1000):
        rank = dpp_model.dpp()
    end_time = time.time()
    cost = (end_time  - start_time) * 1000
    avg = cost / 1000
    print('cost {}ms, avg {}ms'.format(cost, avg))
