import numpy as np
import itertools
import rbo

def get_topic_diversity(beta, topk):
    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k,:].argsort()[-topk:][::-1]
        list_w[k,:] = idx
    n_unique = len(np.unique(list_w))
    TD = n_unique / (topk * num_topics)
    print('Topic diveristy is: {}'.format(TD))
    return TD

class Measure:
    def __init__(self):
        pass

    def score(self):
        pass

class InvertedRBO(Measure):
    def __init__(self, beta):
        """
        :param topics: a list of lists of words
        """
        super().__init__()
        self.beta = beta

    def score(self, topk=10, weight=0.9):
        """
        :param weight: p (float), default 1.0: Weight of each agreement at
         depth d: p**(d-1). When set to 1.0, there is no weight, the rbo
         returns to average overlap.
        :return: rank_biased_overlap over the topics
        """
        num_topics = self.beta.shape[0]
        list_w = np.zeros((num_topics, topk))
        for k in range(num_topics):
            idx = self.beta[k,:].argsort()[-topk:][::-1]
            list_w[k,:] = idx
        self.topics = list_w
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            collect = []
            for list1, list2 in itertools.combinations(self.topics, 2):
                rbo_val = rbo.rbo(list1[:topk], list2[:topk], p=weight)[2]
                collect.append(rbo_val)
            return 1 - np.mean(collect)

