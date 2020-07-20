#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyspark
import json
import random 
import math
import itertools
import sys
import time

start = time.time()

input_path = sys.argv[1]
out_path = sys.argv[2]

def MinComp(l1, l2):
    res = []
    for i in range(len(l1)):
        re = min(l1[i], l2[i])
        res.append(re)
    return res

def splitBand(list_l, num_band):
    res = list()
    r = int(math.ceil(len(list_l) / int(num_band)))
    i = 0
    s = 0
    while s < len(list_l):
        res.append((i, (tuple(list_l[s:s + r]))))
        i += 1
        s += r
    return res


def JaccSim(cand_pair, busi_user_list,reversed_busi_rdd, threshhold = 0.05):
    res = []
    temp = set()
    for pair in cand_pair:
        pair = tuple(sorted(pair))
        if pair not in temp:
            temp.add(pair)
            user1 = set(busi_user_list[pair[0]])
            user2 = set(busi_user_list[pair[1]])
            JaccS = len(user1.intersection(user2))/len(user1.union(user2))
            if JaccS >= threshhold:
                true_pair = {"b1":reversed_busi_rdd[pair[0]],"b2":reversed_busi_rdd[pair[1]], "sim":JaccS}
    #             true_pair = [(reversed_busi_rdd[pair[0]],reversed_busi_rdd[pair[1]]),JaccS]

                res.append(true_pair)
    return res
    
# sc.stop()
sc = pyspark.SparkContext()

# review = sc.textFile('/Users/zailipeng/Desktop/my_research/Important_information/books/CS/Inf553/HW3/train_review.json')
review = sc.textFile(input_path)
rdd = review.map(json.loads).map(lambda row: (row['business_id'],row['user_id'])).persist()
# print(rdd.take(5))

# find distinct user ID with dictionary representation
user_rdd = rdd.map(lambda a: a[1]).distinct().sortBy(lambda a: a).zipWithIndex().map(lambda a: {a[0]:a[1]}).flatMap(lambda a: a.items())
user_rdd_dict = user_rdd.collectAsMap()
# find distinct business ID with dictionary representation
busi_rdd = rdd.map(lambda a: a[0]).distinct().sortBy(lambda a: a).zipWithIndex().map(lambda a: {a[0]:a[1]}).flatMap(lambda a: a.items())
busi_rdd_dict = busi_rdd.collectAsMap()
reversed_busi_dict = {v:k for k,v in busi_rdd_dict.items()}

# generate hash function 
def hash_fun(x,num_hash,m):
    hash_val = list()
    random.seed(42)
    a_nums = random.sample(range(100000,sys.maxsize - 1), num_hash)
    b_nums = random.sample(range(100000,sys.maxsize - 1), num_hash)
    p = 2860486313
    for a, b in zip(a_nums,b_nums):
        val = ((a*x+b)%p)%m
        hash_val.append(val)
    return hash_val

num_hash = 50
m = len(user_rdd_dict)
hash_value_rdd = user_rdd.map(lambda a: (user_rdd_dict[a[0]], hash_fun(a[1],num_hash,m)))

# print(hash_value_rdd.take(5))

# generate user with list of business
user_busi_rdd = rdd.map(lambda a: (user_rdd_dict[a[1]], busi_rdd_dict[a[0]])).groupByKey().map(lambda a: (a[0], list(set(a[1]))))
# print(user_busi_list.take(2))

# generate busi with list of users
busi_user_list = rdd.map(lambda a: (busi_rdd_dict[a[0]],user_rdd_dict[a[1]])).groupByKey().map(lambda a: {a[0]: list(set(a[1]))}).flatMap(lambda a: a.items()).collectAsMap()
# print(busi_user_list[0:5])

# generate min-hash signature 
sig_mat_rdd = user_busi_rdd.leftOuterJoin(hash_value_rdd).map(lambda a: a[1])
sig_mat_rdd = sig_mat_rdd.flatMap(lambda a: [(bix, a[1]) for bix in a[0]]).reduceByKey(MinComp)

#LSH
num_bands = 50
cand_pair = sig_mat_rdd.flatMap(lambda a: [(a[0], tuple(b)) for b in splitBand(a[1], num_bands)]).map(lambda a: (a[1],a[0]))
cand_pair = cand_pair.groupByKey().map(lambda a: list(set(a[1]))).filter(lambda a: len(a)>=2).collect()
cand_pair_s = []
for s in cand_pair:
    pairs = itertools.combinations(s,2)
    cand_pair_s.extend(pairs)
# print(cand_pair_s[0:50])
true_pair = JaccSim(cand_pair_s, busi_user_list,reversed_busi_dict, threshhold = 0.05)
# print(true_pair[0:5], len(true_pair))

# write to the output file

with open(out_path,'w') as zaili:
    for i in true_pair:
        zaili.writelines(json.dumps(i)+'\n')

end = time.time()
print('duration', end-start)

