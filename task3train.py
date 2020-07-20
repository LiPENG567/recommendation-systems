#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
import json
import random 
import math
import itertools
import sys
import time 
import string  
import operator

train_path = sys.argv[1]
out_path = sys.argv[2]
cf_type = sys.argv[3]
start = time.time()



def flat(a):
    res = {}
    for i in a:
        res.update(i)
    return res

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

def JaccSim(cand_pair, busi_user_list,threshhold = 0.01):

    temp = set()

    pair = tuple(sorted(cand_pair))

    user1 = set(busi_user_list[pair[0]])
    user2 = set(busi_user_list[pair[1]])
    numer = len(user1.intersection(user2))
    JaccS = numer/len(user1.union(user2))
    if JaccS >= threshhold and numer >= 3:
        true_pair = (tuple(sorted((pair[0],pair[1]))))
        return true_pair
    else:
        return False
    
def perason(la,lb):
    user_com = set(la.keys()).intersection(set(lb.keys()))
    star_a = [la[k] for k in user_com]
    star_b = [lb[k] for k in user_com]
    pear_cor = 0

    av_a = sum(star_a)/len(star_a)
    av_b = sum(star_b)/len(star_b)
    numerator = sum([(star_a[i]-av_a)*(star_b[i]-av_b) for i in range(len(user_com))])
    demom = math.sqrt(sum([(star_a[i]-av_a)**2 for i in range(len(user_com))]))*math.sqrt(sum([(star_b[i]-av_b)**2 for i in range(len(user_com))]))
    if demom != 0:
        pear_cor = numerator/demom
    return pear_cor

def check3(dict1, dict2):
    ins_len = len(set(dict1.keys()).intersection(set(dict2.keys())))
    if ins_len >= 3:
        return True
    else:
        return False
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

sc = pyspark.SparkContext()
review = sc.textFile(train_path)
# review = sc.textFile('/Users/zailipeng/Desktop/my_research/Important_information/books/CS/Inf553/HW3/train_review.json')
rdd = review.map(json.loads).map(lambda row: (row['business_id'],row['user_id'],row['stars'])).persist()
if cf_type == "item_based":
    user_idx_dict = rdd.map(lambda a: a[1]).distinct().sortBy(lambda a:a).zipWithIndex().map(lambda a: {a[0]:a[1]}).flatMap(lambda a: a.items()).collectAsMap()
    idx_user_dict = {v:k for k,v in user_idx_dict.items()}

    busi_idx_dict = rdd.map(lambda a: a[0]).distinct().sortBy(lambda a:a).zipWithIndex().map(lambda a: {a[0]:a[1]}).flatMap(lambda a: a.items()).collectAsMap()
    idx_busi_dict = {v:k for k,v in busi_idx_dict.items()}

    busi_rdd = rdd.map(lambda a: (busi_idx_dict[a[0]],(user_idx_dict[a[1]],a[2]))).groupByKey().map(lambda a: (a[0],list(a[1]))).filter(lambda a: len(a[1])>= 3).mapValues(lambda a: [{b[0]:b[1]} for b in a]).mapValues(lambda a: flat(a))
    bus_user_star_dict = busi_rdd.map(lambda a: {a[0]:a[1]}).flatMap(lambda a: a.items()).collectAsMap()

    new_rdd = busi_rdd.map(lambda a: a[0])

    # new_rddr = new_rdd.cartesian(new_rdd).filter(lambda a: a[0]< a[1]).map(lambda a: ((a[0][0], a[1][0]), perason(a[0][1], a[1][1]))).filter(lambda a: a[1]>0)
    new_rddr = new_rdd.cartesian(new_rdd).filter(lambda a: a[0]< a[1]).filter(lambda a: check3(bus_user_star_dict[a[0]],bus_user_star_dict[a[1]])).map(lambda a: ((a[0], a[1]), perason(bus_user_star_dict[a[0]],bus_user_star_dict[a[1]]))).filter(lambda a: a[1]>0)
    out_list = new_rddr.map(lambda a: {"b1": idx_busi_dict[a[0][0]], "b2": idx_busi_dict[a[0][1]], "sim": a[1]}).collect()
    # print(out_list[0:5])
else:
        # find distinct user ID with dictionary representation
    user_rdd = rdd.map(lambda a: a[1]).distinct().sortBy(lambda a: a).zipWithIndex().map(lambda a: {a[0]:a[1]}).flatMap(lambda a: a.items())
    user_rdd_dict = user_rdd.collectAsMap()
    reversed_user_dict = {v:k for k,v in user_rdd_dict.items()}

    # find distinct business ID with dictionary representation
    busi_rdd = rdd.map(lambda a: a[0]).distinct().sortBy(lambda a: a).zipWithIndex().map(lambda a: {a[0]:a[1]}).flatMap(lambda a: a.items())
    busi_rdd_dict = busi_rdd.collectAsMap()
    reversed_busi_dict = {v:k for k,v in busi_rdd_dict.items()}

    num_hash = 20
    m = len(busi_rdd_dict)
    hash_value_rdd = busi_rdd.map(lambda a: (busi_rdd_dict[a[0]], hash_fun(a[1],num_hash,m)))

    # print(hash_value_rdd.take(5))

    # generate user with list of business
    user_busi_rdd = rdd.map(lambda a: (user_rdd_dict[a[1]], busi_rdd_dict[a[0]])).groupByKey().map(lambda a: {a[0]: list(set(a[1]))}).flatMap(lambda a: a.items()).collectAsMap()
    user_busi_score_rdd = rdd.map(lambda a: (user_rdd_dict[a[1]], (busi_rdd_dict[a[0]],a[2]))).groupByKey().map(lambda a: {a[0]: list(set(a[1]))}).flatMap(lambda a: a.items()).collectAsMap()

    busi_rddr = rdd.map(lambda a: (user_rdd_dict[a[1]],(busi_rdd_dict[a[0]],a[2]))).groupByKey().map(lambda a: (a[0],list(a[1]))).filter(lambda a: len(a[1])>= 3).mapValues(lambda a: [{b[0]:b[1]} for b in a]).mapValues(lambda a: flat(a))
    user_bus_star_dict = busi_rddr.map(lambda a: {a[0]:a[1]}).flatMap(lambda a: a.items()).collectAsMap()

    # print(user_busi_list.take(2))

    # generate busi with list of users
    busi_user_list = rdd.map(lambda a: (busi_rdd_dict[a[0]],user_rdd_dict[a[1]])).groupByKey().map(lambda a: (a[0], list(set(a[1]))))
    # print(busi_user_list[0:5])

    # generate min-hash signature 
    sig_mat_rdd = busi_user_list.leftOuterJoin(hash_value_rdd).map(lambda a: a[1])
    sig_mat_rdd = sig_mat_rdd.flatMap(lambda a: [(bix, a[1]) for bix in a[0]]).reduceByKey(MinComp)


    #LSH
    num_bands = 20
    cand_pair = sig_mat_rdd.flatMap(lambda a: [(a[0], tuple(b)) for b in splitBand(a[1], num_bands)]).map(lambda a: (a[1],a[0]))
    cand_pair = cand_pair.groupByKey().map(lambda a: list(set(a[1]))).filter(lambda a: len(a)>=2).flatMap(lambda a: [b for b in itertools.combinations(a,2)]).map(lambda a: JaccSim(a, user_busi_rdd,threshhold = 0.01)).filter(lambda a : a!= False).distinct()

    pear_rdd = cand_pair.map(lambda a: ((a[0],a[1]),perason(user_bus_star_dict[a[0]],user_bus_star_dict[a[1]]))).filter(lambda a: a[1]>0)
    out_list = pear_rdd.map(lambda a: {"u1": reversed_user_dict[a[0][0]], "u2": reversed_user_dict[a[0][1]], "sim": a[1]}).collect()

with open(out_path,'w') as zaili:
    for i in out_list:
        zaili.writelines(json.dumps(i)+'\n')

end = time.time()
print('Duration', end-start)


# In[ ]:




