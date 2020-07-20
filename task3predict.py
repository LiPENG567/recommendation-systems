#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import string  
import operator

train_path = sys.argv[1]
test_path = sys.argv[2]
model_path = sys.argv[3]
output_path = sys.argv[4]
cf_type = sys.argv[5]

start = time.time()
sc = pyspark.SparkContext()

N = 7

# train review set
train_review = sc.textFile(train_path)
train_rev_rdd = train_review.map(json.loads).map(lambda row: (row['business_id'],row['user_id'],row['stars'])).persist()

if cf_type == "item_based":
    user_idx_dict = train_rev_rdd.map(lambda a: a[1]).distinct().sortBy(lambda a:a).zipWithIndex().map(lambda a: {a[0]:a[1]}).flatMap(lambda a: a.items()).collectAsMap()
    idx_user_dict = {v:k for k,v in user_idx_dict.items()}

    busi_idx_dict = train_rev_rdd.map(lambda a: a[0]).distinct().sortBy(lambda a:a).zipWithIndex().map(lambda a: {a[0]:a[1]}).flatMap(lambda a: a.items()).collectAsMap()
    idx_busi_dict = {v:k for k,v in busi_idx_dict.items()}

    user_rdd = train_rev_rdd.map(lambda a: (user_idx_dict[a[1]],(busi_idx_dict[a[0]],a[2]))).groupByKey().map(lambda a: (a[0], [(b[0],b[1]) for b in list(set(a[1]))]))
    # print(user_rdd.take(5))

    # train_rev_dict = train_rev_rdd.map(lambda a: {(user_idx_dict[a[1]],busi_idx_dict[a[0]]):a[2]}).flatMap(lambda a: a.items()).collectAsMap()



    # model
    model = sc.textFile(model_path)
    model_rdd_dict= model.map(lambda a: json.loads(a)).map(lambda a: {(busi_idx_dict[a['b1']],busi_idx_dict[a['b2']]):a['sim']}).flatMap(lambda a: a.items()).collectAsMap()
    # # model_list = model.map(json.loads).map(lambda a: (tuple(sorted((a[0],a[1]))),a[2])).distinct().collect()

    # model_rdd = model.map(json.loads).map(lambda row: (row['b1'],row['b2'],row['sim']))
    # model_list = model_rdd.map(lambda a: (tuple(sorted((a[0],a[1]))),a[2])).distinct().collect()
    # print(model_list[0:10])


    def topn(bu_i,j_list, model_dict,N):
        fin = 0
        res = list()
        for a in j_list:
            if a[0] < bu_i:
                key = tuple((a[0],bu_i))
            else:
                key = tuple((bu_i,a[0]))
    #         result.append(tuple(a[1], model_dict.get(key,0)))
            if key in model_dict.keys():
                res.append((a[1], model_dict[key]))

        top_ns = sorted(res,key=lambda item:item[1], reverse=True)[:N]
        nume =sum([float(a[0])*float(a[1]) for a in top_ns])
        donom = sum(float(a[1]) for a in top_ns)
        if donom == 0:
            return (bu_i,0)
        else:
            return (bu_i, nume/donom)                  

    # test review set
    test= sc.textFile(test_path)
    # # # test_rdd = test.map(lambda a: json.loads(a)).map(lambda row: (user_idx_dict[row['user_id']],busi_idx_dict[row['business_id']])).map(lambda a: ((a[0],a[1]), topn(a[0],a[1],model_list,N, train_rev_dict)))
    # # test_rdd = test.map(lambda a: json.loads(a)).map(lambda row: (user_idx_dict[row['user_id']],busi_idx_dict[row['business_id']])).filter(lambda a: a[0] in idx_user_dict.keys() and a[1] in idx_busi_dict.keys())
    test_rdd = test.map(lambda a: json.loads(a)).map(lambda row: (row['user_id'],row['business_id'])).filter(lambda a: a[0] in user_idx_dict.keys() and a[1] in busi_idx_dict.keys()).map(lambda row: (user_idx_dict[row[0]],busi_idx_dict[row[1]]))
    test_rddr = test_rdd.leftOuterJoin(user_rdd).mapValues(lambda a: topn(a[0],a[1],model_rdd_dict,N)).filter(lambda a: a[1][1]!=0)
    test_score_list = test_rddr.map(lambda a: {"user_id": idx_user_dict[a[0]], "business_id": idx_busi_dict[a[1][0]], "stars": a[1][1]}).collect()
else:

    user_idx_dict = train_rev_rdd.map(lambda a: a[1]).distinct().sortBy(lambda a:a).zipWithIndex().map(lambda a: {a[0]:a[1]}).flatMap(lambda a: a.items()).collectAsMap()
    idx_user_dict = {v:k for k,v in user_idx_dict.items()}

    busi_idx_dict = train_rev_rdd.map(lambda a: a[0]).distinct().sortBy(lambda a:a).zipWithIndex().map(lambda a: {a[0]:a[1]}).flatMap(lambda a: a.items()).collectAsMap()
    idx_busi_dict = {v:k for k,v in busi_idx_dict.items()}

    busi_rdd = train_rev_rdd.map(lambda a: (busi_idx_dict[a[0]],(user_idx_dict[a[1]],a[2]))).groupByKey().map(lambda a: (a[0], [(b[0],b[1]) for b in list(set(a[1]))]))

    user_busi_star_dict = train_rev_rdd.map(lambda a: (user_idx_dict[a[1]],a[2])).groupByKey().map(lambda a: {a[0]: list(a[1])}).flatMap(lambda a: a.items()).collectAsMap()
    # print(user_busi_star_dict.take(1))

    # train_rev_dict = train_rev_rdd.map(lambda a: {(user_idx_dict[a[1]],busi_idx_dict[a[0]]):a[2]}).flatMap(lambda a: a.items()).collectAsMap()



    # # model
    model = sc.textFile(model_path)
    model_rdd_dict= model.map(lambda a: json.loads(a)).map(lambda a: {(user_idx_dict[a['u1']],user_idx_dict[a['u2']]):a['sim']}).flatMap(lambda a: a.items()).collectAsMap()
    # # model_list = model.map(json.loads).map(lambda a: (tuple(sorted((a[0],a[1]))),a[2])).distinct().collect()

    # model_rdd = model.map(json.loads).map(lambda row: (row['b1'],row['b2'],row['sim']))
    # model_list = model_rdd.map(lambda a: (tuple(sorted((a[0],a[1]))),a[2])).distinct().collect()
    # print(model_list[0:10])


    def topn(bu_i,j_list, model_dict,N,user_busi_star_dict):
        fin = 0
        res = list()
        for a in j_list:
            if a[0] < bu_i:
                key = tuple((a[0],bu_i))
            else:
                key = tuple((bu_i,a[0]))
    #         result.append(tuple(a[1], model_dict.get(key,0)))
            if key in model_dict.keys():
                res.append(((a[1],a[0]), model_dict[key]))

        top_ns = sorted(res,key=lambda item:item[1], reverse=True)[:N]
        nume1 =sum([float(a[0][0])*float(a[1]) for a in top_ns])
        donom = sum(float(a[1]) for a in top_ns)
        # calculate the average of user 
        bu_i_av = sum([float(m) for m in user_busi_star_dict[bu_i]])/len([float(m) for m in user_busi_star_dict[bu_i]])
    #     user_star_list = [a[0] for a in top_ns]
        redu_sum = list()
        for u in top_ns:
            u_list = [float(m) for m in user_busi_star_dict[u[0][1]]]
            av_u = (sum(u_list)-u[0][0])/len(u_list)*u[1]
            redu_sum.append(av_u)
        nume = nume1 - sum(redu_sum)

        if donom == 0:
            return (bu_i,0)
        else:
            return (bu_i, nume/donom+bu_i_av)                  

    # test review set
    test= sc.textFile(test_path)
    # # # test_rdd = test.map(lambda a: json.loads(a)).map(lambda row: (user_idx_dict[row['user_id']],busi_idx_dict[row['business_id']])).map(lambda a: ((a[0],a[1]), topn(a[0],a[1],model_list,N, train_rev_dict)))
    # # test_rdd = test.map(lambda a: json.loads(a)).map(lambda row: (user_idx_dict[row['user_id']],busi_idx_dict[row['business_id']])).filter(lambda a: a[0] in idx_user_dict.keys() and a[1] in idx_busi_dict.keys())
    test_rdd = test.map(lambda a: json.loads(a)).map(lambda row: (row['user_id'],row['business_id'])).filter(lambda a: a[0] in user_idx_dict.keys() and a[1] in busi_idx_dict.keys()).map(lambda row: (busi_idx_dict[row[1]],user_idx_dict[row[0]]))
    test_rddr = test_rdd.leftOuterJoin(busi_rdd).mapValues(lambda a: topn(a[0],a[1],model_rdd_dict,N,user_busi_star_dict)).filter(lambda a: a[1][1]!=0)

    test_score_list = test_rddr.map(lambda a: {"user_id": idx_user_dict[a[1][0]], "business_id": idx_busi_dict[a[0]], "stars": a[1][1]}).collect()    

with open(output_path,'w') as zaili:
    for i in test_score_list:
        zaili.writelines(json.dumps(i)+'\n')


end = time.time()
print('duration', end-start)

