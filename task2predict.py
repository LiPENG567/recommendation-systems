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
import sys

test_path = sys.argv[1]
model_path = sys.argv[2]
output_path = sys.argv[3]

start = time.time()

sc = pyspark.SparkContext()

model = sc.textFile(model_path)
# rdd = model.map(json.loads).map(lambda row: (row['busi_idx'],row['bu_profile'], row['user_idx'],row['use_profile'])).persist()
rdd = model.map(lambda a: json.loads(a))
user_prof_dict = rdd.filter(lambda a: a['type'] == 'user_profile').map(lambda a: {a['user_idx']:a['use_profile']}).flatMap(lambda a: a.items()).collectAsMap()
busi_prof_dict = rdd.filter(lambda a: a['type'] == 'busi_profile').map(lambda a: {a['busi_idx']:a['bu_profile']}).flatMap(lambda a: a.items()).collectAsMap()

test_review = sc.textFile(test_path)
test_rdd = test_review.map(lambda a: json.loads(a)).map(lambda a: (a['user_id'],a['business_id']))
# print('inital count is', test_rdd.count())
def cos_cal(user_id,business_id,user_prof_dict,busi_prof_dict):
    cos_sim = 0
    if user_id in user_prof_dict.keys() and business_id in busi_prof_dict.keys():
        user_vector = set(user_prof_dict[user_id])
        busi_vector = set(busi_prof_dict[business_id])
        if len(user_vector)>=1 and len(busi_vector)>=1:
            cos_sim = len(user_vector.intersection(busi_vector))/(math.sqrt(len(user_vector))*math.sqrt(len(busi_vector)))
    return cos_sim 

# test_rdd_Cos = test_rdd.map(lambda a: ((a[0],a[1]), cos_cal(a[0],a[1],user_prof_dict,busi_prof_dict)))
test_rdd_Cos = test_rdd.map(lambda a: ((a[0],a[1]), cos_cal(a[0],a[1],user_prof_dict,busi_prof_dict))).filter(lambda a: a[1]>= 0.01)
tet_res_list = test_rdd_Cos.map(lambda a: {'user_id': a[0][0], 'business_id': a[0][1], 'sim': a[1]}).collect()

with open(output_path,'w') as zaili:
    for i in tet_res_list:
        zaili.writelines(json.dumps(i)+'\n')

end = time.time()
print('Duration', end-start)

