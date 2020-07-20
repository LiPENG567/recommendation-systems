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

input_path = sys.argv[1]
out_path = sys.argv[2]
stopword_path = sys.argv[3]

start = time.time()

sc = pyspark.SparkContext()

review = sc.textFile(input_path)
rdd = review.map(json.loads).map(lambda row: (row['business_id'],row['text'],row['user_id'])).persist()
# print(rdd.take(1))
# print(rdd.getNumPartitions())

inputs = open(stopword_path,'r')

stopword = []
for line in inputs:
    lines = line.strip('\n')
    stopword.append(lines)
punc = string.punctuation 
for i in punc:
    stopword.append(i)
# print('stopword is',stopword)

def f(x):
    res = [i.split() for i in x.split('.')]
    res1 = [j for i in res for j in i if ((j.lower() not in stopword) and (j.isdigit()==False))]            
    return res1 

# construct doc 
doc_rdd = rdd.map(lambda a: (a[0],a[1])).mapValues(f).reduceByKey(lambda a,b:a+b).persist()
# # print(rdd.getNumPartitions())
# # # print(doc_rdd.take(1))
N = doc_rdd.count()
doc_word_freq = doc_rdd.flatMap(lambda a: [((a[0], b),1) for b in a[1]]).reduceByKey(lambda a,b:a+b).map(lambda a: ((a[0][0], (a[0][1], a[1])))).groupByKey()
# print(doc_word_freq.take(5))
doc_freq = doc_rdd.flatMapValues(lambda x: x).distinct().map(lambda a: (a[1], a[0])).countByKey()

def tf_idf(word_cnt_list,doc_freq,N):
    # find the max count in a list for normalization
    max_num = max([int(i[1]) for i in word_cnt_list])
    res = {}
    for word in word_cnt_list:
        if word[0] not in res:
            res[word[0]] = int(word[1])/max_num*math.log2(int(N)/int(doc_freq[word[0]]))
    sorted_res = list(sorted(res.items(), key=operator.itemgetter(1),reverse=True))[:200]
    return sorted_res

# # calculate the TF-IDF score using RDD
score_rdd = doc_word_freq.map(lambda a: (a[0], list(tf_idf(a[1],doc_freq,N)))).filter(lambda a: len(a[1])>1)
busi_prof = score_rdd.mapValues(lambda a: [b[0] for b in a]).collect()

def jsonm(data, prof_type, keys):
    res = list()
    for da in data:
        res.append({"type":prof_type, keys[0]:da[0], keys[1]:da[1]})
    return res

model=[]

model.extend(jsonm(busi_prof, 'busi_profile', keys= ['busi_idx','bu_profile']))

def topsize(x):
    res = sorted(x, key=lambda a: a[1],reverse=True)[:200]
    return res    

# AA = [('Public', 1.5515722812619075), ('cauliflower', 1.3117357206638696), ('School', 1.076269543886442), ('beer', 0.9059110263221742), ('buffalo', 0.7555246489750946), ('food', 0.751531966912614), ('patio', 0.7428901650037993), ('burger', 0.7364154085599534), ('brunch', 0.7016054424474355), ('tots', 0.653054858430742), ('Summerlin', 0.6454473906704423), ('server', 0.635118305665709), ('Downtown', 0.5911800164271135), ('menu', 0.5895806720414203), ('grits', 0.5577237772975969), ('pizza', 0.512097088256842)]
# generate user profile
busi_user_rdd = rdd.map(lambda a: (a[0],a[2]))
user_prof_rdd = busi_user_rdd.leftOuterJoin(score_rdd).map(lambda a: a[1]).filter(lambda a: a[1] is not None).reduceByKey(lambda a,b:a+b).mapValues(topsize).mapValues(lambda a: [b[0] for b in a])
# print(user_prof_rdd.take(1))
user_prof = user_prof_rdd.collect()
model.extend(jsonm(user_prof, 'user_profile', keys= ['user_idx','use_profile']))

# write to the model output file 
with open(out_path,'w') as zaili:
    for i in model:
        zaili.writelines(json.dumps(i)+'\n')
end = time.time()
print('Duration', end-start)

