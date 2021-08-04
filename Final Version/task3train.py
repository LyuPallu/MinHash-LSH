from itertools import combinations
import sys
import time
import json
import re
from pyspark import SparkContext
from operator import add
from collections import Counter
from math import sqrt

start = time.time()
sc = SparkContext(appName="CollaborativeFilterRecSys")

train_file = sys.argv[1]
output_file = sys.argv[2]
cf_type = sys.argv[3]

model = []

#_,cf_type = sys.argv
#train_file = "train_review.json"

#---preprocess---
#((user_id, business_id), avg_stars)
reviews = sc.textFile(train_file).map(lambda x: json.loads(x))\
            .map(lambda x:(x['user_id'], x['business_id'], x['stars']))\
            .map(lambda x: ((x[0],x[1]), x[2])).groupByKey()\
            .mapValues(lambda x: sum(x)/len(x))
#(business_id, user_id)
b_u = reviews.map(lambda x: (x[0][1], x[0][0])).groupByKey().mapValues(set).collect()
b_u = dict(b_u)

#(business, (user_id, avg_stars))
b_u_r = reviews.map(lambda x: (x[0][1], (x[0][0], x[1]))).groupByKey().mapValues(dict)

#(user_id, (business_id, avg_stars))
u_b_r = reviews.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().mapValues(dict).collect()
u_b_r = dict(u_b_r)
    
#(business_id)
b_ = b_u_r.map(lambda x: x[0]).zipWithIndex().collect()
b_dict = dict(b_)

#(user_id, business_id)
u_b = reviews.map(lambda x:(x[0][0], b_dict[x[0][1]])).groupByKey().mapValues(set).collect()
u_b_dict = dict(u_b)
    
#--- functions --- 
def pearson_sim(x,dict_):
    #x = ((b1,b2),[u1,u2,u3])
    b1_stars = [dict_[x[0][0]][i] for i in x[1]]
    b2_stars = [dict_[x[0][1]][i] for i in x[1]]
    # average
    b1_avg = sum(b1_stars)/len(b1_stars)
    b2_avg = sum(b2_stars)/len(b2_stars)
    # norm
    b1_norm = [(i-b1_avg) for i in b1_stars]
    b2_norm = [(i-b2_avg) for i in b2_stars]
    # correlation
    member = sum([ (i*j) for i,j in zip(b1_norm, b2_norm)])
    denom = sqrt(sum([i*i for i in b1_norm])) * sqrt(sum([i*i for i in b2_norm]))
    if denom != 0:
        w = member / denom
    else: w = -1
        #if w > 0:
    return(x[0][0], x[0][1], w)

def sig_matrix(dict_):
    sig_matrixs = {}
    for line in dict_:
        for i in range(1,n+1):
            tmp = []
            for item in dict_[line]:
                hash_ = ((769*item + 12289) + i*((193*item + 3079)%389))%m
                tmp.append(hash_)
            sig_matrixs.setdefault(line,[]).append(min(tmp))
    return sig_matrixs           

def Jaccard_sim(x):
    and_ = u_b_dict[x[0]] & u_b_dict[x[1]]
    or_ = u_b_dict[x[0]] | u_b_dict[x[1]]
    member = len(and_)
    denom = len(or_)
    #3 co-business
    if member >= 3 and denom != 0:
        sim_ = member / denom
    #unbrounded..?
    else: sim_ = -1
    return(x[0], x[1],sim_, and_)

def find_bn(x):
    bn = [n_b[i] for i in x[3]]
    return ((x[0],x[1]),bn)

#--- "item_based"---
if cf_type == "item_based":
    can_tmp = reviews.map(lambda x: (x[0][0], x[0][1])).groupByKey().mapValues(set)\
                     .filter(lambda x: len(x[1]) > 1).map(lambda x: list(x[1]))
    candidate = can_tmp.map(lambda x: combinations(x,2)).map(lambda x: list(x))\
                       .flatMap(lambda x: x)
    # ((b1,b2),[u1,u2,u3])
    can_3coU = candidate.map(lambda x:(x,b_u[x[0]]&b_u[x[1]], len(b_u[x[0]]&b_u[x[1]])))\
                       .filter(lambda x: x[2]>=3).map(lambda x: (x[0], list(x[1])))
    
    # Pearson similarity
    # (b1, b2, sim)
    b_u_r_dict = dict(b_u_r.collect())
    true_can = can_3coU.map(lambda x: pearson_sim(x,b_u_r_dict)).distinct().collect()
    
    
    for item in true_can:
        model.extend([{"b1":item[0],"b2":item[1],"sim":item[2]}])
    
#--- "user_based" -- 
elif cf_type == "user_based":
    #parameters
    m = len(b_)
    b = 140
    r = 2
    n = b*r 
    a1,b1,a2,b2 = 769,12289,193,3079
    p = 389
    n_b = {}
    n_b = {v:k for k,v in b_dict.items()}
    #minhash
    minhassh_sig_matrixs = sig_matrix(u_b_dict)
    #LSH
    sig_matrixs = sc.parallelize(minhassh_sig_matrixs.items())
    #buckets = ((num,()),{})
    buckets = sig_matrixs.flatMap(lambda x:[((int(i/r), tuple(x[1][i:i+r])), x[0])for i in range(0,len(x[1]))])\
                        .groupByKey().mapValues(set)\
                        .filter(lambda x: len(x[1])>1)\
                        .map(lambda x: list(x[1]))
    #candidate = (u1,u2)
    candidate = buckets.map(lambda x: list(combinations(x,2))).flatMap(lambda x:x)

    #(u1,u2,sim,{b1,b2,b3})
    Jac_can = candidate.map(lambda x:Jaccard_sim(x)).filter(lambda x:x[2]>=0.01)
    #((u1,u2),[b1,b2,b3])
    Jac_can = Jac_can.map(find_bn)
    #(u1,u2,sim)
    Pea_can = Jac_can.map(lambda x:pearson_sim(x,u_b_r))\
                     .filter(lambda x: x[2]>0).distinct().collect()
    #print(Pea_can.take(5))
    for item in Pea_can:
        model.extend([{"u1":item[0],"u2":item[1],"sim":item[2]}])

with open(output_file, 'w') as f:
    for item in model:
        f.write(json.dumps(item)+'\n')

print("Duration: %s seconds." % (time.time()-start))
