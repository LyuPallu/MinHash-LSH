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
test_file = sys.argv[2]
model_file = sys.argv[3]
output_file = sys.argv[4]
cf_type = sys.argv[5]

output = []
# _, cf_type = sys.argv
# train_file = "train_review.json"
# test_file = "test_review.json"
# model_file = "task3item.model"

# at most N neightbors
N = 9 # 3 5

#(user_id, business_id, avg_stars)
reviews = sc.textFile(train_file).map(lambda x: json.loads(x))\
            .map(lambda x:(x['user_id'], x['business_id'], x['stars']))\
            .map(lambda x: ((x[0],x[1]), x[2])).groupByKey()\
            .mapValues(lambda x: sum(x)/len(x))\
            .map(lambda x:(x[0][0],x[0][1],x[1]))
            
def wfile(x):
    with open(output_file, 'a') as f:
        for item in x:
            f.write(json.dumps(item) + '\n')

with open(output_file, 'w') as f:
    f.write('')

if cf_type == "item_based":
    #(user_id, (business_id, avg_stars))
    u_b_r = reviews.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(dict).collect()
    u_b_r_dict = dict(u_b_r)

    #(business, (user_id, avg_stars))
    b_u_r = reviews.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(dict).collect()
    b_u_r_dict = dict(b_u_r)

    # ---models---
    # ((b1,(b2,sim)),(b2,(b1,sim)))
    b_b_s = sc.textFile(model_file).map(lambda x:json.loads(x))\
                .flatMap(lambda x:((x['b1'],(x['b2'],x['sim'])),(x['b2'],(x['b1'],x['sim']))))\
                 .groupByKey().mapValues(dict)
    b_b_s_dict = dict(b_b_s.collect())
    #()
    b_list = b_b_s.mapValues(lambda x: list(x.items()))
    b_list_dict = dict(b_list.collect())
    
    # ---tests---
    test = sc.textFile(test_file).map(lambda x: json.loads(x))\
             .map(lambda x: (x['user_id'],x['business_id']))
    # ---candidate---
    can = test.filter(lambda x:x[0] in u_b_r_dict  
                           and x[1] in b_u_r_dict 
                           and x[1] in b_b_s_dict)
    def true_can_(x):
        u_, b_, u_r = x[0],x[1],u_b_r_dict[x[0]]
        b_star = []
        for item in u_r:
            if item in b_b_s_dict[b_]: b_star.append(item)
        b_star = [(i, b_b_s_dict[b_][i]) for i in b_star]
        b_star_n = sorted(b_star, key=lambda x: x[1], reverse=True)[:N] #8 3 5

        b_star_w = [i[1] for i in b_star_n]
        b_w = [i[0] for i in b_star_n]
        b_r = [u_b_r_dict[u_][i] for i in b_w]
        member = sum([i*j for i,j in zip(b_r, b_star_w)])
        denom = sum([abs(i) for i in b_star_w])
        if denom != 0:
            stars = member / denom
        else: return None
        return(u_,b_,stars)
    #(u_id, b_id, stars)
    true_can = can.map(lambda x: true_can_(x)).filter(bool)\
                .map(lambda x: {"user_id": x[0], "business_id": x[1], "stars": x[2]})\
                .repartition(4)\
                .foreach(wfile)

elif cf_type == "user_based":
    #(user_id, (business_id, avg_stars))
    u_b_r = reviews.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(dict).collect()
    u_b_r_dict = dict(u_b_r)

    #(business, (user_id, avg_stars))
    b_u_r = reviews.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(dict).collect()
    b_u_r_dict = dict(b_u_r)

    #(business_id, user_id)
    b_u = reviews.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set)
    b_u_dict = dict(b_u.collect())

    # ---models---
    # ((b1,(b2,sim)),(b2,(b1,sim)))
    u_u_s = sc.textFile(model_file).map(lambda x:json.loads(x))\
                .flatMap(lambda x:((x['u1'],(x['u2'],x['sim'])),(x['u2'],(x['u1'],x['sim']))))\
                .groupByKey().mapValues(dict)
    u_u_s_dict = dict(u_u_s.collect())
    #()
    u_list = u_u_s.mapValues(lambda x: list(x.items()))
    u_list_dict = dict(u_list.collect())

    # ---tests---
    test = sc.textFile(test_file).map(lambda x: json.loads(x))\
             .map(lambda x: (x['business_id'],x['user_id']))
    # ---candidate---
    can = test.filter(lambda x:x[1] in u_b_r_dict  
                           and x[0] in b_u_r_dict 
                           and x[1] in u_u_s_dict)

    def true_can_2(x):
        b = x[0]
        u = x[1]
        l = u_list_dict[u]
        l = [i for i in l if i[0] in b_u_r_dict[b]]
        
        nb =[i[0] for i in l]
        stars = [b_u_r_dict[b][i] for i in nb]

        w = [i[1] for i in l]
        x_stars = [u_b_r_dict[u][i] for i in u_b_r_dict[u]]
        nb_avgs =[]
        for item in nb:
            nb_stars =[u_b_r_dict[item][i] for i in u_b_r_dict[item] if i != b]
            nb_avg = sum(nb_stars)/len(nb_stars)
            nb_avgs.append(nb_avg)

        x_avg = sum(x_stars)/len(x_stars)

        norm =[i - j for i,j in zip(stars,nb_avgs)]
        fz = sum([i*j for i,j in zip(norm,w)])
        fm = sum([abs(i) for i in w])
        if fm !=0 and len(l)>=3:
            star = x_avg + fz/fm
        else:
            star = x_avg
        return (u,b,star)

    #(u_,b_,true_star)
    true_can = can.map(lambda x: true_can_2(x))\
                .filter(lambda x: x[2]>0)\
                .map(lambda x:(x[0],x[1],5.0) if x[2]>5 else (x[0],x[1],x[2]))\
                .map(lambda x: {"user_id": x[0], "business_id": x[1], "stars": x[2]})\
                .repartition(4)\
                .foreachPartition(wfile)
    #print(true_can.take(5))
    # output = []
    # for can in true_can.collect():
    #     output.extend([{"user_id": can[0], "business_id": can[1], "stars": can[2]}])

# with open(output_file, 'w') as f:
#     for item in output:
#         f.write(json.dumps(item)+'\n')

print("Duration: %s seconds." % (time.time()-start))      