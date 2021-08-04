import sys
import time
import json
from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict
import math
import random
import itertools

start = time.time()
sc = SparkContext(appName="minihash_LSH")

input_file = sys.argv[1]
output_file = sys.argv[2]

#--preprocess---
lines = sc.textFile(input_file).map(lambda x: json.loads(x))

users = lines.map(lambda x: x['user_id'])
u_numbers = users.distinct().zipWithIndex().collectAsMap()

#(u_id, num)
business = lines.map(lambda x: (x['business_id'], u_numbers[x['user_id']])).distinct()
matrix = business.map(lambda x: (x[0], list([x[1]])))\
                    .reduceByKey(lambda x,y:x+y )\
                    .map(lambda x: (x[0],sorted(x[1])))
matrixs = matrix.collectAsMap()
print(matrix.take(5))

# parameter
m = len(u_numbers)
n = 50 #160 180 30 
band = 50
row = 1
p = 1
paramaters = []
for i in range(0,n):
    a1 = random.randint(1,pow(2,20))
    b1 = random.randint(1,pow(2,20))
    a2 = random.randint(1,pow(2,20))
    b2 = random.randint(1,pow(2,20))
    paramaters.append((a1,b1,a2,b2))

# step 1: signatures
def hash_funtion(x_):
    res = []
    #hash funtion
    for (a1,b1,a2,b2) in paramaters:
        funs = []
        for i in x_:
            hf = ((a1 * i + b1)+(a2 * i + b2))%m
            funs.append(hf)
        res.append(min(funs))
    return res

#(business_id,[numnumnum num])
signature_matrix = matrix.mapValues(lambda x: hash_funtion(x))
                        
# step 2: divide into bands 
def hash_(x):
    res = []
    for i in range(0,len(x[1])):
        tmp = []
        tmp.append(x[1][i])
        if i%row == 0:
            res.append(((hash(tuple(tmp)), i),[x[0]]))
    return res
#(b1,b2)
pairs = signature_matrix.flatMap(hash_)\
                .reduceByKey(lambda a,b: a + b).filter(lambda x : len(x[1]) > 1)\
                .flatMap(lambda x : list(itertools.combinations(x[1],2)))\
                .map(lambda x: tuple(sorted(list(x)))).distinct()

#Jaccard
def Jaccard(x):
    b1,b2 = x[0],x[1]
    u1,u2 = matrixs[b1], matrixs[b2]
    intersection_ = set(u1) & set(u2)
    union_ = set(u1) | set(u2)
    sim = len(intersection_) / len(union_)
    return((b1,b2),sim)
candidates = pairs.map(lambda x: Jaccard(x)).filter(lambda x:x[1]>=0.055)
#print(candidates.take(5))
with open(output_file, "w") as f:
    for x in candidates.collect():
        res = json.dumps({"b1": x[0][0], "b2": x[0][1], "sim": x[1]})
        f.write(res + '\n')
        
print("Duration: % second." % (time.time() - start))