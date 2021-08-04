import sys
import time
import json
import re
from pyspark import SparkContext
from operator import add
from collections import Counter
from math import sqrt

start = time.time()
sc = SparkContext(appName = "RecSys")

input_file = sys.argv[1]
model_file = sys.argv[2]
output_file = sys.argv[3]

lines = sc.textFile(input_file).map(lambda x: json.loads(x))
pairs = lines.map(lambda x: (x['user_id'], x['business_id']))

with open(model_file) as f:
    model = json.load(f)

b_profile = model['business_profile']
u_profile = model['user_profile']

#(u_id,b_id,sim)
def cos_(x):
    if x[0] not in u_profile:
        u_profile[x[0]] = [-1]
    if x[1] not in b_profile:
        b_profile[x[1]] = [-1]

    inter = len(set(u_profile[x[0]]).intersection(set(b_profile[x[1]])))
    sqrt_ = sqrt(len(u_profile[x[0]]))*sqrt(len(b_profile[x[1]]))
    sim = inter / sqrt_
    #if sim >= 0.01:
    return(x[0],x[1],sim)
preds = pairs.map(lambda x: cos_(x)).filter(lambda x:x[2]>= 0.01)\
            .collect()
            
results=[]
for item in preds:
    results.extend([{"user_id":item[0],"business_id":item[1],"sim":item[2]}])

with open(output_file, 'w') as f:
    for item in results:
        f.write(json.dumps(item) + '\n')
print("Duration: %s seconds." % (time.time()-start))