import sys
import time
import json
import re
from pyspark import SparkContext
from operator import add
from collections import Counter
from math import log2

start = time.time()
sc = SparkContext(appName = "RecSys")

input_file = sys.argv[1]
output_file = sys.argv[2]
stopwords_file = sys.argv[3]

output = {}

#---preprocess---
stopwords = sc.textFile(stopwords_file)
stopwords = stopwords.flatMap(lambda x:x.split()).collect()

lines = sc.textFile(input_file).map(lambda x: json.loads(x))
users = lines.map(lambda x: (x['user_id'], x['business_id'])).groupByKey().mapValues(list)

def filter_(x):
    b_id = x["business_id"]
    text = x["text"].lower().replace('\n', ' ')
    text = re.sub(r'[^a-z]'," ",text).split()
    #text = [i for i in text if i not in stopwords]
    return (b_id, text)

reviews = lines.map(filter_).reduceByKey(lambda x,y: x+y)\
                .mapValues(lambda x:list(Counter(i for i in x if i not in stopwords).items()))

n_doc = reviews.count()
words = reviews.flatMapValues(lambda x:x)
#for each word
n_words = words.map(lambda x:x[1]).reduceByKey(add)

#total words count
total_n_words = n_words.map(lambda x: x[1]).sum()

#---remove race words---
race_word = n_words.filter(lambda x:x[1] < 0.000001 * total_n_words)\
                    .map(lambda x: x[0]).collect()
racewords = sc.parallelize(race_word).map(lambda x: (x,(1,1)))
words = words.map(lambda x: (x[1][0], (x[0], x[1][1]))).subtractByKey(racewords)\
                .map(lambda x: (x[1][0], (x[0], x[1][1])))
words_idf = words.map(lambda x: (x[1][0],x[0]))\
                .groupByKey().mapValues(len)\
                .map(lambda x: (x[0],log2(n_doc / x[1])))\
                .collect()
words_idf = dict(words_idf)

#--- max # of word in docs: dict
max_words =  words.map(lambda x: (x[1][0],[x[0],x[1][1]])).groupByKey().mapValues(list)\
                    .map(lambda x: (x[0], max([i[1] for i in x[1]])))\
                    .collect()
max_words_dict = dict(max_words)
#frequent words : 
# (id, (key, (freq/max_words_dict[key]) * words_idf[key]))
tmp_freq = words.map(lambda x: (x[0], (x[1][0], (x[1][1] / words_idf[x[1][0]])* max_words_dict[x[1][0]])))\
                .groupByKey().mapValues(list)

#---select top 200 ---
def select200(x):
    res = x[0]
    sort_200 = sorted(x[1],key = lambda x: x[1], reverse=True)[:200]
    return (res, sort_200)
tmp_freq = tmp_freq.map(select200)

freq_ = tmp_freq.flatMap(lambda x: x[1]).distinct().zipWithIndex().collect()
freq_dict = dict(freq_)

#---business profile---
b_profile = tmp_freq.map(lambda x: (x[0], [freq_dict[i] for i in x[1]])).collect()
b_dict = dict(b_profile)
output['business_profile'] = b_dict

#---user profile---
def build_u_vec(x):
    #x[1]
    list_ = []
    for i in x[1]:
        list_ = list_ + b_dict[i]
    return(x[0], list(set(list_)))
u_profile = users.map(lambda x: build_u_vec(x)).collect()
output['user_profile'] = dict(u_profile)

#---model---
with open(output_file,'w') as f:
    json.dump(output, f)


print("Duration: %s seconds." % (time.time()-start))