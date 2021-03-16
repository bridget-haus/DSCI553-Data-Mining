import sys
from pyspark import SparkContext, SparkConf
from pyspark.mllib.fpm import FPGrowth
import time
import os
import ast

user_filter = int(sys.argv[1])
s = int(sys.argv[2])
input_file = sys.argv[3]
output_file = sys.argv[4]

conf = SparkConf()
sc = SparkContext('local', conf=conf)
plain_text = sc.textFile(input_file)


def construct_baskets(plain_text, user_filter, s):

	case = plain_text.map(lambda x: x.split(',')) \
		.filter(lambda x: x[0] != 'user_id') \
		.map(lambda x: (x[0], [x[1]])) \
		.reduceByKey(lambda x, y: x + y) \
		.map(lambda x: set(x[1])) \
		.filter(lambda x: len(x) > user_filter) \
		.map(lambda x: sorted(x)) \
		.cache()

	return case

start = time.time()
case = construct_baskets(plain_text, user_filter, s)
num_baskets = case.count()
s = s/num_baskets
model = FPGrowth.train(case, minSupport=s, numPartitions=1)
result = model.freqItemsets().collect()

frequent_itemsets = []
for items, frequency in result:
	frequent_itemsets.append(tuple(items))

frequent_itemsets = list(set([tuple(sorted(t)) for t in frequent_itemsets]))
frequent_itemsets = sorted(frequent_itemsets, key=len)

with open('out2.txt', 'r') as f:
	task2 = []
	for i in f:
		i = ast.literal_eval(i)
		task2.append(i)


intersection = set(map(frozenset, task2)) & set(map(frozenset, frequent_itemsets))
len_task2 = len(task2)
len_task3 = len(frequent_itemsets)
len_intersection = len(intersection)

with open(output_file, 'w') as f:
	f.write(f'Task2,{len_task2}' + os.linesep)
	f.write(f'Task3,{len_task3}' + os.linesep)
	f.write(f'Intersection,{len_intersection}' + os.linesep)

end = time.time()
duration = end - start
print(f'Duration: {duration}')