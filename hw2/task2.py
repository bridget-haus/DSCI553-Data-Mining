import sys
from pyspark import SparkContext, SparkConf
import time
import itertools
import os
from math import ceil

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


def pass_one(partition, s, num_baskets):

	partition = list(partition)
	candidates = []

	freq_singles_dict = {}
	for basket in partition:
		for item in basket:
			if item in freq_singles_dict.keys():
				freq_singles_dict[item] += 1
			else:
				freq_singles_dict[item] = 1

	freq_singles_list = []
	for k, v in freq_singles_dict.items():
		if v >= ceil(s * (len(partition)/num_baskets)):
			freq_singles_list.append(k)
			candidates.append((k,))

	for k in range(2, 100):
		k_sets = {}
		for basket in partition:
			if len(basket) >= k:
				intersection = sorted(set(basket) & set(freq_singles_list))
				for item in itertools.combinations(intersection, k):
					item = tuple(item)
					if item in k_sets.keys():
						k_sets[item] += 1
					else:
						k_sets[item] = 1

		k_candidates = []
		for k_set, c in k_sets.items():
			if c >= ceil(s * (len(partition)/num_baskets)):
				k_candidates.append(k_set)

		candidates.append(k_candidates)

		freq_singles_list = set()
		for item in k_candidates:
			freq_singles_list = freq_singles_list.union(set(item))

	return candidates


def pass_two(partition, candidates):

	partition = list(partition)

	final_dict = {}
	for basket in partition:
		for candidate in candidates:
			if type(candidate) == str:
				candidate = (candidate,)
			if set(candidate).issubset(basket):
				if candidate in final_dict.keys():
					final_dict[candidate] += 1
				else:
					final_dict[candidate] = 1

	final_results = []
	for k, v in final_dict.items():
		final_results.append((k, v))

	return final_results


start = time.time()
case = construct_baskets(plain_text, user_filter, s)
num_baskets = case.count()

candidates = case.mapPartitions(lambda part: pass_one(part, s, num_baskets)) \
	.flatMap(lambda x: x) \
	.distinct() \
	.collect()

frequent_itemsets = case.mapPartitions(lambda partition: pass_two(partition, candidates)) \
	.reduceByKey(lambda x, y: x + y) \
	.filter(lambda x: x[1] >= s) \
	.map(lambda x: x[0]) \
	.collect()

for idx, cand in enumerate(candidates):
	if type(cand) == str:
		cand = (cand,)
		candidates[idx] = cand

candidates = list(set([tuple(sorted(t)) for t in candidates]))
candidates = sorted(candidates, key=len)
frequent_itemsets = list(set([tuple(sorted(t)) for t in frequent_itemsets]))
frequent_itemsets = sorted(frequent_itemsets, key=len)

with open(output_file, 'w') as f:
	f.write('Candidates:' + os.linesep)
	candidates = [sorted(list(g)) for k, g in itertools.groupby(candidates, key=len)]
	for cand in candidates:
		if len(cand[0]) == 1:
			cand = '[' + ', '.join("('{}')".format(t[0]) for t in cand) + ']'
		cand = str(cand)[1:-1]
		cand = cand.replace(', (', ',(').replace(') ,', '),')
		f.write(cand + os.linesep + os.linesep)
	f.write('Frequent Itemsets:' + os.linesep)
	frequent_itemsets = [sorted(list(g)) for k, g in itertools.groupby(frequent_itemsets, key=len)]
	for freq in frequent_itemsets:
		if len(freq[0]) == 1:
			freq = '[' + ', '.join("('{}')".format(t[0]) for t in freq) + ']'
		freq = str(freq)[1:-1]
		f.write(freq.replace(', (', ',(').replace(') ,', '),') + os.linesep + os.linesep)

end = time.time()
duration = end - start
print(f'Duration: {duration}')

with open('out2.txt', 'w') as f:
	for freq in frequent_itemsets:
		for item in freq:
			item = str(item)
			f.write(item + os.linesep)
