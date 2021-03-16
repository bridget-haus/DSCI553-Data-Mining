import sys
from pyspark import SparkContext, SparkConf
import time
import itertools
import os

case_number = int(sys.argv[1])
s = int(sys.argv[2])
input_file = sys.argv[3]
output_file = sys.argv[4]

conf = SparkConf()
sc = SparkContext('local', conf=conf)
plain_text = sc.textFile(input_file)
n_buckets = 30


def construct_baskets(plain_text):

	case_1 = plain_text.map(lambda x: x.split(',')) \
		.filter(lambda x: x[0] != 'user_id') \
		.map(lambda x: (x[0], [x[1]])) \
		.reduceByKey(lambda x, y: x + y) \
		.persist()

	case_2 = plain_text.map(lambda x: x.split(',')) \
		.filter(lambda x: x[1] != 'business_id') \
		.map(lambda x: (x[1], [x[0]])) \
		.reduceByKey(lambda x, y: x + y) \
		.persist()

	return case_1, case_2


def frequent_singletons(case, s):

	return case.flatMap(lambda x: x[1]) \
		.map(lambda x: (x, 1)) \
		.reduceByKey(lambda x, y: x + y) \
		.filter(lambda x: x[1] >= s) \
		.map(lambda x: x[0]) \
		.collect()


def create_pairs(case):

	case_rdd = sc.parallelize(case)

	baskets = case_rdd.map(lambda x: [str(i) for i in x[1]]) \
		.map(lambda x: sorted(x)) \
		.collect()

	pairs = []
	for basket in baskets:
		if len(basket) >= 2:
			pair = list(itertools.combinations(basket, 2))
			pairs.append(pair)

	return pairs


def create_candidates(case, freq_priors, k):

	final_candidates = []

	flat_list = set([item for sublist in freq_priors for item in sublist])
	for basket in case:
		intersection = list(flat_list & set(basket[1]))
		if len(basket[1]) >= k:
			k_set = list(itertools.combinations(intersection, k))
			final_candidates.append(sorted(k_set))

	return final_candidates


def hash_func(k_set, n_buckets):

	k_set = [int(x) for x in k_set]
	sum_set = sum(k_set)

	bucket = sum_set % n_buckets
	return bucket


def hash_table(baskets, n_buckets, s):

	hash_dict = {}
	bucket_tbl = []
	for basket in baskets:
		for k_set in basket:
			bucket = hash_func(k_set, n_buckets)
			bucket_tbl.append(bucket)
			if bucket in hash_dict.keys():
				hash_dict[bucket].append(k_set)
			else:
				hash_dict[bucket] = [k_set]

	bitmap = []
	for i in range(n_buckets):
		bucket_count = bucket_tbl.count(i)
		if bucket_count >= s:
			bitmap.append([i, 1])
		else:
			bitmap.append([i, 0])

	return hash_dict, bitmap


def check_candidates(freq_singles, hash_dict, bitmap):

	checked_candidates = set()
	for bucket in bitmap:
		if bucket[1] == 1:
			pairs = hash_dict[bucket[0]]
			for pair in pairs:
				if set(pair).issubset(freq_singles):
					checked_candidates.add(pair)

	return checked_candidates


def pass_two(case, candidates, s):

	final_list = []
	baskets = case.map(lambda x: x[1]).collect()
	for candidate in candidates:
		for basket in baskets:
			if set(candidate).issubset(basket):
				final_list.append(candidate)

	final_list = sc.parallelize(final_list)
	return final_list.map(lambda x: (x, 1)) \
		.reduceByKey(lambda x, y: x + y) \
		.filter(lambda x: x[1] >= s) \
		.map(lambda x: x[0]) \
		.collect()


start = time.time()
candidates = set()
case_1, case_2 = construct_baskets(plain_text)
if case_number == 1:
	case = case_1
if case_number == 2:
	case = case_2
max_k = max(case.map(lambda x: len(x[1])).collect())
freq_singles = frequent_singletons(case, s)
for freq in freq_singles:
	candidates.add((freq,))
pair_candidates = create_pairs(case.collect())
hash_dict, bitmap = hash_table(pair_candidates, n_buckets, s)
checked_candidates = check_candidates(freq_singles, hash_dict, bitmap)
for pair in checked_candidates:
	candidates.add(pair)
for k in range(3, max_k+1):
	k_sets = create_candidates(case.collect(), checked_candidates, k)
	hash_dict, bitmap = hash_table(k_sets, n_buckets, s)
	checked_candidates = check_candidates(freq_singles, hash_dict, bitmap)
	for cand in checked_candidates:
		candidates.add(cand)

frequent_itemsets = pass_two(case, candidates, s)
candidates = list(set([tuple(sorted(t)) for t in candidates]))
frequent_itemsets = list(set([tuple(sorted(t)) for t in frequent_itemsets]))
candidates = sorted(candidates, key=len)
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
	f.write(os.linesep + 'Frequent Itemsets:' + os.linesep)
	frequent_itemsets = [sorted(list(g)) for k, g in itertools.groupby(frequent_itemsets, key=len)]
	for freq in frequent_itemsets:
		if len(freq[0]) == 1:
			freq = '[' + ', '.join("('{}')".format(t[0]) for t in freq) + ']'
		freq = str(freq)[1:-1]
		f.write(freq.replace(', (', ',(').replace(') ,', '),') + os.linesep + os.linesep)

end = time.time()
duration = end - start
print(f'Duration: {duration}')