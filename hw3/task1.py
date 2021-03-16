import sys
from pyspark import SparkContext, SparkConf
import time
import os
import random
import itertools
import json

input_file = sys.argv[1]
output_file = sys.argv[2]

conf = SparkConf()
sc = SparkContext('local', conf=conf)
plain_text = sc.textFile(input_file)
json_data = plain_text.map(lambda x: json.loads(x))


def create_matrix(json_data):

	user_index = json_data.map(lambda x: x['user_id']) \
		.distinct() \
		.zipWithIndex()
	business_index = json_data.map(lambda x: x['business_id']) \
		.distinct() \
		.zipWithIndex()
	business_user = json_data.map(lambda x: (x['user_id'], x['business_id']))
	business_user_join = business_user.join(user_index) \
		.map(lambda x: (x[1][0], x[1][1])) \
		.map(lambda x: (x[0], [x[1]])) \
		.reduceByKey(lambda x, y: x + y) \
		.map(lambda x: (x[0], sorted(set(x[1]))))
	matrix = business_user_join.join(business_index) \
		.map(lambda x: (x[1][1], x[1][0])) \
		.sortByKey()

	return matrix, user_index, business_index


def brute_force(matrix):

	LOG_EVERY_N = 1000000
	i = 0
	similar_businesses = []
	for a, b in itertools.combinations(matrix.collect(), 2):
		intersect = set(a[1]).intersection(set(b[1]))
		uni = set(a[1]).union(set(b[1]))
		jaccard = len(intersect)/len(uni)
		i += 1
		if i % LOG_EVERY_N == 0:
			print(f'logging {i}')
		if jaccard >= .05:
			business_pair = (a[0], b[0])
			similar_businesses.append(business_pair)

	return similar_businesses


def hash_func(matrix, user_count, a, b):

	signature_matrix = {}
	business_id = matrix[0]
	old_rows = matrix[1]
	for i in range(len(a)):
		signature = user_count + 1
		for old_row in old_rows:
			new_row = (a[i] * old_row + b[i]) % user_count
			if new_row < signature:
				signature = new_row
		if business_id in signature_matrix:
			signature_matrix[business_id].append(signature)
		else:
			signature_matrix[business_id] = [signature]

	for key, val in signature_matrix.items():
		signature_matrix = [key] + val

	return signature_matrix


def LSH(signature_matrix, b):

	num_hashes = len(signature_matrix[0]) - 1
	r = int(num_hashes/b)

	all_bands = []
	for i in range(0, b):
		band_hash = {}
		for business_sigs in signature_matrix:
			business_id = business_sigs[0]
			band = business_sigs[r*i+1:r*(i+1)+1]
			band = tuple(band)
			hash_val = hash(band)
			if hash_val in band_hash:
				band_hash[hash_val].append(business_id)
			else:
				band_hash[hash_val] = [business_id]
		all_bands.append(band_hash)

	candidates = []
	for band_num in all_bands:
		for k, v in band_num.items():
			if len(v) > 1:
				candidates.append(v)

	return candidates


def check_jaccard(matrix, candidates, business_index):

	similar_businesses = []
	checker = []
	for pair in candidates:
		first_business = pair[0]
		second_business = pair[1]
		intersect = set(matrix[first_business][1]).intersection(set(matrix[second_business][1]))
		uni = set(matrix[first_business][1]).union(set(matrix[second_business][1]))
		jaccard = len(intersect) / len(uni)
		if jaccard >= .05:
			checker.append(pair)
			sim_dict = {}
			sim_dict['b1'] = business_index[first_business][0]
			sim_dict['b2'] = business_index[second_business][0]
			sim_dict['sim'] = jaccard
			similar_businesses.append(sim_dict)

	return similar_businesses, checker


start = time.time()

matrix, user_index, business_index = create_matrix(json_data)
user_count = user_index.count()

num_hashes = 500
num_bands = 250

a = list(range(2000, 9000))
random.shuffle(a)
a = a[:num_hashes]
b = list(range(1, 1999))
random.shuffle(b)
b = b[:num_hashes]

signature_matrix = matrix.map(lambda x: hash_func(x, user_count, a, b)).collect()
candidates = LSH(signature_matrix, num_bands)

final_candidates = []
for pair in candidates:
	combos = list(itertools.combinations(pair, 2))
	final_candidates.append(combos)

flat_list = [item for sublist in final_candidates for item in sublist]
final_set_candidates = list(set([tuple(sorted(t)) for t in flat_list]))
similar_businesses, checker = check_jaccard(matrix.collect(), final_set_candidates, business_index.collect())

with open(output_file, 'w') as f:

	for sim_dic in similar_businesses:
		f.write(json.dumps(sim_dic) + os.linesep)

end = time.time()
duration = end - start
print(f'Duration: {duration}')