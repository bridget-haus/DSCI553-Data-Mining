from pyspark import SparkContext, SparkConf
import time
import json
import math
import os
import sys

test_file = sys.argv[1]
model_file = sys.argv[2]
output_file = sys.argv[3]

conf = SparkConf()
sc = SparkContext('local', conf=conf)
plain_text = sc.textFile(test_file)
json_data = plain_text.map(lambda x: json.loads(x)).collect()


def cosine_sim(json_data, business_boolean_matrix, user_boolean_matrix):

	cosine_sim_dict = {}
	for pair in json_data:
		dot_prod = 0
		user_id = pair['user_id']
		business_id = pair['business_id']
		if business_id in business_boolean_matrix:
			business_vector = business_boolean_matrix[business_id]
			business_dist = len(business_vector)
			if user_id in user_boolean_matrix:
				user_vector = user_boolean_matrix[user_id]
				user_dist = list(user_boolean_matrix[user_id].values())
				user_dist = sum(map(lambda i: i * i, user_dist))
				for column in business_vector:
					if str(column) in user_vector.keys():
						dot_prod += user_vector[str(column)]
			else:
				user_dist = 0
		else:
			business_dist = 0
			user_dist = 0
		try:
			cosine_distance = dot_prod / math.sqrt(business_dist * user_dist)
		except ZeroDivisionError:
			cosine_distance = 0
		if cosine_distance >= 0.01:
			cosine_sim_dict[(user_id, business_id)] = cosine_distance

	return cosine_sim_dict


start = time.time()

with open(model_file, 'r') as f:

	separator = '/'
	model = f.read()
	model = model.split(separator)
	business_boolean_matrix = json.loads(model[0])
	user_boolean_matrix = json.loads(model[1])

cosine_sim_dict = cosine_sim(json_data, business_boolean_matrix, user_boolean_matrix)

sim_list = []
for ub, sim in cosine_sim_dict.items():

	sim_dict = {}
	user_id = ub[0]
	business_id = ub[1]
	sim_dict['user_id'] = user_id
	sim_dict['business_id'] = business_id
	sim_dict['sim'] = sim
	sim_list.append(sim_dict)


with open(output_file, 'w') as f:

	for dic in sim_list:
		f.write(json.dumps(dic) + os.linesep)

end = time.time()
duration = end - start
print(f'Duration: {duration}')
