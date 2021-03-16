import sys
from pyspark import SparkContext, SparkConf
import json

input_file = sys.argv[1]
output_file = sys.argv[2]
partition_type = sys.argv[3]
n_partitions = int(sys.argv[4])
n = int(sys.argv[5])

conf = SparkConf()
sc = SparkContext('local', conf=conf)


def customized_n_reviews(input_file, n_partitions):

	r = sc.textFile(input_file) \
		.map(lambda x: json.loads(x)) \
		.map(lambda x: (x['business_id'], 1)) \
		.partitionBy(n_partitions)

	return r, r.getNumPartitions(), r.glom().map(len).collect()


def default_n_reviews(input_file):

	r = sc.textFile(input_file) \
		.map(lambda x: json.loads(x)) \
		.map(lambda x: (x['business_id'], 1))

	return r, r.getNumPartitions(), r.glom().map(len).collect()


if partition_type == 'default':
	r, num_partitions, num_items = default_n_reviews(input_file)


if partition_type == 'customized':
	r, num_partitions, num_items = customized_n_reviews(input_file, n_partitions)


result = r.reduceByKey(lambda x, y: x + y) \
		.filter(lambda x: x[1] >= n) \
		.collect()
result_list = [list(i) for i in result]
out_dict = {'n_partitions': num_partitions, 'n_items': num_items, 'result': result_list}
print(out_dict)

with open(output_file, 'w') as f:
	f.write(json.dumps(out_dict))