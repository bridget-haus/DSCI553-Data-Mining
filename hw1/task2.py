import sys
from pyspark import SparkContext, SparkConf
import json

review_file = sys.argv[1]
business_file = sys.argv[2]
output_file = sys.argv[3]
if_spark = sys.argv[4]
n = int(sys.argv[5])

conf = SparkConf()
sc = SparkContext('local', conf=conf)
review_plaintext = sc.textFile(review_file)
business_plaintext = sc.textFile(business_file)


def spark_avg_stars(review_plaintext, business_plaintext, n):

	review_json = review_plaintext.map(lambda x: json.loads(x))
	business_json = business_plaintext.map(lambda x: json.loads(x))

	b_frame = business_json.map(lambda x: (x['business_id'], x['categories'])) \
		.filter(lambda x: x[1] is not None) \
		.map(lambda x: (x[0], x[1].split(', '))) \
		.flatMap(lambda x: [(x[0], category) for category in x[1]])
	r_frame = review_json.map(lambda x: (x['business_id'], x['stars'])) \
		.filter(lambda x: x[1] is not None)
	return b_frame.join(r_frame).map(lambda x: (x[1][0], x[1][1])) \
		.aggregateByKey((0, 0), lambda U, x: (U[0] + x, U[1] + 1), lambda U, V: (U[0] + V[0], U[1] + V[1])) \
		.mapValues(lambda xy: round((float(xy[0]) / xy[1]), 1)) \
		.sortByKey() \
		.sortBy(lambda x: x[1], False) \
		.take(n)


def python_avg_stars(review_file, business_file, n):
	with open(review_file) as f:
		review_dict = {}
		for line in f:
			line = json.loads(line)
			if line['business_id'] in review_dict.keys():
				review_dict[line['business_id']].append(line['stars'])
			else:
				review_dict[line['business_id']] = [line['stars']]

	with open(business_file) as f:
		business_dict = {}
		for line in f:
			line = json.loads(line)
			categories = line['categories']
			if categories != None:
				business_dict[line['business_id']] = categories.split(', ')

	join_dict = {}
	for business_id, categories in business_dict.items():
		if business_id in review_dict.keys():
			for category in categories:
				if category in join_dict.keys():
					join_dict[category].append(review_dict[business_id])
				else:
					join_dict[category] = [review_dict[business_id]]

	for category, stars in join_dict.items():
		join_dict[category] = round(float(sum(sum(x) for x in stars) / sum(len(x) for x in stars)), 1)

	return sorted(join_dict.items(), key=lambda x: (-x[1], x[0]))[:n]


if if_spark == 'spark':
	spark_star_rating = spark_avg_stars(review_plaintext, business_plaintext, n)
	out_dict = {'result': spark_star_rating}

elif if_spark == 'no_spark':
	python_star_rating = python_avg_stars(review_file, business_file, n)
	out_dict = {'result': python_star_rating}

with open(output_file, 'w') as f:
	f.write(json.dumps(out_dict))