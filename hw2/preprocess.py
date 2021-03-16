import json
from pyspark import SparkContext, SparkConf
import pandas as pd


review_file = 'review.json'
business_file = 'business.json'

conf = SparkConf()
sc = SparkContext('local', conf=conf)
review_plaintext = sc.textFile(review_file)
business_plaintext = sc.textFile(business_file)
review_json = review_plaintext.map(lambda x: json.loads(x))
business_json = business_plaintext.map(lambda x: json.loads(x))


def join_data(business_json, review_json):

	b_frame = business_json.map(lambda x: (x['business_id'], x['state'])) \
		.filter(lambda x: x[1] == 'NV')

	r_frame = review_json.map(lambda x: (x['business_id'], x['user_id']))

	return b_frame.join(r_frame).map(lambda x: (x[1][1], x[0])) \
		.collect()


out_json = join_data(business_json, review_json)
cols = ['user_id', 'business_id']
df = pd.DataFrame(out_json, columns=cols)
df.to_csv('data.csv', sep=',', index=False)