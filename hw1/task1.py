import sys
from pyspark import SparkContext, SparkConf
import json

input_file = sys.argv[1]
output_file = sys.argv[2]
stopwords = sys.argv[3]
year = sys.argv[4]
m = int(sys.argv[5])
n = int(sys.argv[6])

conf = SparkConf()
sc = SparkContext('local', conf=conf)
plain_text = sc.textFile(input_file)
json_data = plain_text.map(lambda x: json.loads(x))
punctuation_list = ['(', '[', ',', '.', '!', '?', ':', ';', ']', ')']


def count_reviews(plain_text):

	return plain_text.count()


def count_per_year(json_data, year):

	return json_data.map(lambda x: (x['date'])) \
		.map(lambda x: x[:4]) \
		.filter(lambda x: year in x) \
		.count()


def distinct_users(json_data):

	return json_data.map(lambda x: (x['user_id'])) \
		.distinct() \
		.count()


def top_users(json_data, m):

	return json_data.map(lambda x: (x['user_id'], 1)) \
		.reduceByKey(lambda x, y: x + y) \
		.sortBy(lambda x: x[1], False) \
		.take(m)


def clean_func(json_data):

	review = json_data['text'].lower()
	for p in punctuation_list:
		review = review.replace(p, '')
	return review


def top_words(json_data, punctuation_list, n):

	with open(stopwords) as f:
		stopword_list = []
		for word in f:
			word = word.strip()
			stopword_list.append(word)

	return json_data.map(clean_func) \
		.flatMap(lambda x: x.split()) \
		.map(lambda x: (x, 1)) \
		.reduceByKey(lambda x, y: x + y) \
		.filter(lambda x: x[0] not in stopword_list) \
		.sortBy(lambda x: x[1], False) \
		.take(n)


review_count = count_reviews(plain_text)
year_count = count_per_year(json_data, year)
user_count = distinct_users(json_data)
top_user_count = top_users(json_data, m)
top_word_tuple = top_words(json_data, punctuation_list, n)
top_word_count = [x[0] for x in top_word_tuple]

out_dict = {'A': review_count, 'B': year_count, 'C': user_count, 'D': top_user_count, 'E': top_word_count}

with open(output_file, 'w') as f:
	f.write(json.dumps(out_dict))
