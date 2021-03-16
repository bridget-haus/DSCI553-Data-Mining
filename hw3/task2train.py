from pyspark import SparkContext, SparkConf
import time
import json
from string import digits
from heapq import heapify, heappush, heappop
import math
import sys

input_file = sys.argv[1]
model_file = sys.argv[2]
stopwords = sys.argv[3]

conf = SparkConf()
sc = SparkContext('local', conf=conf)
plain_text = sc.textFile(input_file)
json_data = plain_text.map(lambda x: json.loads(x))
punctuation_list = ['(', '[', ',', '.', '!', '?', ':', ';', ']', ')', '\\n', '\n', '#', '-', '\\', '"', '*']


def clean_func(review):

	review = review.lower().strip()
	for p in punctuation_list:
		review = review.replace(p, ' ')

	return review


with open(stopwords, 'r') as f:

	stopwords = set()
	for line in f:
		line = line.strip()
		stopwords.add(line)

stopwords.add('')


def concat_reviews(json_data):

	return json_data.map(lambda x: (x['business_id'], x['text'])) \
		.reduceByKey(lambda x, y: x + y) \
		.map(lambda x: (x[0], (clean_func(x[1]))))


def user_business_index(json_data):

	return json_data.map(lambda x: (x['user_id'], x['business_id'])) \
		.map(lambda x: (x[0], [x[1]])) \
		.reduceByKey(lambda x, y: x + y)


def clean_and_count(documents, stopwords):

	global_dict = {}
	max_freq = {}

	for document in documents:
		max_word_count = 0
		business_id = document[0]
		review_text = document[1]
		remove_digits = str.maketrans('', '', digits)
		review_text = review_text.translate(remove_digits)
		review_text = review_text.split(' ')
		for word in review_text:
			if len(word) >= 1:
				if word not in stopwords:
					if word in global_dict:
						if business_id in global_dict[word]:
							global_dict[word][business_id] += 1
							word_count = global_dict[word][business_id]
							if word_count > max_word_count:
								max_word_count = word_count
						else:
							global_dict[word][business_id] = 1
							word_count = global_dict[word][business_id]
							if word_count > max_word_count:
								max_word_count = word_count
					else:
						global_dict[word] = {business_id: 1}

		max_freq[business_id] = max_word_count

	return global_dict, max_freq


def business_tf_idf(global_dict, max_freq, total_docs):

	top_200_business = {}
	for word, doc_count in global_dict.items():
		idf = math.log2(total_docs/len(doc_count))
		for business, count in doc_count.items():
			tf = count/max_freq[business]
			tf_idf = tf * idf
			tf_tuple = (tf_idf, word)
			if business in top_200_business:
				heap = top_200_business[business]
				if len(heap) < 200:
					heappush(heap, tf_tuple)
				else:
					min_val = heap[0][0]
					if tf_idf > min_val:
						heappop(heap)
						heappush(heap, tf_tuple)
			else:
				heap = []
				heapify(heap)
				heappush(heap, tf_tuple)
				top_200_business[business] = heap

	return top_200_business


def boolean_matrix_business(top_200_business):

	business_boolean_matrix = {}
	distinct_words = {}
	counter = 0
	for business, top_words in top_200_business.items():
		for top_word in top_words:
			word = top_word[1]
			if word not in distinct_words:
				distinct_words[word] = counter
				counter += 1
			if business in business_boolean_matrix:
				business_boolean_matrix[business].append(distinct_words[word])
			else:
				business_boolean_matrix[business] = [distinct_words[word]]

	return distinct_words, business_boolean_matrix


def boolean_matrix_user(business_boolean_matrix, ub_index):


	user_boolean_matrix = {}

	for ub in ub_index:
		user = ub[0]
		businesses = ub[1]
		user_vector = []
		total_val = len(businesses)

		for business in businesses:
			current_vector = business_boolean_matrix[business]
			user_vector += current_vector

		dup_dict = {}
		final_user_vector = {}

		for val in user_vector:
			if val in dup_dict:
				dup_dict[val] += 1
			else:
				dup_dict[val] = 1

		for k, v in dup_dict.items():
			final_user_vector[k] = round(v/total_val, 3)

		user_boolean_matrix[user] = final_user_vector

	return user_boolean_matrix


start = time.time()
ub_index = user_business_index(json_data).collect()
total_docs = concat_reviews(json_data).count()
documents = concat_reviews(json_data).collect()
global_dict, max_freq = clean_and_count(documents, stopwords)
top_200_business = business_tf_idf(global_dict, max_freq, total_docs)
distinct_words, business_boolean_matrix = boolean_matrix_business(top_200_business)
user_boolean_matrix = boolean_matrix_user(business_boolean_matrix, ub_index)

with open(model_file, "w") as f:

	separator = '/'
	f.write(json.dumps(business_boolean_matrix))
	f.write(separator)
	f.write(json.dumps(user_boolean_matrix))


end = time.time()
duration = end - start
print(f'Duration: {duration}')