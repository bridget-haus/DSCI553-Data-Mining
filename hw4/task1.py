from pyspark import SparkConf,SparkContext,SQLContext
import time
import os
import itertools
import sys
from graphframes import *


def get_businesses(plain_text):

	ub_index = plain_text.map(lambda x: x.split(',')) \
		.filter(lambda x: x[0] != 'user_id') \
		.map(lambda x: (x[0], [x[1]])) \
		.reduceByKey(lambda x, y: x + y) \
		.map(lambda x: (x[0], sorted(set(x[1])))) \
		.collect()

	return ub_index


def create_nodes_edges(ub_index, threshold):

	edge_list = []
	nodes = set()
	for u1, u2 in itertools.combinations(ub_index, 2):
		intersect = set(u1[1]).intersection(set(u2[1]))
		if len(intersect) >= threshold:
			edge_list.append((u1[0], u2[0]))
			edge_list.append((u2[0], u1[0]))
			nodes.add((u1[0],))
			nodes.add((u2[0],))

	return edge_list, nodes


def clean_func(x):

	punctuation_list = ['(', ')', "'", '"']

	x = x.replace(' label=', '').replace('Row(id=', '')
	for p in punctuation_list:
		x = x.replace(p, '')

	return x


def create_graphframe(nodes, edge_list, sqlContext):

	graph_nodes = sqlContext.createDataFrame(nodes, ["id"])
	graph_edges = sqlContext.createDataFrame(edge_list, ["src", "dst"])
	g = GraphFrame(graph_nodes, graph_edges)
	label_prop = g.labelPropagation(maxIter=5)

	return label_prop


def create_communities(label_prop):

	communities = label_prop.rdd.map(str) \
		.map(clean_func) \
		.map(lambda x: x.split(',')) \
		.map(lambda x: (x[1], [x[0]])) \
		.reduceByKey(lambda x, y: x + y) \
		.collect()

	return communities


def main():

	conf = SparkConf().setMaster("local")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)
	os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

	filter_threshold = int(sys.argv[1])
	input_file = sys.argv[2]
	output_file = sys.argv[3]

	plain_text = sc.textFile(input_file)

	start = time.time()
	ub_index = get_businesses(plain_text)
	edge_list, nodes = create_nodes_edges(ub_index, filter_threshold)
	print(f'Nodes: {len(nodes)}')
	print(f'Edges: {len(edge_list)}')


	end = time.time()
	duration = end - start
	print(f'Duration: {duration}')

	start = time.time()
	label_prop = create_graphframe(nodes, edge_list, sqlContext)
	communities = create_communities(label_prop)
	communities = [sorted(x[1]) for x in communities]
	communities = sorted(communities)
	communities = sorted(communities, key=len)

	with open(output_file, 'w') as f:

		for community in communities:
			f.write(str(community).replace('[', '').replace(']', '') + os.linesep)

	end = time.time()
	duration = end - start
	print(f'Duration: {duration}')


if __name__ == "__main__":
	main()
