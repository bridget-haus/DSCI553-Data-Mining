from pyspark import SparkConf, SparkContext
import time
import os
import itertools
import sys
from queue import Queue


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


# noinspection PySetFunctionToLiteral
def construct_graph(edge_list):

	graph = {}
	for edge in edge_list:
		node1 = edge[0]
		node2 = edge[1]
		if node1 in graph:
			graph[node1].add(node2)
		else:
			graph[node1] = set([node2])
		if node2 in graph:
			graph[node2].add(node1)
		else:
			graph[node2] = set([node1])

	return graph


def shortest_path_count(graph, root):

	num_paths = {root: 1}
	level_dict = {0: [root]}
	visited = []
	visit_queue = Queue()
	parents = {root: [root]}
	visit_queue.put(root)

	count = 1

	cur_level = set()
	all_levels = [root]

	while not visit_queue.empty():
		current_node = visit_queue.get()
		edges = graph[current_node]

		visited.append(current_node)
		for edge in edges:
			if edge not in visited and edge not in list(visit_queue.queue):
				visit_queue.put(edge)
			if edge not in all_levels:
				if edge in parents:
					parents[edge].append(current_node)
				else:
					parents[edge] = [current_node]
				if edge in num_paths:
					num_paths[edge] += num_paths[current_node]
				else:
					num_paths[edge] = num_paths[current_node]

		cur_q = visit_queue.queue
		if len(cur_level & set(cur_q)) == 0:
			cur_level = set(cur_q)
			for level in cur_level:
				all_levels.append(level)

			level_dict[count] = list(cur_level)
			count += 1

	return num_paths, parents, level_dict


def edge_scores(num_paths, parents, level_dict):

	edge_dict = {}
	par_edges = {}
	for i in reversed(range(len(level_dict.keys()))):
		if not bool(par_edges):
			for node in level_dict[i]:
				par_edges[node] = []
		for node in level_dict[i]:
			for parent in parents[node]:
				if node != parent:
					sorted_node_parent = tuple(sorted(tuple((node, parent))))
					if parent in par_edges:
						par_edges[parent].append(sorted_node_parent)
					else:
						par_edges[parent] = [sorted_node_parent]

					child_factor = 0
					if node in par_edges:
						for child_edge in par_edges[node]:
							if child_edge in edge_dict:
								child_factor += edge_dict[child_edge]
							else:
								child_factor += 0

					edge_dict[sorted_node_parent] = (num_paths[parent] / num_paths[node]) * (1 + child_factor)

	return edge_dict


def calc_betweenness(nodes, graph):

	betweenness = {}
	communities = {}
	i = 1
	for node in nodes:
		num_paths, parents, level_dict = shortest_path_count(graph, node[0])
		if set(num_paths.keys()) not in communities.values():
			communities[i] = set(num_paths.keys())
			i += 1
		edge_dict = edge_scores(num_paths, parents, level_dict)
		for edge, score in edge_dict.items():
			if edge in betweenness:
				betweenness[edge] += score
			else:
				betweenness[edge] = score

	for k, val in betweenness.items():
		betweenness[k] = (val/2)

	betweenness = {k: v for k, v in sorted(betweenness.items(), key=lambda x: (-x[1], x[0]))}

	return betweenness, communities


def modularity(original_graph, graph, num_edges, communities):

#ki and kj come from new network
#Aij comes from old network

	Q = 0
	for community_num, community in communities.items():
		for i in community:
			k1 = graph[i]
			for j in community:
				k2 = graph[j]
				if j in original_graph[i]:
					Aij = 1
				else:
					Aij = 0
				q = Aij - ((len(k1) * len(k2)) / (2 * num_edges))
				Q += q

	Q = (1 / (2 * num_edges)) * Q

	return Q


def main():

	filter_threshold = int(sys.argv[1])
	input_file = sys.argv[2]
	betweenness_output_file = sys.argv[3]
	community_output_file = sys.argv[4]

	conf = SparkConf().setMaster("local")
	sc = SparkContext(conf=conf)
	plain_text = sc.textFile(input_file)

	start = time.time()
	ub_index = get_businesses(plain_text)
	edge_list, nodes = create_nodes_edges(ub_index, filter_threshold)
	print(f'Nodes: {len(nodes)}')
	print(f'Edges: {len(edge_list)}')

	num_edges = len(edge_list)/2
	graph = construct_graph(edge_list)
	original_graph = graph

	betweenness, communities = calc_betweenness(nodes, graph)

	with open(betweenness_output_file, 'w') as f:

		for edge, between_score in betweenness.items():
			f.write(str(edge) + ', ' + str(between_score) + os.linesep)

	print(len(communities))
	Q = modularity(original_graph, graph, num_edges, communities)
	print(Q)

	max_Q = 0
	final_communities = communities
	while Q > 0:
		edge = list(betweenness.keys())[0]
		print(edge)
		between = list(betweenness.values())[0]
		print(between)
		node1 = edge[0]
		node2 = edge[1]
		graph[node1].remove(node2)
		graph[node2].remove(node1)
		betweenness, communities = calc_betweenness(nodes, graph)
		print(len(communities))
		Q = modularity(original_graph, graph, num_edges, communities)
		print(Q)
		if Q > max_Q:
			max_Q = Q
			final_communities = communities

	final_communities = [sorted(x) for x in final_communities.values()]
	final_communities = sorted(final_communities)
	final_communities = sorted(final_communities, key=len)

	with open(community_output_file, 'w') as f:

		for community in final_communities:
			f.write(str(community).replace('[', '').replace(']', '') + os.linesep)

	end = time.time()
	duration = end - start
	print(f'Duration: {duration}')


if __name__ == "__main__":
	main()