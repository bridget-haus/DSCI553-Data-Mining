import os
import math
import random
import itertools
import time


def get_data(file):

	points = {}
	with open(file) as f:

		for line in f:
			line = line.strip().split(',')
			index = line[0]
			dimension_values = [float(num) for num in line[1:]]
			points[index] = dimension_values

	num_dimensions = len(list(points.values())[0])
	return points, num_dimensions


def point_list_to_dict(point_list, points):

	if point_list is None:
		return {}

	point_dict = {}
	for point in point_list:
		dims = points[str(point)]
		point_dict[str(point)] = dims

	return point_dict


def get_euclidean_dist(point, centroid, num_dimensions):

	diff_squared = []
	for i in range(num_dimensions):
		diff_squared.append((point[i] - centroid[i])**2)

	euclidean_dist = math.sqrt(sum(diff_squared))

	return euclidean_dist


def kmeans_assignment(points, K, num_dimensions):

	centroids = {}
	for i in range(K):
		index, dim = random.choice(list(points.items()))
		centroids[index] = dim

	cluster_assignment = {}
	for p, point in points.items():
		min_dist_bool = True
		for c, centroid in centroids.items():
			euclidean_dist = get_euclidean_dist(point, centroid, num_dimensions)
			if min_dist_bool:
				min_dist = euclidean_dist
				min_dist_bool = False
				min_centroid = c
			if euclidean_dist < min_dist:
				min_dist = euclidean_dist
				min_centroid = c
		if min_centroid in cluster_assignment:
			cluster_assignment[min_centroid].append(p)
		else:
			cluster_assignment[min_centroid] = [p]

	return cluster_assignment


def DS_summarize(cluster_assignment, points, num_dimensions, DS):

	for centroid, cluster in cluster_assignment.items():
		N = len(cluster)
		SUM = []
		SUM_SQ = []
		for i in range(num_dimensions):
			dim_list = []
			dim_sq_list = []
			for point in cluster:
				dims = points[point]
				dim_list.append(dims[i])
				dim_sq_list.append(dims[i]**2)
			sum_dims = sum(dim_list)
			sum_sq_dims = sum(dim_sq_list)
			SUM.append(sum_dims)
			SUM_SQ.append(sum_sq_dims)
		if centroid not in DS:
			DS[centroid] = [N, SUM, SUM_SQ]
		else:
			DS[centroid][0] += N
			DS[centroid][1] += SUM
			DS[centroid][2] += SUM_SQ


	return DS


def CS_summarize(cluster_assignment, points, num_dimensions, CS, RS):

	for centroid, cluster in cluster_assignment.items():
		if centroid == 'NA':
			for point in cluster:
				RS[point] = points[point]
				#del CS[point]
		else:
			if len(cluster) > 1:
				N = len(cluster)
				SUM = []
				SUM_SQ = []
				for i in range(num_dimensions):
					dim_list = []
					dim_sq_list = []
					for point in cluster:
						if point in RS:
							dims = RS[point]
						else:
							dims = points[point]
						dim_list.append(dims[i])
						dim_sq_list.append(dims[i] ** 2)
					sum_dims = sum(dim_list)
					sum_sq_dims = sum(dim_sq_list)
					SUM.append(sum_dims)
					SUM_SQ.append(sum_sq_dims)
				for point in cluster:
					if point in RS:
						del RS[point]
				if centroid not in CS:
					CS[centroid] = [N, SUM, SUM_SQ]
				else:
					CS[centroid][0] += N
					CS[centroid][1] += SUM
					CS[centroid][2] += SUM_SQ
			else:
				if cluster[0] not in RS:
					RS[cluster[0]] = points[cluster[0]]
					#del CS[cluster[0]]

	return CS, RS


def get_standard_deviation(summary, num_dimensions):

	standard_dev = []
	N = summary[0]
	SUM = summary[1]
	SUMSQ = summary[2]
	for i in range(num_dimensions):
		variance = (SUMSQ[i]/N) - ((SUM[i]/N)**2)
		st_dev = math.sqrt(variance)
		standard_dev.append(st_dev)

	return standard_dev


def get_mahalanobis_distance(compare_points, DS, num_dimensions, centroids, CS_flag=False):

	threshold = 2 * math.sqrt(num_dimensions)

	cluster_assignment = {}
	for p, point_dims in compare_points.items():
		min_dist_bool = True
		min_centroid = 'NA'
		for c, summary in DS.items():
			standard_dev = get_standard_deviation(summary, num_dimensions)
			if CS_flag:
				N = summary[0]
				SUM = summary[1]
				centroid_dims = [SUM[i]/N for i in range(num_dimensions)]
			else:
				centroid_dims = centroids[c]
			sq_normalized_dist = []
			for i in range(num_dimensions):
				sq_norm_dist = ((point_dims[i] - centroid_dims[i]) / standard_dev[i])**2
				sq_normalized_dist.append(sq_norm_dist)
			mahalanobis_dist = math.sqrt(sum(sq_normalized_dist))
			if mahalanobis_dist < threshold:
				if min_dist_bool:
					min_dist = mahalanobis_dist
					min_dist_bool = False
					min_centroid = c
				if mahalanobis_dist < min_dist:
					min_dist = mahalanobis_dist
					min_centroid = c
		if min_centroid in cluster_assignment:
			cluster_assignment[min_centroid].append(p)
		else:
			cluster_assignment[min_centroid] = [p]

	return cluster_assignment


def merge_CS(CS, num_dimensions):

	threshold = 2 * math.sqrt(num_dimensions)
	CS_copy = CS.copy()

	for CS_1, CS_2 in itertools.combinations(CS.items(), 2):
		centroid_1 = CS_1[0]
		centroid_2 = CS_2[0]
		if centroid_1 in CS_copy:
			if centroid_2 in CS_copy:
				summary_1 = CS_1[1]
				summary_2 = CS_2[1]
				standard_dev_1 = get_standard_deviation(summary_1, num_dimensions)
				standard_dev_2 = get_standard_deviation(summary_2, num_dimensions)
				average_standard_dev = [(standard_dev_1[i] + standard_dev_2[i])/2 for i in range(num_dimensions)]
				N_1 = summary_1[0]
				SUM_1 = summary_1[1]
				CS_1 = [SUM_1[i] / N_1 for i in range(num_dimensions)]
				N_2 = summary_2[0]
				SUM_2 = summary_2[1]
				SUM_SQ_2 = summary_2[2]
				CS_2 = [SUM_2[i] / N_2 for i in range(num_dimensions)]
				sq_normalized_dist = []
				for i in range(num_dimensions):
					sq_norm_dist = ((CS_1[i] - CS_2[i]) / average_standard_dev[i]) ** 2
					sq_normalized_dist.append(sq_norm_dist)
				mahalanobis_dist = math.sqrt(sum(sq_normalized_dist))
				if mahalanobis_dist < threshold:
					CS_copy[centroid_1][0] += N_2
					CS_copy[centroid_1][1] += SUM_2
					CS_copy[centroid_1][2] += SUM_SQ_2
					del CS_copy[centroid_2]


	return CS_copy


def initialize_clusters(first_file, K, out_file1):

	points, num_dimensions = get_data(first_file)

	subset_size = math.ceil(.1 * len(points))
	subset_list = random.sample(range(len(points)), subset_size)
	points_subset = point_list_to_dict(subset_list, points)

	cluster_assignment = kmeans_assignment(points_subset, 5 * K, num_dimensions)
	outlier_threshold = .005 * subset_size
	outliers = {}
	inliers = {}
	for centroid, cluster in cluster_assignment.items():
		if len(cluster) < outlier_threshold:
			for point in cluster:
				outliers[point] = points_subset[point]
		else:
			for point in cluster:
				inliers[point] = points_subset[point]

	inlier_cluster_assignment = kmeans_assignment(inliers, K, num_dimensions)
	outlier_cluster_assignment = kmeans_assignment(outliers, 3*K, num_dimensions)

	DS = {}
	CS = {}
	RS = {}

	DS = DS_summarize(inlier_cluster_assignment, points_subset, num_dimensions, DS)
	centroids = {}
	for centroid in DS.keys():
		centroids[centroid] = points[centroid]

	CS, RS = CS_summarize(outlier_cluster_assignment, points_subset, num_dimensions, CS, RS)

	remaining_list = set(list(points.keys())) - set(list(points_subset.keys()))
	remaining_points = point_list_to_dict(remaining_list, points)

	DS_assignment = get_mahalanobis_distance(remaining_points, DS, num_dimensions, centroids)
	unassigned_list = DS_assignment.pop('NA', None)

	DS = DS_summarize(DS_assignment, points, num_dimensions, DS)
	DS_cluster_count = len(DS.keys())
	DS_point_count = sum([v[0] for v in DS.values()])

	unassigned_points = point_list_to_dict(unassigned_list, points)
	CS_assignment = get_mahalanobis_distance(unassigned_points, CS, num_dimensions, centroids, CS_flag=True)
	CS, RS = CS_summarize(CS_assignment, points, num_dimensions, CS, RS)
	if len(RS) > 0:
		RS_assignment = kmeans_assignment(RS, 3*K, num_dimensions)
		CS, RS = CS_summarize(RS_assignment, points, num_dimensions, CS, RS)
		# for centroid, points in RS_assignment.items():
		# 	CS_assignment[centroid] = points
		# CS_assignment.pop('NA', None)
	else:
		RS_assignment = {}

	RS_point_count = len(RS)

	CS = merge_CS(CS, num_dimensions)
	CS_cluster_count = len(CS.keys())
	CS_point_count = sum([v[0] for v in CS.values()])

	print(DS_point_count + CS_point_count + RS_point_count)

	with open(out_file1, "w") as f:
		header = "round_id,nof_cluster_discard,nof_point_discard,nof_cluster_compression,nof_point_compression,nof_point_retained"
		data = [1, DS_cluster_count, DS_point_count, CS_cluster_count, CS_point_count, RS_point_count]
		f.write(header + os.linesep)
		f.write(','.join(str(i) for i in data))
		f.write(os.linesep)

	for centroid in DS.keys():
		centroids[centroid] = points[centroid]

	return DS, CS, RS, num_dimensions, centroids, DS_assignment, CS_assignment, RS_assignment, inlier_cluster_assignment, outlier_cluster_assignment


def generate_clusters(file, K, out_file1, out_file2, DS, CS, RS, round_id, centroids):

	points, num_dimensions = get_data(file)
	compare_points = points

	DS_assignment = get_mahalanobis_distance(compare_points, DS, num_dimensions, centroids)
	unassigned_list = DS_assignment.pop('NA', None)

	DS = DS_summarize(DS_assignment, points, num_dimensions, DS)
	DS_cluster_count = len(DS.keys())
	DS_point_count = sum([v[0] for v in DS.values()])

	unassigned_points = point_list_to_dict(unassigned_list, points)
	CS_assignment = get_mahalanobis_distance(unassigned_points, CS, num_dimensions, centroids, CS_flag=True)
	CS, RS = CS_summarize(CS_assignment, points, num_dimensions, CS, RS)
	if len(RS) > 0:
		RS_assignment = kmeans_assignment(RS, 3 * K, num_dimensions)
		CS, RS = CS_summarize(RS_assignment, points, num_dimensions, CS, RS)
		# for centroid, points in RS_assignment.items():
		# 	CS_assignment[centroid] = points
		# CS_assignment.pop('NA', None)
	else:
		RS_assignment = {}

	RS_point_count = len(RS)

	CS = merge_CS(CS, num_dimensions)
	CS_cluster_count = len(CS.keys())
	CS_point_count = sum([v[0] for v in CS.values()])

	print(DS_point_count + CS_point_count + RS_point_count)

	with open(out_file1, "a") as f:
		data = [round_id, DS_cluster_count, DS_point_count, CS_cluster_count, CS_point_count, RS_point_count]
		f.write(','.join(str(i) for i in data))
		f.write(os.linesep)

	for centroid in DS.keys():
		if centroid not in centroids:
			centroids[centroid] = points[centroid]

	return DS, CS, RS, num_dimensions, centroids, DS_assignment, CS_assignment, RS_assignment


def main():

	start = time.time()
	input_path = 'test1'
	K = 10
	out_file1 = 'intermediate.txt'
	out_file2 = 'cluster_answer.txt'

	# final_DS = {}
	# final_CS = {}
	# final_RS = {}
	final_assignment = {}

	for i, filename in enumerate(os.listdir(input_path)):
		file = f'{input_path}/{filename}'
		if i == 0:
			DS, CS, RS, num_dimensions, centroids, DS_assignment, CS_assignment, RS_assignment, inlier_cluster_assignment, outlier_cluster_assignment = initialize_clusters(file, K, out_file1)
			for centroid, points in inlier_cluster_assignment.items():
				for point in points:
					final_assignment[point] = centroid
			for centroid, points in outlier_cluster_assignment.items():
				for point in points:
					final_assignment[point] = centroid
		else:
			DS, CS, RS, num_dimensions, centroids, DS_assignment, CS_assignment, RS_assignment = generate_clusters(file, K, out_file1, out_file2, DS, CS, RS, i+1, centroids)
		for centroid, points in CS_assignment.items():
			for point in points:
				final_assignment[point] = centroid
		for centroid, points in RS_assignment.items():
			for point in points:
				final_assignment[point] = centroid
		for centroid, points in DS_assignment.items():
			for point in points:
				final_assignment[point] = centroid

	print(set(list(final_assignment.values())))

	DS_copy = DS.copy()
	for CS_centroid, CS_summary in CS.items():
		min_dist_bool = True
		N = CS_summary[0]
		SUM = CS_summary[1]
		SUM_SQ = CS_summary[2]
		CS_dims = [SUM[i]/N for i in range(num_dimensions)]
		for DS_centroid, DS_summary in DS.items():
			DS_dims = centroids[DS_centroid]
			euclidean_dist = get_euclidean_dist(CS_dims, DS_dims, num_dimensions)
			if min_dist_bool:
				min_dist = euclidean_dist
				min_dist_bool = False
				min_centroid = DS_centroid
			if euclidean_dist < min_dist:
				min_dist = euclidean_dist
				min_centroid = DS_centroid
		DS_copy[min_centroid][0] += N
		DS_copy[min_centroid][1] += SUM
		DS_copy[min_centroid][2] += SUM_SQ

	end = time.time()
	duration = end - start
	print(f'Duration: {duration}')


if __name__ == "__main__":
	main()
