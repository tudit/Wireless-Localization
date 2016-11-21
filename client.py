import sys;
from sklearn.metrics import accuracy_score;
from ast import literal_eval as make_tuple;
import requests;
import json;
import utils;
import random;

class CellData:
	def __init__(self ,centroid, feature_vectors):
		self.centroid = centroid;
		self.feature_vectors = feature_vectors;

#Radio Environment Map
def generate_rem(K, T, SD, transmitter_locs, seed):
	print("Generating Radio Environment Map.....");
	grid_data = []; # contains grid level data
	feature_vectors = []; # contains consolidated data
	labels = [];
	
	for i in range(0, utils.WIDTH, K):
		row_data = [];	
		for j in range(0, utils.WIDTH, K):
			centroid = ((i + K) / 2, (j + K) / 2); #centroid is a tuple (x,y)
			#generate feature vectors(20) at the centriod and append to features vector list
			feature_vectors_at_centroid = generate_feature_vectors(K, T, SD, centroid, transmitter_locs, feature_vectors, labels, seed);
			cell_data = CellData(centroid, feature_vectors_at_centroid);	
			row_data.append(cell_data);		
		grid_data.append(row_data);
	
	return grid_data, labels, feature_vectors;	


#Feature vectors are <RSSI1,RSSI2,RSSI3...RSSIk> for transmitters <T1,T2,T3...Tk>

def split_into_training_test_data(data, labels, K, seed):
	#random.seed(seed);
	data_shuffled = []
	labels_shuffled = []

	index_shuffled = [i for i in range(len(data))];
	random.shuffle(index_shuffled);
	for i in index_shuffled:
		data_shuffled.append(data[i]);
		labels_shuffled.append(labels[i]);

	test_size = int(0.1 * ((int(utils.WIDTH / K)) ** 2));
	train_size = len(data) - test_size;

	return data_shuffled[:train_size], data_shuffled[train_size:], labels_shuffled[:train_size], labels_shuffled[train_size:];

def generate_feature_vectors(K, T, SD, centroid, transmitter_locs, feature_vector, labels, seed):
	feature_vectors_at_centroid = [];
	for i in range(utils.FEATURES_PER_CELL):	
		features = [];
		for j in range(T):
			d = utils.eucledian_distance(centroid, transmitter_locs[j]);
			if d == 0:
				#centroid coincides with one of the transmitter, ignore that centroid
				return feature_vectors_at_centroid;
			pl_d = utils.generate_power_at_d(d, K, SD, seed);
			features.append(pl_d);
	
		feature_vectors_at_centroid.append(features);
		feature_vector.append(features);
		labels.append(centroid);
	
	return feature_vectors_at_centroid;


def predict_gaussian_naive_bayes(training_data, training_labels, test_data, test_labels, errors):
	print("Predicting GNB.....");
	predictor = GaussianNB();
	
	#convert training labels from integer tuples to strings
	training_labels_str = [str(label) for label in training_labels];
	#test_labels_str = [str(label) for label in test_labels]
	
	pred_labels_str = predictor.fit(training_data, training_labels_str).predict(test_data);
	pred_labels = [make_tuple(label) for label in pred_labels_str];
	
	#error = accuracy_score(pred_labels, test_labels_str, normalize = False);
	for i in range(len(pred_labels)):
		errors.append(utils.eucledian_distance(pred_labels[i], test_labels[i]));


#N transmitters
#K*K configuration of cells

def simulate_data(K, T, SD):
	
	training_data = [];
	training_labels = [];
	test_data = [];
	test_labels = [];
	
	grid_indices, transmitter_locs = utils.generate_transmitter_locations(K, T, utils.SEED);
	grid_data, labels, feature_vectors = generate_rem(K, T, SD, transmitter_locs, utils.SEED);
	training_data, test_data, training_labels, test_labels = split_into_training_test_data(feature_vectors, labels, K, utils.SEED);

	return training_data, training_labels, test_data, test_labels;

	
if __name__ == '__main__':
	
	if utils.K > utils.WIDTH:
		print("K should be less than 200" );

	training_data, training_labels, test_data, test_labels = simulate_data(utils.K, utils.T, utils.SD);
	print(len(training_data), len(training_labels), len(test_data), len(test_labels));	
	
	url = "http://localhost:8000/location";
	train_payload = {"feature_vectors" : training_data, "labels" : training_labels};
	train_res = requests.post(url, data = json.dumps(train_payload));
	
	test_payload = {"feature_vectors" : test_data,  "labels" : test_labels};
	test_res = requests.post(url, data = json.dumps(test_payload));
	
	print(train_res.text, test_res.text);
	print(train_res.status_code, test_res.status_code);
