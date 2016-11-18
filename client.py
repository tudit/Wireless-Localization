import sys;
import math;
from sklearn.naive_bayes import GaussianNB;
from sklearn.metrics import accuracy_score;
import random;
from ast import literal_eval as make_tuple;
import requests;
import json;

POWER_T = 10 * math.log10(16 * 0.001);# Tramistted power mW to dBm 
ALPHA = 2.5;

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
	
	for i in range(0, 200, K):
		row_data = [];	
		for j in range(0, 200, K):
			centroid = ((i + K)/2, (j + K)/2); #centroid is a tuple (x,y)
			#generate feature vectors(20) at the centriod and append to features vector list
			feature_vectors_at_centroid = generate_feature_vectors(K, T, SD, centroid, transmitter_locs, feature_vectors, labels, seed);
			cell_data = CellData(centroid, feature_vectors_at_centroid);	
			row_data.append(cell_data);		
		grid_data.append(row_data);
	
	return grid_data, labels, feature_vectors;	



def generate_transmitter_locations(K, T, seed):
	print("Generating Transmitter Locations.....");
	#random.seed(seed);
	cells_per_row = int(200 / K);
	cells_per_col = cells_per_row;

	# generating random grid indices
	grid_row = random.sample(range(cells_per_row), T); 
	grid_col = random.sample(range(cells_per_col), T);
	grid_loc_transmitters = [(grid_row[i], grid_col[i]) for i in range(len(grid_row))];

	#computing actual x, y cordinates
	centroid_x = [get_centroid(row_index, K) for row_index in grid_row ]; 
	centroid_y = [get_centroid(col_index, K) for col_index in grid_col ];
	transmitters = [(centroid_x[i], centroid_y[i]) for i in range(len(centroid_x))];

	return grid_loc_transmitters, transmitters;

#get actual X and Y co-ordinates given grid indices
def get_centroid(grid_index, K):
	return ((grid_index * K) + (K / 2.0));

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

	test_size = int(0.1 * ((int(200 / K)) ** 2));
	train_size = len(data) - test_size;

	return data_shuffled[:train_size], data_shuffled[train_size:], labels_shuffled[:train_size], labels_shuffled[train_size:];

def generate_feature_vectors(K, T, SD, centroid, transmitter_locs, feature_vector, labels, seed):
	feature_vectors_at_centroid = [];
	for i in range(20):	
		features = [];
		for j in range(T):
			d = eucledian_distance(centroid, transmitter_locs[j]);
			if d == 0:
				#centroid coincides with one of the transmitter, ignore that centroid
				return feature_vectors_at_centroid;
			pl_d = generate_power_at_d(d, K, SD, seed);
			features.append(pl_d);
	
		feature_vectors_at_centroid.append(features);
		feature_vector.append(features);
		labels.append(centroid);
	
	return feature_vectors_at_centroid;

def generate_power_at_d(d, K, SD, seed):
	#random.seed(seed);
	noise = random.gauss(0, SD);
	pl = 10 * ALPHA * math.log10(d) + noise;
	pr = POWER_T - pl;		
	return pr;

def predict_gaussian_naive_bayes(training_data, training_labels, test_data, test_labels, errors):
	print("Predicting.....");
	predictor = GaussianNB();
	
	#convert training labels from integer tuples to strings
	training_labels_str = [str(label) for label in training_labels];
	#test_labels_str = [str(label) for label in test_labels]
	
	pred_labels_str = predictor.fit(training_data, training_labels_str).predict(test_data);
	pred_labels = [make_tuple(label) for label in pred_labels_str];
	
	#error = accuracy_score(pred_labels, test_labels_str, normalize = False);
	for i in range(len(pred_labels)):
		errors.append(eucledian_distance(pred_labels[i], test_labels[i]));
		
	
def eucledian_distance(v1,v2):
	if len(v1) != len(v2):
		print("***Both vectors are not of equal length!!***");
	square_differences = [(v1[i] - v2[i]) ** 2 for i in range(len(v1))];
	return math.sqrt(sum(square_differences));


#N transmitters
#K*K configuration of cells

def simulate_data(K, T, SD):
	
	training_data = [];
	training_labels = [];
	test_data = [];
	test_labels = [];
	
	for i in range(10):
		grid_indices, transmitter_locs = generate_transmitter_locations(K, T, i);
		#print(transmitter_locs, grid_indices);

		grid_data, labels, feature_vectors = generate_rem(K, T, SD, transmitter_locs, i);

		training_data_i, test_data_i, training_labels_i, test_labels_i = split_into_training_test_data(feature_vectors, labels, K, i);
		#print(training_data, test_data, training_labels, test_labels);
		#predict_gaussian_naive_bayes(training_data, training_labels, test_data, test_labels, errors);
		training_data.extend(training_data_i);
		training_labels.extend(training_labels_i);
		test_data.extend(test_data_i);
		test_labels.extend(test_labels_i);	
	return training_data, training_labels, test_data, test_labels;

	
if __name__ == '__main__':
	
	K = int(sys.argv[1]); #K is grid resolution
	if K > 200:
		print("K should be less than 200" );

	T = int(sys.argv[2]); #N is number of cell towers
	SD = float(sys.argv[3]) #SD is shadowing variance
	training_data, training_labels, test_data, test_labels = simulate_data(K, T, SD);
	print(len(training_data), len(training_labels), len(test_data), len(test_labels));	
	
	url = "http://localhost:8000/location";
	train_payload = {"predict" : 0, "_id" : 1, "feature_vectors" : training_data, "labels" : training_labels};
	train_res = requests.post(url, data = json.dumps(train_payload));
	
	test_payload = {"predict" : 1, "_id" : 1, "feature_vectors" : test_data,  "labels" : test_labels};
	test_res = requests.post(url, data = json.dumps(test_payload));
	
	print(train_res.text, test_res.text);
	print(train_res.status_code, test_res.status_code);
