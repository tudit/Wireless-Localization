import falcon;
import pymongo;
import json;
from sklearn.naive_bayes import GaussianNB;
from sklearn import mixture;
from ast import literal_eval as make_tuple;
import utils;
import os;
import pickle;

api = application = falcon.API();

K = 40;
T = 5;
TRNG_SIZE = 1000;
BATCH_SIZE = 100;

def get_locations():
	locations = dict();
	cols  = int(200 / K);
	loc_count = 0;
	for i in range(0, 200, K):
		for j in range(0, 200, K):
			centroid = ((i + K) / 2, (j + K) / 2);
			locations[loc_count] = centroid;
			loc_count += 1; 	
	return locations;

def get_init_means():
	mean_init = [];
	grid_indices, transmitter_locs = utils.generate_transmitter_locations(K, T, utils.SEED);
	centroids = get_locations();
	for i in range(len(centroids)):
		mean_cell = [];
		for j in range(T):
			d = utils.eucledian_distance(transmitter_locs[j], centroids[i]);
			if d == 0:
				mean_cell.append(utils.POWER_T);
			else:	 
				mean_cell.append(utils.generate_power_at_d(d, K, 0, utils.SEED));
		
		mean_init.append(mean_cell);
	print(len(mean_init));
	return mean_init;

class LocationData(object):
	client = pymongo.MongoClient();
	db = client.test;
	count_trng = 0;
	locations = get_locations();
	predictor = mixture.GaussianMixture(n_components = int(200 / K) * int(200 / K), covariance_type = 'full',\
				 warm_start = True, means_init = get_init_means());
	gmm = None;

	def get_location(self, loc):
		return self.locations[loc];
	
	# def predict_GMM(self, training_data, test_data, ncomp, covar_type):
	# 	print("Predicting using GMM.....");
	# 	predictor = mixture.GaussianMixture(n_components = ncomp, covariance_type = covar_type);
	# 	gmm = predictor.fit(training_data);
	# 	pred_labels = gmm.predict(test_data);
	# 	print(pred_labels)
	# 	print(gmm.means_)
	# 	print(gmm.covariances_);
	# 	return pred_labels.tolist(); #since it returns np array which is no json serializable

	# def predict_gaussian_naive_bayes(self, training_data, training_labels, test_data, test_labels, errors):
	# 	print("Predicting using GNB.....");
	# 	predictor = GaussianNB();
		
	# 	#convert training labels from integer tuples to strings
	# 	training_labels_str = [str(label) for label in training_labels];
	# 	#test_labels_str = [str(label) for label in test_labels]
		
	# 	pred_labels_str = predictor.fit(training_data, training_labels_str).predict(test_data);
	# 	pred_labels = [make_tuple(label) for label in pred_labels_str];
		
	# 	#error = accuracy_score(pred_labels, test_labels_str, normalize = False);
	# 	for i in range(len(pred_labels)):
	# 		errors.append(utils.eucledian_distance(pred_labels[i], test_labels[i]));

	# 	return pred_labels;	
	
	def get_features_from_db(self, i):
		return self.db.data.find_one({ "_id": i});

	def store_features_in_db(self, i, feature_vectors, gmm, count_trng):
		self.db.data.insert({"_id" : i, "feature_vectors" : feature_vectors, "gmm" : gmm,\
			"trng_count" : trng_count});
		
	# def on_get(self,req,res):	
	# 	i = req.get_param("_id");
	# 	test_data = req.get_param_as_list("test_data[]");
	# 	test_labels = req.get_param_as_list("test_labels[]");
	# 	print(len(test_data), len(test_labels), i);
	# 	data = self.get_features_from_db(i);
	# 	print(data);
	# 	training_data = data["feature_vectors"];
	# 	training_labels = data["labels"];
	# 	errors = [];
	# 	print(len(training_data),len(training_labels));
	# 	#self.predict_gaussian_naive_bayes(training_data, training_labels, test_data, test_labels, errors);
	# 	pred_vals = self.predict_GMM(training_data, test_data, 9, 'full');
	# 	#avg_error = float(sum(errors))/len(errors);
	# 	#response = {"avg_error" : avg_error};
	# 	response = {"predictions" : pred_vals};
	# 	res.body = json.dumps(response);
	# 	res.status = falcon.HTTP_200;
	
	def predict(self, test_data, test_labels, res):
		print("Testing");
		errors = [];
		pred_classes = [];
		pred_locs = [];
		
		for start_index in range(0, len(test_data), 100):
			end_index = start_index + 100;
			test_batch = test_data[start_index : end_index];
			pred_classes.extend(self.gmm.predict(test_batch).tolist());
	
		for i in range(len(pred_classes)):
			loc = self.get_location(pred_classes[i]);
			pred_locs.append(loc);
			errors.append(utils.eucledian_distance(loc, test_labels[i]));
		
		avg_error = float(sum(errors))/len(errors);
		response = {"predictions" : pred_locs, "avg_error" : avg_error};
		res.body = json.dumps(response);
	
	def train(self, tr_data):
		tr_data_count = len(tr_data);
		
		#Already trained count_trng data
		#Remaining TRNG_SIZE - count_trng data is trained in batches
		##of size BATCH_SIZE
		
		end_trng = min(tr_data_count, TRNG_SIZE - self.count_trng);
		for start_index in range(0, end_trng, BATCH_SIZE):
			end_index = start_index + 100;
			tr_batch = tr_data[start_index : end_index];
			#tr_labels = data["labels"];
			self.gmm = self.predictor.fit(tr_batch);
			#TODO: Store into DB
			#store_features_in_db(1, tr_batch, deepcopy.copy(self.gmm));
			self.count_trng += len(tr_batch);		
		
		with open("gmm.pkl", "wb") as fid:
				trng_md = {"gmm" : self.gmm, "trng_count" : self.count_trng}
				pickle.dump(trng_md, fid);

	def on_post(self, req, res):
		raw_json = req.stream.read();
		data = json.loads(str(raw_json, encoding = "utf-8"));
		
		#TODO: get data from DB and initialize in case of a failure
		#db_data = get_features_from_db(1);
		#gmm = db_data["gmm"];
		
		#if there is training data in db and training count is 0 (server restarted)
		#if gmm != None and self.count_trng == 0:
		# self.gmm = gmm;
		if self.count_trng == 0 and os.path.isfile('gmm.pkl'):
			with open("gmm.pkl", "rb") as fid:
				trng_md = pickle.load(fid);
				self.gmm = trng_md["gmm"];
				self.count_trng = trng_md["trng_count"];

		#if training size hasn't been reached, model requires more training
		if self.count_trng < TRNG_SIZE:
			print("Training");
			#train in batches
			tr_data = data["feature_vectors"];
			tr_labels = data["labels"];
			self.train(tr_data);

			# TRNG_SIZE - self.count_trng is how more training is required
			# if tr_data has more data than required for training, the difference 
			##has to be used for prediction
			if len(tr_data) > TRNG_SIZE - self.count_trng:
				start_index = TRNG_SIZE - self.count_trng;
				self.predict(tr_data[start_index:], tr_labels[start_index:], res);			
		
		else:
			# model has been sufficiently trained, now we can start prediction
			test_data = data["feature_vectors"];
			test_labels = data["labels"];
			self.predict(test_data, test_labels, res);
		
		res.status = falcon.HTTP_200;

	# def on_post_old(self, req, res):
	# 	raw_json = req.stream.read();
	# 	data = json.loads(str(raw_json, encoding = "utf-8"));
	# 	if data["predict"] == 0:
	# 		self.store_features_in_db(data["_id"], data["feature_vectors"], data["labels"]);
	# 		res.status = falcon.HTTP_201;
		
	# 	elif data["predict"] == 1:
	# 		errors = [];
	# 		test_data = data["feature_vectors"];
	# 		test_labels = data["labels"];
			
	# 		i = data["_id"];
	# 		db_data = self.get_features_from_db(i);
	# 		training_data = db_data["feature_vectors"];
	# 		training_labels = db_data["labels"];
			
	# 		#predictions = self.predict_gaussian_naive_bayes(training_data, training_labels, test_data, test_labels, errors);
	# 		predictions = self.predict_GMM(training_data, test_data, 9, 'full');
	# 		#avg_error = float(sum(errors))/len(errors);
	# 		#response = {"avg_error" : avg_error, "predictions" : predictions};
	# 		response = {"predictions" : predictions};
	# 		res.body = json.dumps(response);
	# 		res.status = falcon.HTTP_200;				
		
	# 	else:
	# 		res.status = falcon.HTTP_400;	

data = LocationData();
api.add_route('/location', data);