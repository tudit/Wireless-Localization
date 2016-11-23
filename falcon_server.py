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

num_cells = int(utils.WIDTH / utils.K) ** 2;
TRNG_SIZE = int(0.8 * num_cells * utils.FEATURES_PER_CELL);
BATCH_SIZE = 100;

def get_locations():
	locations = dict();
	cols  = int(utils.WIDTH / utils.K);
	loc_count = 0;
	for i in range(0, utils.WIDTH, utils.K):
		for j in range(0, utils.WIDTH, utils.K):
			centroid = ((i + utils.K) / 2, (j + utils.K) / 2);
			locations[loc_count] = centroid;
			loc_count += 1; 	
	return locations;

def get_init_means():
	mean_init = [];
	grid_indices, transmitter_locs = utils.generate_transmitter_locations(utils.K, utils.T, utils.SEED);
	centroids = get_locations();
	for i in range(len(centroids)):
		mean_cell = [];
		for j in range(utils.T):
			d = utils.eucledian_distance(transmitter_locs[j], centroids[i]);
			if d == 0:
				mean_cell.append(utils.POWER_T);
			else:	 
				mean_cell.append(utils.generate_power_at_d(d, utils.K, 0, utils.SEED));
		
		mean_init.append(mean_cell);
	print(len(mean_init));
	return mean_init;

class LocationData(object):
	client = pymongo.MongoClient();
	db = client.test;
	count_trng = 0;
	locations = get_locations();
	predictor = mixture.GaussianMixture(n_components = int(utils.WIDTH / utils.K) * int(utils.WIDTH / utils.K), covariance_type = 'full',\
				 warm_start = True, means_init = get_init_means());
	gmm = None;

	def get_location(self, loc):
		return self.locations[loc];
	
	def get_features_from_db(self, i):
		return self.db.data.find_one({ "_id": i});

	def store_features_in_db(self, i, feature_vectors, gmm, count_trng):
		self.db.data.insert({"_id" : i, "feature_vectors" : feature_vectors, "gmm" : gmm,\
			"trng_count" : trng_count});
		
	
	def predict(self, test_data, test_labels, res):
		print("Testing");
		errors = [];
		pred_classes = [];
		pred_locs = [];
		
		for start_index in range(0, len(test_data), BATCH_SIZE):
			end_index = start_index + BATCH_SIZE;
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
			end_index = start_index + BATCH_SIZE;
			tr_batch = tr_data[start_index : end_index];
			#tr_labels = data["labels"];
			self.gmm = self.predictor.fit(tr_batch);
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

data = LocationData();
api.add_route('/location', data);