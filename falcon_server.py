import falcon;
import pymongo;
import json;
from sklearn.naive_bayes import GaussianNB;
from sklearn import mixture;
from ast import literal_eval as make_tuple;
import utils;
import os;
import pickle;
import numpy as np;
import localization as lz;

api = application = falcon.API();


num_cells = int(utils.WIDTH_PER_BLOCK / utils.K) ** 2;
#TRNG_SIZE = int(0.8 * num_cells * utils.FEATURES_PER_CELL);
TRNG_SIZE = 500;
BATCH_SIZE = TRNG_SIZE;
transmitter_block_indices, transmitter_locs = utils.generate_transmitter_locations(int(utils.WIDTH / utils.BLOCKS), utils.T, utils.SEED);

def get_grid_indices(blocks):
	grid_indices = [];
	for block in blocks:
		grid_index = [int((i + 0.5) * utils.CELLS_PER_BLOCK) for i in block];
		grid_indices.append(grid_index);
	return grid_indices;	

transmitter_grid_indices = get_grid_indices(transmitter_block_indices);
print(transmitter_grid_indices, transmitter_locs);

def get_locations(x_block, y_block):
	x_offset = x_block * utils.WIDTH_PER_BLOCK;
	y_offset = y_block * utils.WIDTH_PER_BLOCK;

	locations = dict();
	cols  = int(utils.WIDTH_PER_BLOCK / utils.K);
	loc_count = 0;
	for i in range(x_offset, x_offset + utils.WIDTH_PER_BLOCK, utils.K):
		for j in range(y_offset, y_offset + utils.WIDTH_PER_BLOCK, utils.K):
			centroid = ((i + utils.K) / 2, (j + utils.K) / 2);
			locations[loc_count] = centroid;
			loc_count += 1; 	
	return locations;

def get_init_means(x_block, y_block):
	mean_init = [];
	centroids = get_locations(x_block, y_block);
	for i in range(len(centroids)):
		mean_cell = [];
		for j in range(utils.T):
			d = utils.eucledian_distance(transmitter_locs[j], centroids[i]);
			if d == 0:
				mean_cell.append(utils.POWER_T);
			else:	 
				mean_cell.append(utils.generate_power_at_d(d, utils.K, 0, utils.SEED));
		
		mean_init.append(mean_cell);
	return mean_init;

class LocationData(object):
	client = pymongo.MongoClient();
	db = client.test;
	counts_trng = [0 for i in range(utils.BLOCKS ** 2)];
	
	locations = dict();
	for i in range(utils.BLOCKS):
		for j in range(utils.BLOCKS):
			locations[i * utils.BLOCKS + j] = get_locations(i, j);

	predictors = dict();
	n_classes = int(utils.WIDTH_PER_BLOCK / utils.K) ** 2;
	for i in range(utils.BLOCKS):
		predictor_row = [];
		for j in range(utils.BLOCKS):
			predictor = mixture.GaussianMixture(n_components = n_classes, covariance_type = 'full',\
				 warm_start = True, means_init = get_init_means(i, j));
			predictors[i * utils.BLOCKS + j] = predictor;

	gmm = dict();

	def get_location(self, block, loc):
		return self.locations[block][loc];
	
	def get_features_from_db(self, i):
		return self.db.data.find_one({ "_id": i});

	def store_features_in_db(self, i, feature_vectors, gmm, count_trng):
		self.db.data.insert({"_id" : i, "feature_vectors" : feature_vectors, "gmm" : gmm,\
			"trng_count" : trng_count});
		
	
	def predict(self, block, test_data, test_labels, res, errors, pred_locs):
		print("Testing (Block : %d, Size: %d)" %(block, len(test_data)));
		pred_classes = [];

		for start_index in range(0, len(test_data), BATCH_SIZE):
			end_index = start_index + BATCH_SIZE;
			test_batch = test_data[start_index : end_index];
			pred_classes.extend(self.gmm[block].predict(test_batch).tolist());
	
		for i in range(len(pred_classes)):
			loc = self.get_location(block, pred_classes[i]);
			pred_locs.append(loc);
			errors.append(utils.eucledian_distance(loc, test_labels[i]));
	
	def train(self, block, tr_data):
		print("Training (Block: %d, Size: %d)" %(block, len(tr_data)));
		tr_data_count = len(tr_data);
		
		#Already trained count_trng data
		#Remaining TRNG_SIZE - count_trng data is trained in batches
		##of size BATCH_SIZE
		
		end_trng = min(tr_data_count, TRNG_SIZE - self.counts_trng[block]);
		for start_index in range(0, end_trng, BATCH_SIZE):
			end_index = start_index + BATCH_SIZE;
			tr_batch = tr_data[start_index : end_index];
			#tr_labels = data["labels"];
			self.gmm[block] = self.predictors[block].fit(tr_batch);
			self.counts_trng[block] += len(tr_batch);		
		
		pkl_file = "gmm.pkl" + str(block);
		with open(pkl_file, "wb") as fid:
				trng_md = {"gmm" : self.gmm[block],\
					"trng_count": self.counts_trng[block]};
				pickle.dump(trng_md, fid);

	def get_block(self, feature_vector):
		return self.heuristic2(feature_vector);

	def heuristic1(self, feature_vector):
		max_rss_i = feature_vector.index(max(feature_vector));
		grid_x_y = transmitter_grid_indices[max_rss_i];
		print("Grid", grid_x_y);
		block_x_y = [int(i / utils.CELLS_PER_BLOCK) for i in grid_x_y];
		print("Block", block_x_y);
		return block_x_y;

	def heuristic2(self, feature_vector):
		print("Feature Vector", feature_vector);
		proj = lz.Project(mode = '2D', solver = 'LSE'); #mode = '2D' and solver = LSE
		rssi = sorted(zip(feature_vector), reverse = True)[:3];
		print("Top 3", rssi);

		rssi1_i = feature_vector.index(rssi[0][0]);
		rssi2_i = feature_vector.index(rssi[1][0]);
		rssi3_i = feature_vector.index(rssi[2][0]);
		trans_loc1 = transmitter_locs[rssi1_i];
		trans_loc2 = transmitter_locs[rssi2_i];
		trans_loc3 = transmitter_locs[rssi3_i];
		proj.add_anchor('rssi1', trans_loc1);
		proj.add_anchor('rssi2', trans_loc2);
		proj.add_anchor('rssi3', trans_loc3);
		
		target, label = proj.add_target();
		d1 = utils.compute_distance_from_rssi(rssi[0][0]);
		target.add_measure('rssi1', d1);
		d2 = utils.compute_distance_from_rssi(rssi[1][0]);
		target.add_measure('rssi2', d2);
		d3 = utils.compute_distance_from_rssi(rssi[2][0]);
		target.add_measure('rssi3', d3);
		
		proj.solve();
		x, y = (target.loc.x, target.loc.y);
		x = abs(x);
		y = abs(y);
		# centroid coordinates are never greater than 1000
		if x >= 1000:
			x = 999;
		if y >= 1000:
			y = 999;
		print("x", x, "y", y);
		grid_x_y = [int(i / utils.K) for i in (x, y)];
		print("Grid: ", grid_x_y);
		block_x_y = [int(i / utils.CELLS_PER_BLOCK) for i in grid_x_y];
		print("Block: ", block_x_y);
		return block_x_y;
	
	def on_post(self, req, res):
		raw_json = req.stream.read();
		raw_data = json.loads(str(raw_json, encoding = "utf-8"));
		
		feature_vectors = raw_data["feature_vectors"];
		labels = raw_data["labels"];

		block_feature_vectors = dict();
		block_labels = dict();

		for i in range(utils.BLOCKS):
			for j in range(utils.BLOCKS):
				block_feature_vectors[i * utils.BLOCKS + j]	= [];
				block_labels[i * utils.BLOCKS + j]	= [];

		count = 0;		
		for i, feature_vector in enumerate(feature_vectors):
			x, y = self.get_block(feature_vector);
			print("Block: ", x * utils.BLOCKS + y);
			block_feature_vectors[x * utils.BLOCKS + y].append(feature_vector);
			block_labels[x * utils.BLOCKS + y].append(labels[i]);
			count += 1;
		print(count);	
		#TODO: get data from DB and initialize in case of a failure
		#db_data = get_features_from_db(1);
		#gmm = db_data["gmm"];
		
		#if there is training data in db and training count is 0 (server restarted)
		#if gmm != None and self.count_trng == 0:
		# self.gmm = gmm;
		errors = [];
		pred_locs = [];
		print("Training Size per Block: %d" %(TRNG_SIZE));
		for block in block_feature_vectors:
			print("Block:", block, "Block Size:", len(block_feature_vectors[block]));
			#Very few features to train?
			if self.counts_trng[block] < TRNG_SIZE and\
				len(block_feature_vectors[block]) < self.n_classes:	
				print("skipping block");
				continue;
			
			pkl_file = "gmm.pkl" + str(block);
			if self.counts_trng[block] == 0 and os.path.isfile(pkl_file):
				with open(pkl_file, "rb") as fid:
					trng_md = pickle.load(fid);
					self.gmm[block] = trng_md["gmm"];
					self.counts_trng[block] = trng_md["trng_count"];

			#if training size hasn't been reached, model requires more training
			if self.counts_trng[block] < TRNG_SIZE:
				#train in batches
				tr_data = block_feature_vectors[block];
				tr_labels = block_labels[block];
				self.train(block, tr_data);

				# TRNG_SIZE - self.count_trng is how more training is required
				# if tr_data has more data than required for training, the difference 
				##has to be used for prediction
				if len(tr_data) > TRNG_SIZE:
					start_index = len(tr_data) - TRNG_SIZE;
					self.predict(block, tr_data[start_index:], tr_labels[start_index:], res, errors, pred_locs);			
			
			else:
				# model has been sufficiently trained, now we can start prediction
				test_data = block_feature_vectors[block];
				test_labels = block_labels[block];
				self.predict(block, test_data, test_labels, res, errors, pred_locs);
		
		#if any predictions did happen
		if len(pred_locs) > 0:
			avg_error = float(sum(errors))/len(errors);
			#response = {"predictions" : pred_locs, "avg_error" : avg_error, "errors" : errors};
			response = {"avg_error" : avg_error, "Test Size" : len(errors)};
			res.body = json.dumps(response);
		res.status = falcon.HTTP_200;

data = LocationData();
api.add_route('/location', data);