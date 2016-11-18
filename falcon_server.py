import falcon;
import pymongo;
import json;
from sklearn.naive_bayes import GaussianNB;
from ast import literal_eval as make_tuple;
import math;

api = application = falcon.API();

class LocationData(object):
	client = pymongo.MongoClient();
	db = client.test;
	
	def eucledian_distance(self, v1, v2):
		if len(v1) != len(v2):
			print("***Both vectors are not of equal length!!***");
		square_differences = [(v1[i] - v2[i]) ** 2 for i in range(len(v1))];
		return math.sqrt(sum(square_differences));

	def predict_gaussian_naive_bayes(self, training_data, training_labels, test_data, test_labels, errors):
		print("Predicting.....");
		predictor = GaussianNB();
		
		#convert training labels from integer tuples to strings
		training_labels_str = [str(label) for label in training_labels];
		#test_labels_str = [str(label) for label in test_labels]
		
		pred_labels_str = predictor.fit(training_data, training_labels_str).predict(test_data);
		pred_labels = [make_tuple(label) for label in pred_labels_str];
		
		#error = accuracy_score(pred_labels, test_labels_str, normalize = False);
		for i in range(len(pred_labels)):
			errors.append(self.eucledian_distance(pred_labels[i], test_labels[i]));

		return pred_labels;	
	
	def get_features_from_db(self, i):
		return self.db.data.find_one({ "_id": i});

	def store_features_in_db(self, i, feature_vectors, labels):
		self.db.data.insert({"_id" : i, "feature_vectors" : feature_vectors, "labels" : labels});
		
	def on_get(self,req,res):	
		i = req.get_param("_id");
		test_data = req.get_param_as_list("test_data[]");
		test_labels = req.get_param_as_list("test_labels[]");
		print(len(test_data), len(test_labels), i);
		data = self.get_features_from_db(i);
		print(data);
		training_data = data["feature_vectors"];
		training_labels = data["labels"];
		errors = [];
		print(len(training_data),len(training_labels));
		self.predict_gaussian_naive_bayes(training_data, training_labels, test_data, test_labels, errors);
		avg_error = float(sum(errors))/sum(errors);
		response = {"avg_error" : avg_error};
		res.body = json.dumps(response);
		res.status = falcon.HTTP_200;
	
	def on_post(self, req, res):
		raw_json = req.stream.read();
		data = json.loads(str(raw_json, encoding = "utf-8"));
		if data["predict"] == 0:
			self.store_features_in_db(data["_id"], data["feature_vectors"], data["labels"]);
			res.status = falcon.HTTP_201;
		
		elif data["predict"] == 1:
			errors = [];
			test_data = data["feature_vectors"];
			test_labels = data["labels"];
			
			i = data["_id"];
			db_data = self.get_features_from_db(i);
			training_data = db_data["feature_vectors"];
			training_labels = db_data["labels"];
			
			predictions = self.predict_gaussian_naive_bayes(training_data, training_labels, test_data, test_labels, errors);
			avg_error = float(sum(errors))/len(errors);
			response = {"avg_error" : avg_error, "predictions" : predictions};
			res.body = json.dumps(response);
			res.status = falcon.HTTP_200;				
		
		else:
			res.status = falcon.HTTP_400;	

data = LocationData();
api.add_route('/location', data);