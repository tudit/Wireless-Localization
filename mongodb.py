import pymongo;

if __name__ == '__main__':
	client = pymongo.MongoClient();
	db = client.test;
	db.employee.insert({"eid" : "1234" ,"ename" : "abc"});
	employee = db.employee.find_one();
	print(employee);