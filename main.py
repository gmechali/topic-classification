# main executable

import json
import pandas as pd
import numpy as np
import datacommons_pandas as dc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import sys
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean('create_training_data_set', False, 'Outputs a training data set')
flags.DEFINE_string('training_data_set_filename', 'names_to_topics', 'Filename to use for training data.')
flags.DEFINE_boolean('evaluate_model', False, 'Outputs predictions for test data')
flags.DEFINE_boolean('test_accuracy', False, 'Whether to test accuracy with test_train_split.')
flags.DEFINE_string('test_data_set_path', 'dc_sample_data.csv', 'Path to the csv file to use for training data.')
flags.DEFINE_string('prediction_filename', 'prediction.txt', 'Filename for the prediction')
FLAGS(sys.argv) 

def filter_topics(variable):
	if variable.startswith("dc/topic"):
		return True
	return False

def filter_statVars(variable):
	# TODO(gmechali): Confirm there are no StatVarGroups associated with topics.
	return not filter_topics(variable) and not filter_statVarPeerGroups(variable)

def filter_statVarPeerGroups(variable):
	if variable.startswith("dc/svpg"):
		return True
	return False

def get_topics():
	"""Fetches all topics in the Data Commons Graph, starting at dc/topic/Root
	and iterating through it using BFS. Topics have subtopics via the relevantVariable
	property.
	"""
	dc_topics = []

	next_level_topics = ['dc/topic/Root']
	while len(next_level_topics) > 0:
		response = dc.get_property_values(next_level_topics,'relevantVariable')
		next_level_topics.clear()

		for topic in response:
			next_level_topics.extend(list(filter(filter_topics, response[topic])))

		if len(next_level_topics) > 0:
			dc_topics.extend(next_level_topics)
	
	return dc_topics

def get_names_from_topics(dc_topics):
	"""Fetch the names of all statVars associated to the list of topics passed in.
	We rely on the statVars directly associated with topics, or associated indirectly
	via a StatVarPeerGroup.
	"""
	response = dc.get_property_values(dc_topics,'relevantVariable')
	topics_to_stat_vars = {}

	for topic in response:
		# Gets all statVars and StatVarPeerGroups from the response.
		stat_vars = list(filter(filter_statVars, response[topic]))
		stat_var_peer_groups = list(filter(filter_statVarPeerGroups, response[topic]))

		if stat_var_peer_groups:
			# Fetch statVars associated with the statVarPeerGroups found.
			svpg_resp = dc.get_property_values(stat_var_peer_groups,'member')
			for svpg in svpg_resp:
				stat_vars.extend(svpg_resp[svpg])

		# TODO(gmechali): Confirm if there can be double nestedness. 
		topics_to_stat_vars[topic] = stat_vars

	print("Done fetching topic to stat vars")

	# topics_to_stat_vars is a dict containing the topic as the key, and the list of all
	# associated statVars as the value.

	names_to_topics = {}
	for topic in topics_to_stat_vars:
		if topics_to_stat_vars[topic]:
			# Fetch names of all statVars
			response = dc.get_property_values(topics_to_stat_vars[topic],'name')
			stat_var_names = []
			for stat_var in response:
				stat_var_names.extend(response[stat_var])

			for name in stat_var_names:
				names_to_topics[name] = topic
	
	# Formatted as: "Total Population": "dc/topic/Demographics"
	return names_to_topics

def fetch_training_data():
	print("Fetching training data")

	print("Fetching all topics")
	dc_topics = get_topics()

	print("Now we have found all DataCommons topics, for a total topic count of ", len(dc_topics))

	# DC API marks a limit of 500 nodes per API call so we should be able to get all the topic metadata in one.
	# Note we're duping calls that were already made for simplicity then.
	names_to_topics = get_names_from_topics(dc_topics)

	# Write the final output of Topics to list of names for associated variable into a file, so we don't have to always
	# regenerate this!
	with open(FLAGS.training_data_set_filename+'.json', 'w') as convert_file: 
		convert_file.write(json.dumps(names_to_topics))

	# Formatted as: "Total Population": "dc/topic/Demographics"
	# Keeping it since it's more human-readable.
	with open(FLAGS.training_data_set_filename + '.txt', 'w') as f:
		for key, value in names_to_topics.items():
			f.write('%s:%s\n' % (key, value))


def train_model():
	"""Train the model using json output.
	Returns the model and the CountVectorizer. 
	"""
	f = open(FLAGS.training_data_set_filename+'.json')
	data = json.load(f)

	X = []
	y = []
	for name in data:
		X.append(name)
		y.append(data[name])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

	vectorizer = CountVectorizer(stop_words="english", analyzer="word", ngram_range=(1,3))

	X_train = vectorizer.fit_transform(X_train)

	# MNB CLASSIFICATION
	model = LogisticRegression()
	model.fit(X_train,y_train)

	if FLAGS.test_accuracy:
		X_test = vectorizer.transform(X_test)
		test_prediction = model.predict(X_test)
		print("Accuracy: ", accuracy_score(y_test, test_prediction))

	return model, vectorizer

def evaluate():
	"""Evaluates a given model using test data. We first try based on the `Name` test data, and fall back to `Chart Title`
	Outputs the prediction in a json file. 
	"""
	model, vectorizer = train_model()

	test_df = pd.read_csv(FLAGS.test_data_set_path, usecols=["Name", "Chart Title", "StatVar"])

	# Classify based on Name.
	name_samples = test_df['Name'].values.astype('U')
	X_test = vectorizer.transform(name_samples)
	name_prediction = model.predict(X_test)

	# Classify based on Chart Title.
	chart_title_samples = test_df['Chart Title'].values.astype('U')
	X_test = vectorizer.transform(chart_title_samples)
	title_prediction = model.predict(X_test)

	# Store the predictions in dictionary. Keep the Names prediction when available. Fall back to Chart Title.
	index = 0
	predictions = {}
	for pr in name_prediction:
		if name_samples[index] != "nan":
			predictions[name_samples[index]] = pr
		index += 1

	index = 0
	for pr in title_prediction:
		if chart_title_samples[index] != "nan" and name_samples[index] == "nan":
			predictions[chart_title_samples[index]] = pr
		index += 1

	# Output the prediction in txt file.
	with open(FLAGS.prediction_filename, 'w') as f:
		for key, value in predictions.items():
			f.write('%s:%s\n' % (key, value))

def main():
	if FLAGS.create_training_data_set:
		fetch_training_data()
	if FLAGS.evaluate_model:
		evaluate()


if __name__ == "__main__":
	main()