# main executable

import json
import pandas as pd
import numpy as np
import datacommons_pandas as dc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


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

	topics_to_names = {}
	for topic in topics_to_stat_vars:
		if topics_to_stat_vars[topic]:
			# Fetch names of all statVars
			response = dc.get_property_values(topics_to_stat_vars[topic],'name')
			stat_var_names = []
			for stat_var in response:
				stat_var_names.extend(response[stat_var])
			topics_to_names[topic] = "; ".join(stat_var_names)
	
	# Formatted as: "dc/topic/Demographics": "Total Population; Population Density; Rate of Population Growth" (3 statVars)
	return topics_to_names

def train_model():
	"""Train the model using json output.
	Returns the model and the CountVectorizer. 
	"""
	f = open('topic_to_names.json')
	data = json.load(f)

	X_train = []
	y_train = []
	for topic in data:
		y_train.append(topic)
		X_train.append(data[topic])

	vectorizer = CountVectorizer(stop_words="english", analyzer="word")

	X_train = vectorizer.fit_transform(X_train)

	# MNB CLASSIFICATION
	mnb = MultinomialNB()
	mnb.fit(X_train,y_train)
	return mnb, vectorizer


def fetch_training_data():
	print("Fetching training data")

	print("Fetching all topics")
	dc_topics = get_topics()

	print("Now we have found all DataCommons topics, for a total topic count of ", len(dc_topics))

	# DC API marks a limit of 500 nodes per API call so we should be able to get all the topic metadata in one.
	# Note we're duping calls that were already made for simplicity then.
	topics_to_names = get_names_from_topics(dc_topics)

	# Write the final output of Topics to list of names for associated variable into a file, so we don't have to always
	# regenerate this!
	with open('topic_to_names.json', 'w') as convert_file: 
		convert_file.write(json.dumps(topics_to_names))

	# Formatted as: "dc/topic/Demographics": "Total Population; Population Density; Rate of Population Growth"
	# Keeping it since it's more human-readable.
	with open("topic_to_names.txt", 'w') as f:
		for key, value in topics_to_names.items():
			f.write('%s:%s\n' % (key, value))


def fetch_test_data():
	"""Extract test data from sample data set. Use both the `Name` and `Chart Title` columns for classification.
	Returns a dataframe with the selected data.
	"""
	return pd.read_csv("dc_sample_data.csv", usecols=["Name", "Chart Title", "StatVar"])

def evaluate(model, vectorizer):
	"""Evaluates a given model using test data. We first try based on the `Name` test data, and fall back to `Chart Title`
	Outputs the prediction in a json file. 
	"""
	test_df = fetch_test_data()
	chart_title_samples = test_df['Chart Title'].values.astype('U')
	X_test = vectorizer.transform(chart_title_samples)
	title_prediction = model.predict(X_test)

	name_samples = test_df['Name'].values.astype('U')
	X_test = vectorizer.transform(name_samples)
	name_prediction = model.predict(X_test)

	index=0
	fail = 0
	predictions = {}
	for pr in name_prediction:
		if name_samples[index] != "nan":
			predictions[name_samples[index]] = pr
		index += 1

	index = 0
	for pr in title_prediction:
		if chart_title_samples[index] != "nan":
			predictions[chart_title_samples[index]] = pr
		index += 1

	with open("classified_test_data.txt", 'w') as f:
		for key, value in predictions.items():
			f.write('%s:%s\n' % (key, value))
		


if __name__ == "__main__":
    model, vectorizer = train_model()
    evaluate(model, vectorizer)