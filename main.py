# main executable

import json
import pandas as pd
import datacommons_pandas as dc


def filter_topics(variable):
	if variable.startswith("dc/topic"):
		return True
	return False

def filter_statVars(variable):
	return not filter_topics(variable) and not filter_statVarGroups(variable) and not filter_statVarPeerGroups(variable)

def filter_statVarGroups(variable):
	if variable.startswith("dc/svg"):
		return True
	return False

def filter_statVarPeerGroups(variable):
	if variable.startswith("dc/svpg"):
		return True
	return False

def get_topics():
	dc_topics = []

	next_level_topics = ['dc/topic/Root']
	while len(next_level_topics) > 0:
		# TODO: Maybe use relevantVariableList
		response = dc.get_property_values(next_level_topics,'relevantVariable')
		next_level_topics.clear()

		new_topics = []
		for topic in response:
			new_topics = response[topic]
			new_topics = list(filter(filter_topics, new_topics))
			next_level_topics.extend(new_topics)

		if len(next_level_topics) > 0:
			dc_topics.extend(next_level_topics)
	
	return dc_topics

def get_names_from_topics(dc_topics):
	response = dc.get_property_values(dc_topics,'relevantVariable')
	topics_to_stat_vars = {}

	count = 0
	for topic in response:
		# get all non-topic relevant variables
		stat_vars = list(filter(filter_statVars, response[topic]))
		stat_var_groups = list(filter(filter_statVarGroups, response[topic]))
		stat_var_peer_groups = list(filter(filter_statVarPeerGroups, response[topic]))

		# Confirmed that there are not StatVarGroups, only SVPGs
		if stat_var_peer_groups:
			svpg_resp = dc.get_property_values(stat_var_peer_groups,'member')
			for svpg in svpg_resp:
				stat_vars.extend(svpg_resp[svpg])

		# TODO what if double nestedness
		topics_to_stat_vars[topic] = stat_vars

	print("Done fetching topic to stat vars")
	topics_to_names = {}
	for topic in topics_to_stat_vars:
		if topics_to_stat_vars[topic]:
			response = dc.get_property_values(topics_to_stat_vars[topic],'name')
			stat_var_names = []
			for stat_var in response:
				stat_var_names.extend(response[stat_var])
			topics_to_names[topic] = stat_var_names

	# Write the final output of Topics to list of names for associated variable into a file, so we don't have to always
	# regenerate this!
	with open('topic_to_names.txt', 'w') as convert_file: 
		convert_file.write(json.dumps(topics_to_names))
	


def fetch_training_data():
	print("Fetching training data")

	print("Fetching all topics")
	dc_topics = get_topics()

	print("Now we have found all DataCommons topics, for a total topic count of ", len(dc_topics))

	# DC API marks a limit of 500 nodes per API call so we should be able to get all the topic metadata in one.
	# Note we're duping calls that were already made for simplicity then.
	get_names_from_topics(dc_topics)




def main():
	# Columns we want to read. Some rows have no name but have a ChartTitle.
	usecols = ["Name", "StatVar", "ConciseChartTitle"]

	# Store all columns selected in a data frame.
	df = pd.read_csv("dc_sample_data.csv", usecols=usecols)

	# print output.
	# print(df.to_string())


if __name__ == "__main__":
    fetch_training_data()