# topic-classification

Topic Classification for Sample Data Set using Logistic Regression.

This repository contains a simple implementation of a topic classifier trained on public data from the Data Commons.

## Training Data

We iterate through the Data Commons Topics, starting at `dc/topic/Root`, and navigating through all child topics. For each topic, we collect all StatVars associated, either directly, or via StatVarPeerGroups.
Our training data consists of ~1300 rows containing the statVar name to the classified Topic.

In order to re-run the steps to create the training data, use the --create_training_data_set flag.

## Test Train Split

When evaluating the model, we split off a portion of it for training, and a portion for testing. The accuracy score on the model initially hovered around 55%, when using MultinomialNaiveBayes. Through tweaks of ngram range applied, and switching to a LogisticRegression model, we reached a peak accuracy score ~83%.

In order to run the accuracy score, use the --test_accuracy flag.

## Classifying Sample Data

Using the --evaluate_model flag will run the classification on the file `dc_sample_data.csv`, and output the predictions.

You can also pass in a different file with column name(s) "Name" and/or "Chart Title" via the --test_data_set_path flag. The file must be a CSV.

## How To Run

Creating the Training data:
```
python3 main.py --create_training_data_set --training_data_set_filename="names_to_topics"
```

Running classifier on sample data and testing accuracy:
```
python3 main.py --evaluate_model --test_data_set_path="dc_sample_data.csv" --test_accuracy
```