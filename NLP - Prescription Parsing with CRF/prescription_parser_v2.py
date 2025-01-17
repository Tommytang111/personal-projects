#Purpose: Use Named Entity Recognition (NER) to extract prescription information from text 
#Model(s): Conditional Random Field
#Tommy Tang
#Nov 15th, 2024

#IMPORT LIBRARIES
from nltk.tag import pos_tag
from sklearn_crfsuite import CRF, metrics
from sklearn.metrics import make_scorer,confusion_matrix
from pprint import pprint
from sklearn.metrics import f1_score,classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import string
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterSampler
import scipy.stats

#DATA
""" 
INPUT SIG: sigs --> ['take 3 tabs for 10 days'] 
TOKENS: input_sigs --> [['take', '3', 'tabs', 'for', '10', 'days']]      
LABELS: output_labels --> [['Method','Qty', 'Form', 'FOR', 'Duration', 'DurationUnit']]        
"""
sigs = ["for 5 to 6 days", "inject 2 units", "x 2 weeks", "x 3 days", "every day", "every 2 weeks", "every 3 days", "every 1 to 2 months", "every 2 to 6 weeks", "every 4 to 6 days", "take two to four tabs", "take 2 to 4 tabs", "take 3 tabs orally bid for 10 days at bedtime", "swallow three capsules tid orally", "take 2 capsules po every 6 hours", "take 2 tabs po for 10 days", "take 100 caps by mouth tid for 10 weeks", "take 2 tabs after an hour", "2 tabs every 4-6 hours", "every 4 to 6 hours", "q46h", "q4-6h", "2 hours before breakfast", "before 30 mins at bedtime", "30 mins before bed", "and 100 tabs twice a month", "100 tabs twice a month", "100 tabs once a month", "100 tabs thrice a month", "3 tabs daily for 3 days then 1 tab per day at bed", "30 tabs 10 days tid", "take 30 tabs for 10 days three times a day", "qid q6h", "bid", "qid", "30 tabs before dinner and bedtime", "30 tabs before dinner & bedtime", "take 3 tabs at bedtime", "30 tabs thrice daily for 10 days ", "30 tabs for 10 days three times a day", "Take 2 tablets a day", "qid for 10 days", "every day", "take 2 caps at bedtime", "apply 3 drops before bedtime", "take three capsules daily", "swallow 3 pills once a day", "swallow three pills thrice a day", "apply daily", "apply three drops before bedtime", "every 6 hours", "before food", "after food", "for 20 days", "for twenty days", "with meals"]
input_sigs = [['for', '5', 'to', '6', 'days'], ['inject', '2', 'units'], ['x', '2', 'weeks'], ['x', '3', 'days'], ['every', 'day'], ['every', '2', 'weeks'], ['every', '3', 'days'], ['every', '1', 'to', '2', 'months'], ['every', '2', 'to', '6', 'weeks'], ['every', '4', 'to', '6', 'days'], ['take', 'two', 'to', 'four', 'tabs'], ['take', '2', 'to', '4', 'tabs'], ['take', '3', 'tabs', 'orally', 'bid', 'for', '10', 'days', 'at', 'bedtime'], ['swallow', 'three', 'capsules', 'tid', 'orally'], ['take', '2', 'capsules', 'po', 'every', '6', 'hours'], ['take', '2', 'tabs', 'po', 'for', '10', 'days'], ['take', '100', 'caps', 'by', 'mouth', 'tid', 'for', '10', 'weeks'], ['take', '2', 'tabs', 'after', 'an', 'hour'], ['2', 'tabs', 'every', '4-6', 'hours'], ['every', '4', 'to', '6', 'hours'], ['q46h'], ['q4-6h'], ['2', 'hours', 'before', 'breakfast'], ['before', '30', 'mins', 'at', 'bedtime'], ['30', 'mins', 'before', 'bed'], ['and', '100', 'tabs', 'twice', 'a', 'month'], ['100', 'tabs', 'twice', 'a', 'month'], ['100', 'tabs', 'once', 'a', 'month'], ['100', 'tabs', 'thrice', 'a', 'month'], ['3', 'tabs', 'daily', 'for', '3', 'days', 'then', '1', 'tab', 'per', 'day', 'at', 'bed'], ['30', 'tabs', '10', 'days', 'tid'], ['take', '30', 'tabs', 'for', '10', 'days', 'three', 'times', 'a', 'day'], ['qid', 'q6h'], ['bid'], ['qid'], ['30', 'tabs', 'before', 'dinner', 'and', 'bedtime'], ['30', 'tabs', 'before', 'dinner', '&', 'bedtime'], ['take', '3', 'tabs', 'at', 'bedtime'], ['30', 'tabs', 'thrice', 'daily', 'for', '10', 'days'], ['30', 'tabs', 'for', '10', 'days', 'three', 'times', 'a', 'day'], ['take', '2', 'tablets', 'a', 'day'], ['qid', 'for', '10', 'days'], ['every', 'day'], ['take', '2', 'caps', 'at', 'bedtime'], ['apply', '3', 'drops', 'before', 'bedtime'], ['take', 'three', 'capsules', 'daily'], ['swallow', '3', 'pills', 'once', 'a', 'day'], ['swallow', 'three', 'pills', 'thrice', 'a', 'day'], ['apply', 'daily'], ['apply', 'three', 'drops', 'before', 'bedtime'], ['every', '6', 'hours'], ['before', 'food'], ['after', 'food'], ['for', '20', 'days'], ['for', 'twenty', 'days'], ['with', 'meals']]
output_labels = [['FOR', 'Duration', 'TO', 'DurationMax', 'DurationUnit'], ['Method', 'Qty', 'Form'], ['FOR', 'Duration', 'DurationUnit'], ['FOR', 'Duration', 'DurationUnit'], ['EVERY', 'Period'], ['EVERY', 'Period', 'PeriodUnit'], ['EVERY', 'Period', 'PeriodUnit'], ['EVERY', 'Period', 'TO', 'PeriodMax', 'PeriodUnit'], ['EVERY', 'Period', 'TO', 'PeriodMax', 'PeriodUnit'], ['EVERY', 'Period', 'TO', 'PeriodMax', 'PeriodUnit'], ['Method', 'Qty', 'TO', 'Qty', 'Form'], ['Method', 'Qty', 'TO', 'Qty', 'Form'], ['Method', 'Qty', 'Form', 'PO', 'BID', 'FOR', 'Duration', 'DurationUnit', 'AT', 'WHEN'], ['Method', 'Qty', 'Form', 'TID', 'PO'], ['Method', 'Qty', 'Form', 'PO', 'EVERY', 'Period', 'PeriodUnit'], ['Method', 'Qty', 'Form', 'PO', 'FOR', 'Duration', 'DurationUnit'], ['Method', 'Qty', 'Form', 'BY', 'PO', 'TID', 'FOR', 'Duration', 'DurationUnit'], ['Method', 'Qty', 'Form', 'AFTER', 'Period', 'PeriodUnit'], ['Qty', 'Form', 'EVERY', 'Period', 'PeriodUnit'], ['EVERY', 'Period', 'TO', 'PeriodMax', 'PeriodUnit'], ['Q46H'], ['Q4-6H'], ['Qty', 'PeriodUnit', 'BEFORE', 'WHEN'], ['BEFORE', 'Qty', 'M', 'AT', 'WHEN'], ['Qty', 'M', 'BEFORE', 'WHEN'], ['AND', 'Qty', 'Form', 'Frequency', 'Period', 'PeriodUnit'], ['Qty', 'Form', 'Frequency', 'Period', 'PeriodUnit'], ['Qty', 'Form', 'Frequency', 'Period', 'PeriodUnit'], ['Qty', 'Form', 'Frequency', 'Period', 'PeriodUnit'], ['Qty', 'Form', 'Frequency', 'FOR', 'Duration', 'DurationUnit', 'THEN', 'Qty', 'Form', 'Frequency', 'PeriodUnit', 'AT', 'WHEN'], ['Qty', 'Form', 'Duration', 'DurationUnit', 'TID'], ['Method', 'Qty', 'Form', 'FOR', 'Duration', 'DurationUnit', 'Qty', 'TIMES', 'Period', 'PeriodUnit'], ['QID', 'Q6H'], ['BID'], ['QID'],['Qty', 'Form', 'BEFORE', 'WHEN', 'AND', 'WHEN'], ['Qty', 'Form', 'BEFORE', 'WHEN', 'AND', 'WHEN'], ['Method', 'Qty', 'Form', 'AT', 'WHEN'], ['Qty', 'Form', 'Frequency', 'DAILY', 'FOR', 'Duration', 'DurationUnit'], ['Qty', 'Form', 'FOR', 'Duration', 'DurationUnit', 'Frequency', 'TIMES', 'Period', 'PeriodUnit'], ['Method', 'Qty', 'Form', 'Period', 'PeriodUnit'], ['QID', 'FOR', 'Duration', 'DurationUnit'], ['EVERY', 'PeriodUnit'], ['Method', 'Qty', 'Form', 'AT', 'WHEN'], ['Method', 'Qty', 'Form', 'BEFORE', 'WHEN'], ['Method', 'Qty', 'Form', 'DAILY'], ['Method', 'Qty', 'Form', 'Frequency', 'Period', 'PeriodUnit'], ['Method', 'Qty', 'Form', 'Frequency', 'Period', 'PeriodUnit'], ['Method', 'DAILY'], ['Method', 'Qty', 'Form', 'BEFORE', 'WHEN'], ['EVERY', 'Period', 'PeriodUnit'], ['BEFORE', 'FOOD'], ['AFTER', 'FOOD'], ['FOR', 'Duration', 'DurationUnit'], ['FOR', 'Duration', 'DurationUnit'], ['WITH', 'FOOD']]

#FUNCTIONS
def tuples_maker(inp, out):
    """
    Combines input and output into a list of tuples.
    """
    #Container
    sample_data = []
    #Iterate over input and output
    for i, j in enumerate(inp):
        #Container for each sentence
        sentence = []
        for k in range(len(out[i])):
            pair = [inp[i][k], out[i][k]]
            #Add pair to the sentence
            sentence.append(tuple(pair))
        #Add sentence to the container
        sample_data.append(sentence)
    return sample_data

def triples_maker(whole_data):
    """
    Adds POS tags to the tuples.
    """
    triples = []
    for element in whole_data:
        sentence = []
        for tup in element:
            l = list(tup)
            #Insert POS tag into second position
            l.insert(1, pos_tag([l[0]])[0][1])
            sentence.append(tuple(l))
        triples.append(sentence)
    return triples

def token_to_features(doc, i):
    """
    Extract features from a given token.
    Features: SOS, EOS, lowercase, uppercase, title, digit, postag, previous_tag, next_tag
    """
    word, postag = doc[i][:2]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1, postag1 = doc[i-1][:2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(doc)-1:
        word1, postag1 = doc[i+1][:2]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': word1.isdigit(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features

def get_feats(data):
    """
    Runs feature extractor on training data.
    """
    feats = []
    labels = []
    for sentence in data:
        sent_feats = []
        sent_labels = []
        for i in range(len(sentence)):
            sent_feats.append(token_to_features(sentence, i))
            sent_labels.append(sentence[i][2])
        feats.append(sent_feats)
        labels.append(sent_labels)
    return feats, labels

def predict(sig):
    """
    predict(sig)
    Purpose: Labels the given sig into corresponding labels
    @param sig. A Sentence  # A medical prescription sig written by a doctor
    @return     A list      # A list with predicted labels (first level of labeling)
    >>> predict('2 tabs every 4 hours')
    [['Qty', 'Form', 'EVERY', 'Period', 'PeriodUnit']]
    >>> predict('2 tabs with food')
    [['Qty', 'Form', 'WITH', 'FOOD']]
    >>> predict('2 tabs qid x 30 days')
    [['Qty', 'Form', 'QID', 'FOR', 'Duration', 'DurationUnit']]
    """
    # Tokenize the input sentence
    tokens = sig.split()
    
    # Create tuples with POS tags
    tuples = [(token, pos_tag([token])[0][1]) for token in tokens]
    
    # Extract features
    features = [token_to_features(tuples, i) for i in range(len(tuples))]
    
    # Load the trained CRF model
    crf = joblib.load('crf_model.pkl')
    
    # Predict the labels
    predicted_labels = crf.predict([features])[0]
    
    # Combine tokens with their predicted labels
    predictions = list(zip(tokens, predicted_labels))
    
    return predictions

#PIPELINE
# Create the training data
sample_data = triples_maker(tuples_maker(input_sigs, output_labels))

# Extract features and labels
X, y = get_feats(sample_data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the lengths of the splits
print(f"Number of training feature sets: {len(X_train)}")
print(f"Number of training labels: {len(y_train)}")
print(f"Number of testing feature sets: {len(X_test)}")
print(f"Number of testing labels: {len(y_test)}")

# Define parameter grid
param_grid = {
    'algorithm': ['lbfgs', 'l2sgd', 'ap', 'pa', 'arow'],
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
    'max_iterations': [50, 100, 200]
}

param_list = list(ParameterSampler(param_grid, n_iter=50, random_state=42))

# Convert c1 and c2 to strings
for params in param_list:
    params['c1'] = str(params['c1'])
    params['c2'] = str(params['c2'])

# Create the CRF model
crf = CRF()

# Use RandomizedSearchCV for hyperparameter tuning
rs = RandomizedSearchCV(crf, param_grid, cv=3, verbose=1, n_jobs=-1, n_iter=50, scoring='f1_weighted')

# Fit the model
rs.fit(X_train, y_train)

# Get the best parameters
best_params = rs.best_params_
print(f"Best parameters: {best_params}")

# Train the CRF model with the best parameters
crf = CRF(**best_params)
crf.fit(X_train, y_train)

# Save the model
joblib.dump(crf, 'crf_model.pkl')

# Generate predictions
y_pred = crf.predict(X_test)

# Print the classification report
print(metrics.flat_classification_report(y_test, y_pred, digits=3))

#SAMPLE PREDICTIONS
prediction1 = predict("take 2 tabs every 6 hours x 10 days")
prediction2 = predict("2 capsu for 10 day at bed")
prediction3 = predict("5 days 2 tabs at bed")
print('Predictions:')
print(prediction1, prediction2, prediction3, sep='\n')





















