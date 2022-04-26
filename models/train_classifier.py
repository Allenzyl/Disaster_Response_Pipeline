import sys
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

import sqlite3
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import subprocess



def load_data(database_filepath):
    """
    Load data from database
    
    arguments:
         database_filepath: database name
         
    outputs:
        X: messages 
        y: everything esle category names.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(y.columns)
    return X, y, category_names


def tokenize(text):
    """ 
    Normalize and tokenize
    
    arguments:
        text: the text to be tokenized
        
    Outputs:
        words: cleaned token
    """
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    words = [PorterStemmer().stem(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w) for w in words]

    return words


def build_model():
    """
    Build machine learning pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    params = {'clf__estimator__n_estimators': [10, 20],
             'vect__min_df': [5, 10]}

    cv = GridSearchCV(pipeline, param_grid=params)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model performance
    
    arguments:
        model: model pipeline
        X_test: independent test dataset
        y_test: dependent test dataset
        category_names
    output:
        scores
    """
    y_pred = model.predict(X_test)
    for idx, col in enumerate(category_names):
        precision, recall, fscore, support = precision_recall_fscore_support(Y_test[col], y_pred[:, idx])
        print('\nReport for the column ({}):\n'.format(col))
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F-score: {}'.format(fscore))


def save_model(model, model_filepath):
    """
    Save model as a pickle file
    
    arguments:
        model: a machine learning classifier
        model_filepath: path to save the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()