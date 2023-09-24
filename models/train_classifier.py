import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    '''
    Load a SQLite database and split into X, y matrices
    INPUT 
        database_filepath - path to a database file
        
    OUTPUT
        X - matrix of feature values
        y - target variable
        labels - the category names
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql("SELECT * FROM combined", engine)
    X = df.message
    y = df[df.columns[4:]]
    labels = y.columns
    return X, y, labels


def tokenize(text):
    '''
    Process text data by tokenizing and lemmatizing
    INPUT 
        text - text to process
        
    OUTPUT
        clean_tokens - processed tokens
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build pipeline and use GridSearch to find better parameters
        
    OUTPUT
        cv - pipeline using grid search cross validation
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
         'clf__estimator__n_estimators': [5, 10],
         'clf__estimator__min_samples_split': [2, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Evaluate the model by applying it to a test set and providing results 
    (f1 score, precision and recall for each category)
    INPUT 
        model - an ML pipeline
        X_test - test features
        Y_test - test target variable
        category_names - labels for multi-output
    '''
    y_pred = model.predict(X_test)
    
    y_pred = pd.DataFrame(y_pred, columns = y_test.columns)
    
    for column in y_test.columns:
        results_report = classification_report(y_test[column],y_pred[column])
        print(results_report) 


def save_model(model, model_filepath):
    '''
    Main function to kick off data processing
    INPUT 
    model - an ML pipeline
    model_filepath - a path to save the trained model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Main function to create and test a pipeline
    '''
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