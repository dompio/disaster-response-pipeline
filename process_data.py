import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load messages and categories datasets from csv files
    INPUT 
        messages_filepath - the path to the messages dataset
        categories_filepath - the path to the categories dataset
        
    OUTPUT
        df - a dataframe resulting from merging messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer', on=['id'])
    return df


def clean_data(df):
    '''
    Clean the categories by converting the values to integers; drop duplicates
    INPUT 
        df - a dataframe resulting from merging messages and categories
        
    OUTPUT
        df - cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the 1st row and use it to extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] = categories[column].astype('int')
  
    df.drop('categories', axis=1, inplace=True)
    categories.index = df.index
    df = pd.concat([df,categories], axis=1)
    df.drop_duplicates(keep='last', inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Save dataframe to a SQLite database
    INPUT 
        df - a preprocessed dataframe
        database_filename - the filepath to save the database to
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('combined', engine, if_exists = 'replace', index=False)


def main():
    '''
    Main function to kick off data processing
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()