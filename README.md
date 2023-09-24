
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
1. Clone the repository:

  ```bash
  git clone https://github.com/dompio/disaster-response-pipeline.git
  ```
2. Navigate to the project directory:

  ```
  cd disaster-response-pipeline
  ```
3. Run the following commands in the project's root directory to set up your database and model.

    - To run the ETL pipeline that cleans data and saves the dataframe in a database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains and saves the classifier as a pickle file:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to [http://0.0.0.0:3001/](http://0.0.0.0:3001/).
   
* The code should run using Python 3.*.
* Machine Learning and NLP: numpy, pandas, sklearn, re, nltk
* Database and file processing: sqlalchemy, pickle
* Web application and charts: flask, plotly
* Additional: sys

## Project Motivation<a name="motivation"></a>

In this project, I used a data set containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that they could potentially be sent to an appropriate disaster relief agency.

The project includes a web app where a user can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

## File Descriptions <a name="files"></a>

`disaster_categories.csv` file contains the message categories information.
`disaster_messages.csv` file contains the messages dataset.
`process_data.py` file preprocesses the data and saves it to a database.
`train_classifier.py` file is used to initialise, train, and save a model.
`run.py` file contains the flask web application with data visualisations.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The dataset comes from Stack Overflow Insights - Stack Overflow Annual Developer Survey for 2023, the licensing for the data and other information can be found [here](https://insights.stackoverflow.com/survey).
