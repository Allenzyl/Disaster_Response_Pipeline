# Disaster Response Pipeline Project

### Project Overview:
This code is designed to initiate a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Installation and library needed:
This project was written in Python 3.8. Packages needed:
- pandas
- numpy
- re
- pickle
- nltk
- flask
- json
- plotly
- sklearn
- sqlalchemy
- sys

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Go to `app` directory: `cd app`
3. Run your web app: `python run.py`
4. Click the `PREVIEW` button to open the homepage

### Folder structure:
- app: Folder cointains app files.
    - run.py: Code iniate the web app.
- data: Folder contains messages and categories datasets in csv format and SQL database created by process_data.py code.
    - process_data.py: Code reads in message data and message categories in csv formata, clean and creates a SQL database
- models: Folder contains ML code and model output in .pkl format.
    - train_classifier.py: Code trains the ML model to classify the text with the SQL database

### Summary of the results
Screenshot 1:
