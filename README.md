# Spacy_NER

ANNOTATION :

Data can be annotated for free at www.dataturks.com

Download annotated data (json format)
Data is converted to spacy format in our code (convert_dataturks_to_spacy function)



TRAINING :

pip install -U spacy(to install spacy)
python -m spacy download en (to download model)

If you want to train a model from scratch set model=None in our code and make sure to give a path where you want to save your new model (you usually need at least a few hundred examples for both training and evaluation) 

To update an existing model give existing model path ( you can already achieve decent results with very few examples – as long as they’re representative.)

Set no of iterations and drop



python spacy_train.py to start training 


