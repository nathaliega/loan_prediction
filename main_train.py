import pandas as pd
from src.load_data import load_data
from src.preprocessing import Preprocessing
from src.train import TrainModel

train, test = load_data('data/loan_data.csv')

preprocessor = Preprocessing(train, test, 'Loan_Status')
train_preprocessed, test_preprocessed = preprocessor.preprocess_data()

trainer = TrainModel(train_preprocessed, test_preprocessed, 'Loan_Status')

model = trainer.train_model()
trainer.save_model(model, 'models/model.pkl')


