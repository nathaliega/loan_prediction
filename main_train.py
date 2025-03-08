from src.load_data import load_data
from src.preprocessing import Preprocessing
from src.train import TrainModel
from src.evaluate import Evaluation

train, test = load_data('data/loan_data_set.csv')

data_prep = Preprocessing(train, test, target_column='Loan_Status')
train, test = data_prep.preprocess_data()

trainer = TrainModel(train)
trainer.optimize()
model = trainer.train()
trainer.save_model(model)

evaluation = Evaluation(model_path='../models/model.pkl', test_data=test, output_path='../metrics/metrics.csv')
evaluation.evaluate()