from src.load_data import load_data
from src.preprocessing import PreprocessingPipeline
from src.train import TrainModel
from src.evaluate import Evaluation


train, test = load_data('data/loan_data_set.csv')

pipeline = PreprocessingPipeline()
train, test = pipeline.preprocess_data(train_data=train, test_data=test)

trainer = TrainModel(train)
trainer.optimize()
model = trainer.train()
trainer.save_model(model)

evaluation = Evaluation(model_path='../models/model.pkl', test_data=test, output_path='../metrics/metrics.csv')
evaluation.evaluate()