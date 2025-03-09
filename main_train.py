from src.load_data import load_data
from src.preprocessing import PreprocessingPipeline
from src.train import ModelTrainer
from src.evaluate import evaluate_model



train, test = load_data('data/loan_data_set.csv')

pipeline = PreprocessingPipeline()
train, test = pipeline.preprocess_data(train_data=train, test_data=test)

trainer = ModelTrainer(train)
trainer.optimize()
model = trainer.train()
trainer.save_model(model)

evaluate_model(model_path='models/model.pkl', test_data=test, output_path='metrics/metrics.csv')