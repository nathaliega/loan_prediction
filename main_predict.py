import pandas as pd
from src._pipeline import PreprocessingPipeline
from src.inference import ModelPredictor

df = pd.read_csv('data/loan_data_predict.csv')

pipeline = PreprocessingPipeline()
df = pipeline.transform_for_inference(df)

model = ModelPredictor('models/model.pkl')
preds = model.predict(df)

