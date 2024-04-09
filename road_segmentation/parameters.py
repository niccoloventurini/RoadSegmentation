import torch

DATA_PATH = "C:/Users/nicco/OneDrive/Desktop/Applicazioni/Neuro-X/ML/project 2/ml-project-2-ml-project2-min/data"
SUBMISSION_DIR = "C:/Users/nicco/OneDrive/Desktop/Applicazioni/Neuro-X/ML/project 2/ml-project-2-ml-project2-min/submission"
SUBMISSION_PATH = "C:/Users/nicco/OneDrive/Desktop/Applicazioni/Neuro-X/ML/project 2/ml-project-2-ml-project2-min/submission/submission.csv"
PATCH_SIZE = 400 #value find with cross validation 400
BATCH_SIZE = 20
LR =  0.001 #value find with cross validation 0.001
MAX_ITER = 50
THRESHOLD = 0.25 #value find with cross validation 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SIZE = 0.2
K_FOLD = 5
SEED= 56

THRESHOLD_VALIDATION_VECTOR = [0.15,0.17,0.20, 0.25, 0.30, 0.35] # best found 0.17
LEARNING_RATE_VALIDATION_VECTOR = [0.0003, 0.0001, 0.001, 0.01]  # best found 0.001
PATCH_SIZE_VALIDATION_VECTOR = [80,400]  # best found 400

