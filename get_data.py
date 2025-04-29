from datasets import load_dataset
import pandas as pd

dataset = load_dataset("brackozi/Resume", split="train")
dataset.to_csv("datasets/resumes.csv")

job_listing = pd.read_csv("datasets/job_listing.csv")
resumes = pd.read_csv("datasets/resumes.csv")

resumes.rename({"Category":"Title"}, inplace=True)
resumes["Category"] = "Resume"
job_listing["Category"] = "Joblisting"
resumes = resumes.sample(n=100, random_state=42)
data = pd.concat([resumes,job_listing])

data.drop(columns="Unnamed: 0", inplace=True)

data.to_csv("datasets/data.csv")