import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from visual_search_app import load_dataset
import pickle

df = load_dataset()
db_paths = set(df['Local Image Path'].dropna().tolist())

with open('image_mapping.pkl', 'rb') as f:
    faiss_paths = pickle.load(f)

print(f"Total paths in FAISS: {len(faiss_paths)}")
print(f"Total valid paths in DB df: {len(db_paths)}")

matched_paths = 0
for p in faiss_paths:
    norm_p = os.path.normpath(p).lower()
    if any(os.path.normpath(db_p).lower() == norm_p for db_p in db_paths):
        matched_paths += 1

print(f"Paths in FAISS that match a DB row: {matched_paths} out of {len(faiss_paths)}")
