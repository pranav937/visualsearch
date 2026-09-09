import os
import shutil
import pickle
import pandas as pd
from sqlalchemy import create_engine

# 1. Get valid UUIDs from DB
DB_URL = "postgresql://postgres:Jadequest%403009@3.111.57.216:5432/jaxmart_db"
engine = create_engine(DB_URL)
query = 'SELECT id as listing_id FROM listings'
db_df = pd.read_sql(query, engine)
valid_uuids = set(db_df['listing_id'].astype(str).tolist())

# 2. Delete orphaned folders in products/
base_product_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "products")
deleted_count = 0
if os.path.exists(base_product_dir):
    for folder_name in os.listdir(base_product_dir):
        folder_path = os.path.join(base_product_dir, folder_name)
        if os.path.isdir(folder_path):
            if folder_name not in valid_uuids:
                shutil.rmtree(folder_path, ignore_errors=True)
                deleted_count += 1

print(f"Deleted {deleted_count} orphaned product folders.")

# 3. Delete old FAISS index files
index_file = "visual_search_index.faiss"
mapping_file = "image_mapping.pkl"

if os.path.exists(index_file):
    os.remove(index_file)
    print(f"Deleted {index_file}")
if os.path.exists(mapping_file):
    os.remove(mapping_file)
    print(f"Deleted {mapping_file}")

# 4. Now run rebuild
import visual_search
engine = visual_search.VisualSearchEngine()
engine.build_index_from_directory("products")
print("Rebuilt fresh index!")
