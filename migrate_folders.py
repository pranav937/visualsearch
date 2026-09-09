import os
import pandas as pd
from sqlalchemy import create_engine
import shutil

DB_URL = "postgresql://postgres:Jadequest%403009@3.111.57.216:5432/jaxmart_db"
engine = create_engine(DB_URL)
query = 'SELECT id as listing_id, title as "Product Name" FROM listings'
db_df = pd.read_sql(query, engine)

CSV_FILE = "all_data_final_with_images.csv"
csv_df = pd.read_csv(CSV_FILE)

img_mapping = csv_df[['Product Name', 'Local Image Path']].drop_duplicates(subset=['Product Name'])
df = pd.merge(db_df, img_mapping, on='Product Name', how='inner')

base_product_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "products")

renamed_count = 0
not_found_count = 0

for _, row in df.iterrows():
    new_uuid = str(row['listing_id'])
    local_path = str(row['Local Image Path'])
    
    if pd.notna(local_path) and local_path.startswith('products'):
        # Extract old UUID from the path
        # Example path: products\5e5d088d-8eed-47e4-9964-ec70b3b12269\1.png
        parts = local_path.replace('\\', '/').split('/')
        if len(parts) >= 3:
            old_uuid = parts[1]
            old_dir = os.path.join(base_product_dir, old_uuid)
            new_dir = os.path.join(base_product_dir, new_uuid)
            
            if os.path.exists(old_dir):
                if old_dir != new_dir:
                    print(f"Renaming {old_uuid} to {new_uuid}")
                    try:
                        os.rename(old_dir, new_dir)
                        renamed_count += 1
                    except Exception as e:
                        print(f"Error renaming {old_dir}: {e}")
            else:
                if not os.path.exists(new_dir):
                    not_found_count += 1

print(f"Successfully renamed {renamed_count} folders.")
print(f"Folders not found: {not_found_count}")
