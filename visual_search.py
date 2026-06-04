import os
# Set HF Cache to G drive if available to avoid C drive space issues
os.environ["HF_HOME"] = "G:/jaxmart/.hf_cache"

import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
import faiss
import pickle
from typing import List, Tuple, Union
class VisualSearchEngine:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", index_file: str = "visual_search_index.faiss", mapping_file: str = "image_mapping.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.index_file = index_file
        self.mapping_file = mapping_file
        
        self.model = None
        self.processor = None
        self.index = None
        self.image_paths = []
        self.embedding_dim = 512 # Default for clip-vit-base-patch32

    def load_model(self):
        if self.model is None:
            self.model = CLIPVisionModelWithProjection.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.embedding_dim = self.model.config.projection_dim

    def load_index(self) -> bool:
        """Returns True if index was successfully loaded, False otherwise."""
        if os.path.exists(self.index_file) and os.path.exists(self.mapping_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.mapping_file, 'rb') as f:
                self.image_paths = pickle.load(f)
            return True
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.image_paths = []
            return False

    def get_image_embedding(self, image: Union[str, Image.Image]) -> np.ndarray:
        self.load_model()
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.image_embeds
            
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()

    def build_index_from_directory(self, image_dir: str, batch_size: int = 32):
        self.load_model()
        self.load_index()
        
        print(f"Scanning directory {image_dir} for images...")
        all_image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    all_image_files.append(os.path.join(root, file))
                    
        print(f"Found {len(all_image_files)} images. Generating embeddings...")
        
        new_embeddings = []
        new_paths = []
        
        for i in range(0, len(all_image_files), batch_size):
            batch_paths = all_image_files[i:i+batch_size]
            batch_images = []
            valid_paths = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    pass
            
            if not batch_images: continue
                
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.image_embeds
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                
            new_embeddings.append(features.cpu().numpy())
            new_paths.extend(valid_paths)
            
            print(f"Processed {min(i+batch_size, len(all_image_files))}/{len(all_image_files)} images")

        if new_embeddings:
            embeddings_matrix = np.vstack(new_embeddings)
            self.index.add(embeddings_matrix)
            self.image_paths.extend(new_paths)
            
            faiss.write_index(self.index, self.index_file)
            with open(self.mapping_file, 'wb') as f:
                pickle.dump(self.image_paths, f)
            print(f"Successfully added {len(new_paths)} images to the index.")

    def search_similar_images(self, query_image: Union[str, Image.Image], top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
            
        query_embedding = self.get_image_embedding(query_image)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(dist)))
                
        return results

if __name__ == "__main__":
    # Run this to build the index!
    engine = VisualSearchEngine()
    engine.build_index_from_directory("products")
