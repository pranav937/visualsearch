import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Use a local cache directory for HuggingFace to avoid disk space issues while remaining cross-platform compatible
os.environ["HF_HOME"] = "./.hf_cache"

import streamlit as st
import pandas as pd
from PIL import Image as PILImage
from visual_search import VisualSearchEngine
from sqlalchemy import create_engine

# Streamlit UI Configuration
st.set_page_config(page_title="JaxMart Visual Search", page_icon="🔍", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #B0BEC5;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔍 JaxMart Visual Search</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload an image to find similar products in the Live Database</div>', unsafe_allow_html=True)
st.markdown("---")

DB_URL = "postgresql://postgres:Jadequest%403009@3.111.57.216:5432/jaxmart_db"

@st.cache_data
def load_dataset():
    try:
        # Fetch from database
        engine = create_engine(DB_URL)
        query = '''
        SELECT 
            l.id as listing_id,
            l.title as "Product Name",
            c.name as "Category",
            c.name as "Subcategory",
            bp."businessName" as "Company Name",
            pd."pricePerUnit" as "Price",
            pd."minOrderQty" as "MOQ",
            l."avgRating" as "Rating",
            l."reviewCount" as "Reviews"
        FROM listings l
        LEFT JOIN categories c ON l."categoryId" = c.id
        LEFT JOIN product_details pd ON l.id = pd."listingId"
        LEFT JOIN business_profiles bp ON l."sellerId" = bp."userId"
        '''
        df = pd.read_sql(query, engine)
        
        local_image_paths = []
        base_product_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "products")
        for listing_id in df['listing_id']:
            img_path = None
            if pd.notna(listing_id):
                listing_dir = os.path.join(base_product_dir, str(listing_id))
                if os.path.exists(listing_dir):
                    for ext in ['png', 'jpg', 'jpeg', 'webp']:
                        candidate = os.path.join(listing_dir, f"1.{ext}")
                        if os.path.exists(candidate):
                            # Ensure we store relative path as in original FAISS index
                            img_path = os.path.join("products", str(listing_id), f"1.{ext}")
                            break
            local_image_paths.append(img_path)
            
        df['Local Image Path'] = local_image_paths
        
        if 'Product Name' in df.columns:
            df = df.dropna(subset=['Product Name'])
        df = df.astype(str)
        return df

    except Exception as e:
        st.error(f"Error loading data from Database: {e}")
        return pd.DataFrame()

# Load Dataset (Once)
df_global = load_dataset()

@st.cache_resource
def load_visual_search_engine():
    try:
        import huggingface_hub.constants
        huggingface_hub.constants.HF_HUB_CACHE = "./.hf_cache"
        os.environ["HF_HOME"] = "./.hf_cache"
        os.environ["TRANSFORMERS_CACHE"] = "./.hf_cache"
        
        engine = VisualSearchEngine()
        if engine.load_index():
            return engine
        return None
    except Exception as e:
        st.error(f"Error loading visual search engine: {e}")
        return None

vs_engine = load_visual_search_engine()

uploaded_image = st.file_uploader("Upload a product image", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None and vs_engine is not None:
    st.image(uploaded_image, width=300, caption="Your Uploaded Image")
    st.markdown("---")
    
    with st.spinner("Analyzing image and searching visually similar products..."):
        try:
            img = PILImage.open(uploaded_image)
            # Find the best match to identify the product's subcategory
            best_match_results = vs_engine.search_similar_images(img, top_k=5)
            
            if best_match_results:
                matched_subcategory = None
                
                # Find the subcategory of the closest matched product
                for img_path, score in best_match_results:
                    norm_path = os.path.normpath(str(img_path)).lower()
                    if not df_global.empty and 'Local Image Path' in df_global.columns:
                        # Normalize column paths for safe comparison
                        matched_rows = df_global[df_global['Local Image Path'].apply(lambda x: os.path.normpath(str(x)).lower() if pd.notna(x) else '') == norm_path]
                        if not matched_rows.empty:
                            p_row = matched_rows.iloc[0]
                            matched_subcategory = p_row.get('Subcategory', None)
                            
                            # If subcategory is missing or nan, fallback to Category
                            if pd.isna(matched_subcategory) or str(matched_subcategory).strip() == '' or str(matched_subcategory).lower() == 'nan':
                                matched_subcategory = p_row.get('Category', None)
                            if matched_subcategory:
                                break
                
                if matched_subcategory and str(matched_subcategory).lower() != 'nan':
                    st.success(f"**Identified Category/Subcategory:** {matched_subcategory}")
                    st.markdown(f"### 📷 All Products in '{matched_subcategory}':")
                    
                    # Filter dataset by this subcategory or category
                    subcat_df = df_global[(df_global['Subcategory'] == matched_subcategory) | (df_global['Category'] == matched_subcategory)]
                    
                    # Remove exact duplicate product names so pictures don't repeat
                    subcat_df = subcat_df.drop_duplicates(subset=['Product Name'])
                    
                    # Limit to 15 products to not overwhelm the UI
                    display_df = subcat_df.head(15)
                    
                    cols = st.columns(3)
                    
                    for i, (_, row) in enumerate(display_df.iterrows()):
                        col_idx = i % 3
                        with cols[col_idx]:
                            with st.container(border=True):
                                local_img = row.get('Local Image Path', '')
                                if os.path.exists(local_img):
                                    try:
                                        img_pil = PILImage.open(local_img).convert('RGB')
                                        img_pil = img_pil.resize((300, 300))
                                        st.image(img_pil) 
                                    except Exception:
                                        pass
                                
                                p_name = row.get('Product Name', 'Unknown')
                                if len(p_name) > 40:
                                    p_name = p_name[:37] + "..."
                                    
                                st.markdown(f"<h4 style='color: #FF4B2B; margin-bottom: 5px; min-height: 45px;'>{p_name}</h4>", unsafe_allow_html=True)
                                
                                # All details displayed directly without hiding
                                st.markdown(f"**💰 Price:** <span style='color: #4CAF50; font-weight: bold;'>{row.get('Price', 'N/A')}</span>", unsafe_allow_html=True)
                                st.markdown(f"**🏢 Company:** {row.get('Company Name', 'N/A')}")
                                st.markdown(f"**📍 Location:** {row.get('Location', 'N/A')}")
                                st.markdown(f"**📦 MOQ:** {row.get('MOQ', 'N/A')}")
                                st.markdown(f"**⭐ Rating:** {row.get('Rating', 'N/A')} ({row.get('Reviews', '0')} reviews)")
                                st.markdown("<br>", unsafe_allow_html=True)
                else:
                    st.warning("Could not identify a clear Subcategory for this image in the database.")
            else:
                st.warning("Visual Search index is empty. Please wait for the background indexing to finish.")
        except Exception as e:
            st.error(f"Visual search failed: {e}")
