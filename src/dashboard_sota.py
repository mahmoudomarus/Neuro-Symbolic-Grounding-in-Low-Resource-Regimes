"""
Phase 14: SOTA Dashboard - Zero-Shot Recognition Interface.

Interactive dashboard for exploring DINO + CLIP capabilities:
- Upload any image
- Type any concept or description
- Get instant recognition without training

Run from project root: streamlit run src/dashboard_sota.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image

from src.world_model.encoder_sota import SOTA_Encoder
from src.language.binder_sota import SOTA_Binder
from src.memory.episodic import EpisodicMemory


@st.cache_resource
def load_sota_models():
    """Load DINO and CLIP models (cached)."""
    try:
        encoder = SOTA_Encoder(freeze=True)
        binder = SOTA_Binder(freeze=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = encoder.to(device)
        binder = binder.to(device)
        
        return encoder, binder, device
    except Exception as e:
        return None, None, None, str(e)


def main() -> None:
    st.set_page_config(page_title="SOTA Mode - Zero-Shot Recognition", layout="wide")
    
    st.title("🧠 SOTA Mode: Zero-Shot Recognition")
    st.caption("Powered by DINO ViT + CLIP — Recognizes ANY concept without training")
    
    # Load models
    result = load_sota_models()
    if len(result) == 4:
        st.error(f"Failed to load models: {result[3]}")
        st.info("Install required packages: pip install timm open-clip-torch")
        st.stop()
    
    encoder, binder, device = result
    
    # Sidebar: Model info
    with st.sidebar:
        st.header("Model Information")
        st.write(f"**Visual Encoder:** {encoder.model_name}")
        st.write(f"- Embedding dim: {encoder.out_channels}")
        st.write(f"- Type: Pre-trained DINO ViT")
        
        st.write(f"**Language Binder:** {binder.model_name}")
        st.write(f"- Embedding dim: {binder.embedding_dim}")
        st.write(f"- Type: OpenAI CLIP")
        
        st.write(f"**Device:** {device}")
        
        st.divider()
        
        st.header("How It Works")
        st.write("""
        1. **DINO** understands visual concepts from millions of images
        2. **CLIP** maps text and images to the same vector space
        3. Together, they enable **zero-shot recognition** of any concept
        """)
        
        st.divider()
        
        st.header("Try It!")
        st.write("""
        - Upload any image
        - Type concepts to search for
        - Get instant recognition
        - No training required!
        """)
    
    # Initialize session state
    if "memory" not in st.session_state:
        st.session_state.memory = EpisodicMemory()
    if "memory_display" not in st.session_state:
        st.session_state.memory_display = []
    if "step" not in st.session_state:
        st.session_state.step = 0
    
    # Main layout: 3 columns
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("📷 Image Upload")
        
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "gif", "webp", "bmp"],
        )
        
        if uploaded_file is not None:
            # Load and display image
            img_pil = Image.open(uploaded_file)
            
            # Convert to RGB if needed
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            
            st.image(img_pil, caption="Uploaded Image", use_container_width=True)
            
            # Convert to tensor
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                   std=[0.26862954, 0.26130258, 0.27577711]),
            ])
            
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            # Encode image
            with torch.no_grad():
                img_features = binder.embed_image(img_tensor)
                img_features = F.normalize(img_features, dim=1)
            
            # Store in memory if requested
            if st.button("💾 Remember this image"):
                step = st.session_state.step
                st.session_state.memory.store(img_features, step, {"source": "upload"})
                st.session_state.memory_display.append({
                    "Step": step,
                    "Source": "Upload",
                })
                st.session_state.step += 1
                st.success(f"Stored as memory #{step}")
                st.rerun()
            
            # Store for use in other columns
            st.session_state.current_image_features = img_features
            st.session_state.current_image = img_pil
        else:
            st.info("Upload an image to begin")
            st.session_state.current_image_features = None
    
    with col2:
        st.subheader("🔍 Zero-Shot Classification")
        
        if "current_image_features" in st.session_state and st.session_state.current_image_features is not None:
            img_features = st.session_state.current_image_features
            
            # Predefined concepts
            st.write("**Quick Test:**")
            quick_concepts = st.text_input(
                "Enter concepts (comma-separated)",
                value="a dog, a cat, a car, a building, food, nature",
                help="e.g., a red ferrari, a cute puppy, abstract art"
            )
            
            if quick_concepts:
                concepts = [c.strip() for c in quick_concepts.split(",") if c.strip()]
                
                # Encode concepts
                with torch.no_grad():
                    text_features = binder.embed_text(concepts)
                    text_features = F.normalize(text_features, dim=1)
                
                # Compute similarities
                similarities = (img_features @ text_features.T).squeeze(0)
                similarities = F.softmax(similarities, dim=0)
                
                # Display results
                st.write("**Classification Results:**")
                
                # Create dataframe
                results_df = pd.DataFrame({
                    "Concept": concepts,
                    "Confidence": [f"{s.item()*100:.1f}%" for s in similarities],
                    "Score": similarities.cpu().numpy(),
                })
                results_df = results_df.sort_values("Score", ascending=False)
                
                # Show top prediction prominently
                top_concept = results_df.iloc[0]["Concept"]
                top_confidence = results_df.iloc[0]["Confidence"]
                st.success(f"**Top Match:** {top_concept} ({top_confidence})")
                
                # Show all results as bar chart
                st.dataframe(
                    results_df[["Concept", "Confidence"]],
                    use_container_width=True,
                    hide_index=True,
                )
                
                # Visual bar chart
                st.bar_chart(results_df.set_index("Concept")["Score"])
            
            st.divider()
            
            # Free-form text query
            st.write("**Free-Form Query:**")
            freeform_query = st.text_input(
                "Describe what you're looking for",
                placeholder="e.g., something you can eat, a mode of transport",
            )
            
            if freeform_query:
                with torch.no_grad():
                    query_features = binder.embed_text(freeform_query)
                    query_features = F.normalize(query_features, dim=1)
                
                similarity = (img_features @ query_features.T).item()
                match_pct = (similarity + 1) / 2 * 100  # Map [-1, 1] to [0, 100]
                
                if match_pct > 60:
                    st.success(f"Strong match: {match_pct:.1f}%")
                elif match_pct > 40:
                    st.warning(f"Moderate match: {match_pct:.1f}%")
                else:
                    st.error(f"Weak match: {match_pct:.1f}%")
                
                st.progress(match_pct / 100, text=f"Similarity: {match_pct:.1f}%")
        
        else:
            st.info("Upload an image in the left column to begin classification")
    
    with col3:
        st.subheader("🧠 Memory Palace")
        
        # Memory query
        st.write("**Query Memory:**")
        memory_query = st.text_input(
            "Search for memories",
            placeholder="e.g., animals, vehicles, nature",
        )
        
        if memory_query and len(st.session_state.memory._storage) > 0:
            query_features = binder.embed_text(memory_query)
            matches = st.session_state.memory.recall(query_features, threshold=0.5)
            
            if matches:
                st.success(f"Found {len(matches)} matches:")
                for step, sim in matches[:5]:  # Show top 5
                    st.write(f"- Memory #{step}: {sim:.3f} similarity")
            else:
                st.info("No matches found (try lowering the threshold)")
        
        st.divider()
        
        # Recent memories
        st.write("**Recent Memories:**")
        if st.session_state.memory_display:
            recent = st.session_state.memory_display[-10:]
            df = pd.DataFrame(recent)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No memories stored yet")
        
        # Clear memory button
        if st.button("🗑️ Clear All Memory"):
            st.session_state.memory.clear()
            st.session_state.memory_display = []
            st.session_state.step = 0
            st.success("Memory cleared!")
            st.rerun()
    
    # Footer
    st.divider()
    st.caption("💡 **Tip:** Try uploading images of rare objects (platypus, exotic cars, unusual architecture) to see zero-shot recognition in action!")


if __name__ == "__main__":
    main()
