"""
Streamlit UI for AI Dungeon Master
Simple interface with chat and image display
"""

import streamlit as st # type: ignore
import requests
import base64
from PIL import Image
import io
from datetime import datetime
import uuid

# Configuration
import os
API_URL = os.getenv("API_URL", "http://fastapi-service:8000")

# Page configuration
st.set_page_config(
    page_title="AI Dungeon Master",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #8B4513;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .dm-message {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .scene-tag {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "api_available" not in st.session_state:
    st.session_state.api_available = None

def check_api_health():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def send_message(message: str):
    """Send message to API and get response"""
    try:
        response = requests.post(
            f"{API_URL}/api/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The models might be loading...")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

def display_image_from_base64(base64_str):
    """Display image from base64 string"""
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        st.error(f"Error displaying image: {e}")
        return None

# Header
st.markdown('<div class="main-header">🎲 AI Dungeon Master</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-Model Orchestration Demo</div>', unsafe_allow_html=True)

# Check API health
if st.session_state.api_available is None:
    with st.spinner("Checking API connection..."):
        st.session_state.api_available = check_api_health()

if not st.session_state.api_available:
    st.warning("⚠️ API not available. Please ensure the backend is running.")
    if st.button("Retry Connection"):
        st.session_state.api_available = check_api_health()
        st.rerun()

# Main layout - Two columns
col1, col2 = st.columns([1, 1])

# Left column - Chat interface
with col1:
    st.subheader("📜 Adventure Log")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message"><b>You:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message dm-message"><b>Dungeon Master:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
                if "scene" in msg and msg["scene"]:
                    st.markdown(
                        f'<div class="scene-tag">🎨 Scene: {msg["scene"]}</div>',
                        unsafe_allow_html=True
                    )
    
    # Input area
    st.markdown("---")
    
    # Example actions
    with st.expander("💡 Example Actions"):
        st.markdown("""
        - I enter the tavern
        - I look around the room
        - I approach the bartender
        - I draw my sword
        - I open the ancient door
        - I search for traps
        - I cast a light spell
        """)
    
    # User input
    user_input = st.text_input(
        "What do you do?",
        placeholder="Describe your action...",
        key="user_input",
        label_visibility="collapsed"
    )
    
    col_send, col_clear = st.columns([3, 1])
    
    with col_send:
        send_button = st.button("🎲 Take Action", use_container_width=True, type="primary")
    
    with col_clear:
        if st.button("🔄 New Game", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_image = None
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    
    # Process input
    if send_button and user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get response
        with st.spinner("🎲 The Dungeon Master is thinking..."):
            response = send_message(user_input)
        
        if response:
            # Add DM response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("narrative", ""),
                "scene": response.get("scene_description"),
                "timestamp": datetime.now().isoformat()
            })
            
            # Update image if available
            if response.get("image_base64"):
                st.session_state.current_image = response["image_base64"]
        
        st.rerun()

# Right column - Image display
with col2:
    st.subheader("🖼️ Scene Visualization")
    
    if st.session_state.current_image:
        image = display_image_from_base64(st.session_state.current_image)
        if image:
            st.image(image, caption="Current Scene (POV)", use_column_width=True)
    else:
        # Placeholder
        st.info("🎨 Scene images will appear here as you explore the world")
        st.markdown("""
        <div style='text-align: center; padding: 50px; background-color: #f5f5f5; border-radius: 10px; margin-top: 20px;'>
            <h3 style='color: #888;'>🏰</h3>
            <p style='color: #888;'>Your adventure awaits...</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Info section
    with st.expander("ℹ️ About This Demo"):
        st.markdown("""
        ### Multi-Model Orchestration
        
        This demo showcases:
        - **Text Model (Mistral-7B)**: Generates narrative responses
        - **Image Model (SDXL)**: Creates scene visuals with Stable Diffusion XL
        - **Triton Inference Server**: GPU-accelerated serving
        - **MicroK8s**: Container orchestration
        
        #### Tech Stack:
        - Frontend: Streamlit
        - Backend: FastAPI
        - Model Serving: Nvidia Triton (PyTorch backend)
        - Orchestration: MicroK8s
        """)

# Footer
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.metric("Session", st.session_state.session_id[:8] + "...")

with col_footer2:
    st.metric("Messages", len(st.session_state.messages))

with col_footer3:
    status = "🟢 Online" if st.session_state.api_available else "🔴 Offline"
    st.metric("API Status", status)

