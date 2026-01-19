"""
Streamlit application for PDF Merger Bot.
"""
import os
import io
import uuid
import streamlit as st
from datetime import datetime
from pathlib import Path
from pdf_processor import PDFProcessor
# Page configuration
st.set_page_config(
    page_title="PDF Merger",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'merged_pdf' not in st.session_state:
    st.session_state.merged_pdf = None

# Initialize PDF processor
pdf_processor = PDFProcessor()

# Create uploads directory
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Title and description
st.title("📄 PDF Merger Tool")
st.markdown("Combine multiple PDF files into one document effortlessly!")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **PDF Merger** is a simple tool to:
    - Upload multiple PDF files
    - Arrange them in any order
    - Merge them into a single document
    - Download the merged PDF
    """)
    
    st.info("🔒 **Security & Privacy:** Once you download your merged PDF, no files are saved in memory. Your privacy is our priority!")
    
    st.divider()
    
    st.subheader("📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Files Uploaded", len(st.session_state.uploaded_files))
    with col2:
        st.metric("Ready to Merge", "Yes" if len(st.session_state.uploaded_files) > 1 else "No")
    
    st.divider()
    
    st.markdown("## 👨‍💻 About This Tool")
    st.markdown("**Created by:** Chrisma Hestiwibowo")
    st.markdown("**Role:** Project Manager, AI Developer & Data Scientist")
    st.markdown("**✨ Project:** PDF Merger - Secure Document Processing")
    st.markdown(f"**📅 Launched:** {datetime.now().strftime('%B %Y')}")
    st.markdown("---")
    st.markdown("### 🔗 Connect With Me")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/chrismahestiwibowo)")
    with col2:
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/chrismahestiwibowo-tech)")
    st.markdown("---")
    st.markdown("💡 **Tip:** This app is part of my AI portfolio! Check back for updates as I integrate more advanced ML/AI techniques.")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Step 1: Upload PDF Files")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="You can select multiple PDF files at once"
    )
    
    if uploaded_files:
        # Save uploaded files to session state
        file_details = []
        for file in uploaded_files:
            # Save file temporarily
            unique_filename = f"{len(st.session_state.uploaded_files)}_{file.name}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            with open(file_path, 'wb') as f:
                f.write(file.getbuffer())
            
            file_details.append({
                'id': str(uuid.uuid4()),  # Unique ID for each file
                'name': file.name,
                'path': file_path,
                'size': file.size
            })
        
        st.session_state.uploaded_files = file_details
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")

with col2:
    st.subheader("Actions")
    if st.button("🗑️ Clear All", use_container_width=True):
        # Clean up files
        for file_detail in st.session_state.uploaded_files:
            if os.path.exists(file_detail['path']):
                os.remove(file_detail['path'])
        st.session_state.uploaded_files = []
        st.session_state.merged_pdf = None
        st.rerun()

# Display uploaded files
if st.session_state.uploaded_files:
    # Ensure all files have unique IDs
    for file_detail in st.session_state.uploaded_files:
        if 'id' not in file_detail:
            file_detail['id'] = str(uuid.uuid4())
    
    st.subheader("Step 2: Arrange Files")
    
    st.markdown("**📁 Arrange your files:**")
    st.info("💡 Use the arrow buttons to reorder, or delete files with the trash icon")
    
    # Create placeholder for dynamic content
    placeholder = st.empty()
    
    # First pass - check all buttons and collect actions
    action_buttons = {}
    for idx, file_detail in enumerate(st.session_state.uploaded_files, 1):
        file_id = file_detail['id']
        cols = st.columns([0.3, 0.2, 0.2, 0.2, 3.5, 0.3])
        
        with cols[0]:
            pass  # Number will be shown in second pass
        
        with cols[1]:
            if idx > 1:
                up_clicked = st.button("⬆️", key=f"up_{file_id}", help="Move up")
                if up_clicked:
                    current_idx = next(i for i, f in enumerate(st.session_state.uploaded_files) if f['id'] == file_id)
                    if current_idx > 0:
                        temp = st.session_state.uploaded_files[current_idx - 1]
                        st.session_state.uploaded_files[current_idx - 1] = st.session_state.uploaded_files[current_idx]
                        st.session_state.uploaded_files[current_idx] = temp
                        st.rerun()
        
        with cols[2]:
            if idx < len(st.session_state.uploaded_files):
                down_clicked = st.button("⬇️", key=f"down_{file_id}", help="Move down")
                if down_clicked:
                    current_idx = next(i for i, f in enumerate(st.session_state.uploaded_files) if f['id'] == file_id)
                    if current_idx < len(st.session_state.uploaded_files) - 1:
                        temp = st.session_state.uploaded_files[current_idx + 1]
                        st.session_state.uploaded_files[current_idx + 1] = st.session_state.uploaded_files[current_idx]
                        st.session_state.uploaded_files[current_idx] = temp
                        st.rerun()
        
        with cols[3]:
            remove_clicked = st.button("🗑️", key=f"remove_{file_id}", help="Remove")
            if remove_clicked:
                if os.path.exists(file_detail['path']):
                    os.remove(file_detail['path'])
                st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f['id'] != file_id]
                st.rerun()
    
    # Second pass - display current state
    st.markdown("**Current order:**")
    for idx, file_detail in enumerate(st.session_state.uploaded_files, 1):
        cols = st.columns([0.1, 5])
        with cols[0]:
            st.write(f"**{idx}.**")
        with cols[1]:
            st.markdown(f"📄 {file_detail['name']} ({file_detail['size'] / 1024:.1f} KB)")
    
    st.divider()
    
    # Merge button
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🔗 Merge PDFs", use_container_width=True, type="primary"):
            with st.spinner("Merging PDFs..."):
                try:
                    # Get file paths and verify they exist
                    pdf_files = []
                    for f in st.session_state.uploaded_files:
                        if os.path.exists(f['path']):
                            pdf_files.append({"path": f['path'], "pages": None})
                        else:
                            st.error(f"⚠️ File not found: {f['name']}")
                    
                    if len(pdf_files) == 0:
                        st.error("❌ No valid PDF files found to merge")
                    elif len(pdf_files) < len(st.session_state.uploaded_files):
                        st.warning(f"⚠️ Only {len(pdf_files)} out of {len(st.session_state.uploaded_files)} files found")
                    
                    if len(pdf_files) > 0:
                        # Merge PDFs
                        output_path = os.path.join(UPLOAD_FOLDER, f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                        result = pdf_processor.merge_pdfs(pdf_files, output_path)
                        
                        if result.get('success'):
                            # Read merged PDF
                            with open(output_path, 'rb') as f:
                                st.session_state.merged_pdf = f.read()
                            st.success("✅ PDFs merged successfully!")
                        else:
                            st.error(f"❌ {result.get('error', 'Unknown error during merge')}")
                    
                except Exception as e:
                    st.error(f"❌ Error merging PDFs: {str(e)}")

    
    # Download merged PDF
    if st.session_state.merged_pdf:
        st.subheader("Step 3: Download Merged PDF")
        
        st.download_button(
            label="📥 Download Merged PDF",
            data=st.session_state.merged_pdf,
            file_name=f"merged_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
        st.info("✨ Your merged PDF is ready! Click the button above to download it.")

else:
    st.info("👆 Start by uploading one or more PDF files above!")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    <p>🚀 PDF Merger v1.0 | Made with Proud and Love</p>
</div>
""", unsafe_allow_html=True)
