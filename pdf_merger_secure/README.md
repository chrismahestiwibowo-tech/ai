# PDF Merger Bot

A user-friendly web application for merging and reordering PDF files with an intuitive interface.

## Features

✅ **Drag & Drop Upload**: Easily upload multiple PDF files  
✅ **Drag to Reorder**: Rearrange the merge order by dragging files  
✅ **Page Selection**: Choose specific pages from each PDF  
✅ **Page Range Support**: Select ranges like "1-5,8,10"  
✅ **Live Preview**: See what your merged PDF will look like  
✅ **Fool-Proof UI**: Clear guidance at each step  
✅ **Batch Processing**: Merge up to 50 files at once  
✅ **File Size Limit**: 50 MB per file  

## Installation

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)

### Setup

1. Navigate to the project directory:
```bash
cd pdf_agent\pdf_merger
```

2. Create and activate virtual environment (if not already done):
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # On Windows PowerShell
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and go to:
```
http://127.0.0.1:5000
```

## Usage

### Step 1: Upload PDF Files
- Drag and drop PDF files into the upload area, or
- Click "Choose Files" to browse and select PDFs
- Maximum 50 files, 50 MB each

### Step 2: Arrange & Select Pages
- **Reorder**: Drag the drag handles (⋮⋮) to change merge order
- **Select Pages**: Click "Select Pages" to choose specific pages
  - Select "All Pages" to include the entire PDF
  - Select "Custom Selection" to pick specific pages
  - Use formats like: `1,3,5-7,10`

### Step 3: Review & Merge
- Review the merge preview showing:
  - File order
  - Selected pages from each file
  - Total page count
- Click "Merge PDFs" to create the merged file

### Step 4: Download
- Download your merged PDF
- Click "Start New Merge" to merge another set of files

## Features Explained

### Fool-Proof UI Elements
1. **Clear Step Indicators**: Each section shows which step you're on
2. **Visual Feedback**: Files highlight when hovered, drag indicators show
3. **Hints and Tooltips**: Every action has guidance text
4. **Validation**: Invalid page numbers are caught with helpful error messages
5. **Progress Indicators**: Loading spinner shows when processing
6. **Success/Error States**: Clear visual feedback on actions

### Security
- Files are stored in a temporary uploads folder
- Files are deleted after download/session cleanup
- Filenames are sanitized
- Path traversal is prevented

## File Structure

```
pdf_merger/
├── app.py                    # Flask application
├── pdf_processor.py          # PDF processing logic
├── requirements.txt          # Dependencies
├── templates/
│   └── index.html           # HTML template
├── static/
│   ├── style.css            # Styling
│   └── script.js            # Frontend logic
└── uploads/                 # Temporary upload folder
```

## API Endpoints

- `POST /api/session/new` - Create new session
- `POST /api/upload` - Upload PDF file
- `GET /api/files` - Get uploaded files
- `POST /api/remove-file` - Remove uploaded file
- `POST /api/merge` - Merge PDFs
- `GET /api/download/<filename>` - Download merged PDF
- `POST /api/cleanup` - Clean up session

## Troubleshooting

### "Only PDF files are allowed"
- Ensure the file is a valid PDF format
- Check file extension is `.pdf`

### "Invalid page number"
- Page numbers must be 1-based (not 0-based)
- Example: For a 10-page PDF, use pages 1-10, not 0-9

### Port already in use
- Change the port in `app.py`:
```python
app.run(debug=True, host='127.0.0.1', port=5001)
```

## Performance

- Single file merge: < 1 second
- 10 files merge: 2-5 seconds (depending on file sizes)
- Uploads and merging are reasonably fast for typical PDFs

## Limitations

- Maximum 50 files per merge session
- Maximum 50 MB per file
- Works best with PDFs that have text/image layers
- Scanned PDFs (image-only) are supported but page extraction may vary

## Future Enhancements

- PDF compression options
- Batch processing mode
- History of recent merges
- Page extraction preview
- Rotation and deletion of pages
- PDF splitting functionality

## License

This project is provided as-is for personal and educational use.

## Support

For issues or questions, check that:
1. Flask is running on localhost:5000
2. All required packages are installed
3. Browser supports modern JavaScript (ES6+)
4. Sufficient disk space for temporary files
