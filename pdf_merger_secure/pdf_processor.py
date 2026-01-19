"""
PDF Processing Module for merging and sorting PDFs.
"""
import os
from pypdf import PdfWriter, PdfReader
from typing import List, Dict, Tuple


class PDFProcessor:
    """Handles PDF operations like merging and sorting."""
    
    def __init__(self):
        self.uploaded_files = {}
        
    def get_pdf_info(self, file_path: str) -> Dict:
        """
        Get information about a PDF file.
        Returns page count and basic metadata.
        """
        try:
            reader = PdfReader(file_path)
            return {
                "success": True,
                "page_count": len(reader.pages),
                "filename": os.path.basename(file_path)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def merge_pdfs(self, pdf_files: List[Dict], output_path: str) -> Dict:
        """
        Merge multiple PDFs with optional page selection.
        
        pdf_files format:
        [
            {
                "path": "/path/to/file.pdf",
                "pages": [0, 2, 5]  # specific pages (0-indexed), or None for all
            }
        ]
        """
        try:
            writer = PdfWriter()
            
            for pdf_info in pdf_files:
                file_path = pdf_info["path"]
                pages = pdf_info.get("pages", None)
                
                if not os.path.exists(file_path):
                    return {
                        "success": False,
                        "error": f"File not found: {file_path}"
                    }
                
                reader = PdfReader(file_path)
                total_pages = len(reader.pages)
                
                # If no specific pages selected, add all
                if pages is None or pages == "all":
                    pages = list(range(total_pages))
                
                # Validate page numbers
                for page_num in pages:
                    if not isinstance(page_num, int) or page_num < 0 or page_num >= total_pages:
                        return {
                            "success": False,
                            "error": f"Invalid page number {page_num} in {os.path.basename(file_path)}"
                        }
                    writer.add_page(reader.pages[page_num])
            
            # Write merged PDF
            with open(output_path, "wb") as output_file:
                writer.write(output_file)
            
            return {
                "success": True,
                "message": "PDF merged successfully",
                "output_file": output_path,
                "file_size": os.path.getsize(output_path)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error merging PDFs: {str(e)}"
            }
    
    def reorder_pages(self, file_path: str, page_order: List[int], output_path: str) -> Dict:
        """
        Reorder pages in a PDF file.
        
        page_order: List of page indices (0-based) in desired order
        """
        try:
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            reader = PdfReader(file_path)
            writer = PdfWriter()
            total_pages = len(reader.pages)
            
            # Validate page order
            if not all(isinstance(p, int) and 0 <= p < total_pages for p in page_order):
                return {
                    "success": False,
                    "error": "Invalid page order. All pages must be valid indices."
                }
            
            if len(page_order) != total_pages:
                return {
                    "success": False,
                    "error": f"Page order must contain all {total_pages} pages"
                }
            
            for page_num in page_order:
                writer.add_page(reader.pages[page_num])
            
            with open(output_path, "wb") as output_file:
                writer.write(output_file)
            
            return {
                "success": True,
                "message": "Pages reordered successfully",
                "output_file": output_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error reordering pages: {str(e)}"
            }


def create_pdf_processor():
    """Factory function to create PDF processor instance."""
    return PDFProcessor()
