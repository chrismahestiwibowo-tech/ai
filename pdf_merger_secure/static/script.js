// Global state
let appState = {
    sessionId: null,
    files: [],
    mergeOrder: [],
    pageSelections: {},
    mergedFile: null
};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
});

// Initialize application
async function initializeApp() {
    try {
        const response = await fetch('/api/session/new', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            appState.sessionId = data.session_id;
        }
    } catch (error) {
        showError('Failed to initialize application');
    }
}

// Setup all event listeners
function setupEventListeners() {
    // Upload area
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');

    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        handleDrop(e);
    });

    // Action buttons
    document.getElementById('resetBtn').addEventListener('click', resetApp);
    document.getElementById('mergeBtn').addEventListener('click', mergePdfs);
    document.getElementById('downloadBtn').addEventListener('click', downloadPdf);
    document.getElementById('startAgainBtn').addEventListener('click', resetApp);
    document.getElementById('errorRetryBtn').addEventListener('click', resetApp);

    // Modal
    document.getElementById('closeModal').addEventListener('click', closeModal);
    document.getElementById('cancelModal').addEventListener('click', closeModal);
    document.getElementById('confirmPageSelection').addEventListener('click', confirmPageSelection);

    // Page option radio buttons
    document.querySelectorAll('input[name="pageOption"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            const customInput = document.getElementById('customPageInput');
            customInput.style.display = e.target.value === 'custom' ? 'block' : 'none';
        });
    });
}

// Handle file selection
async function handleFileSelect(e) {
    const files = e.target.files;
    await uploadFiles(Array.from(files));
    e.target.value = ''; // Reset input
}

// Handle drag and drop
async function handleDrop(e) {
    const files = e.dataTransfer.files;
    await uploadFiles(Array.from(files));
}

// Upload files
async function uploadFiles(files) {
    if (!appState.sessionId) {
        showError('Session not initialized');
        return;
    }

    const validFiles = files.filter(file => {
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            showError(`${file.name} is not a PDF file`);
            return false;
        }
        if (file.size > 50 * 1024 * 1024) {
            showError(`${file.name} exceeds 50 MB limit`);
            return false;
        }
        return true;
    });

    if (validFiles.length === 0) return;

    if (appState.files.length + validFiles.length > 50) {
        showError('Maximum 50 files allowed');
        return;
    }

    // Show progress
    document.getElementById('uploadProgress').style.display = 'block';
    document.getElementById('uploadArea').style.opacity = '0.5';
    document.getElementById('browseBtn').disabled = true;

    let uploadedCount = 0;
    for (const file of validFiles) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('session_id', appState.sessionId);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                appState.files.push(data.file);
                appState.mergeOrder.push(data.file.id);
            } else {
                showError(data.error || 'Upload failed');
            }
        } catch (error) {
            showError(`Error uploading ${file.name}`);
        }

        uploadedCount++;
    }

    // Hide progress and update UI
    document.getElementById('uploadProgress').style.display = 'none';
    document.getElementById('uploadArea').style.opacity = '1';
    document.getElementById('browseBtn').disabled = false;

    updateFileList();
    updateUI();
}

// Update file list display
function updateFileList() {
    const fileList = document.getElementById('fileList');
    const fileCount = document.getElementById('fileCount');
    const filesContainer = document.getElementById('filesContainer');

    if (appState.files.length === 0) {
        fileList.style.display = 'none';
        return;
    }

    fileCount.textContent = appState.files.length;
    filesContainer.innerHTML = '';

    appState.files.forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-info">
                <div class="file-name">${escapeHtml(file.name)}</div>
                <div class="file-pages">${file.pages} page${file.pages !== 1 ? 's' : ''}</div>
            </div>
            <div class="file-actions">
                <button class="btn-remove" onclick="removeFile('${file.id}')">Remove</button>
            </div>
        `;
        filesContainer.appendChild(fileItem);
    });

    fileList.style.display = 'block';
}

// Remove file
async function removeFile(fileId) {
    try {
        const response = await fetch('/api/remove-file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: appState.sessionId,
                file_id: fileId
            })
        });

        const data = await response.json();
        if (data.success) {
            appState.files = appState.files.filter(f => f.id !== fileId);
            appState.mergeOrder = appState.mergeOrder.filter(id => id !== fileId);
            delete appState.pageSelections[fileId];
            updateFileList();
            updateUI();
        }
    } catch (error) {
        showError('Error removing file');
    }
}

// Update UI visibility
function updateUI() {
    const sortingSection = document.getElementById('sortingSection');
    const actionsSection = document.getElementById('actionsSection');

    if (appState.files.length > 0) {
        sortingSection.style.display = 'block';
        actionsSection.style.display = 'block';
        updateSortableList();
        updatePreview();
    } else {
        sortingSection.style.display = 'none';
        actionsSection.style.display = 'none';
    }
}

// Update sortable list
function updateSortableList() {
    const sortableList = document.getElementById('sortableList');
    sortableList.innerHTML = '';

    appState.mergeOrder.forEach((fileId, index) => {
        const file = appState.files.find(f => f.id === fileId);
        if (!file) return;

        const pageSelection = appState.pageSelections[fileId];
        let pageText = 'All pages';
        if (pageSelection && pageSelection.pages !== 'all') {
            pageText = `Pages: ${pageSelection.pages.join(', ')}`;
        }

        const item = document.createElement('div');
        item.className = 'sortable-item';
        item.draggable = true;
        item.dataset.fileId = fileId;
        item.innerHTML = `
            <div class="drag-handle">⋮⋮</div>
            <div class="sortable-item-content">
                <div class="sortable-item-name">${index + 1}. ${escapeHtml(file.name)}</div>
                <div class="sortable-item-pages">${pageText}</div>
            </div>
            <div class="page-selector">
                <button class="btn-select-pages" onclick="openPageModal('${fileId}', ${file.pages})">Select Pages</button>
            </div>
        `;

        // Drag events
        item.addEventListener('dragstart', handleDragStart);
        item.addEventListener('dragend', handleDragEnd);
        item.addEventListener('dragover', handleDragOver);
        item.addEventListener('drop', handleDrop);

        sortableList.appendChild(item);
    });
}

// Drag and drop handlers
let draggedElement = null;

function handleDragStart(e) {
    draggedElement = this;
    this.classList.add('dragging');
}

function handleDragEnd(e) {
    this.classList.remove('dragging');
    draggedElement = null;
}

function handleDragOver(e) {
    if (!draggedElement || draggedElement === this) return;
    e.preventDefault();

    const allItems = document.querySelectorAll('.sortable-item');
    const afterElement = getDragAfterElement(e.clientY);

    const sortableList = document.getElementById('sortableList');
    if (afterElement == null) {
        sortableList.appendChild(draggedElement);
    } else {
        sortableList.insertBefore(draggedElement, afterElement);
    }

    // Update merge order
    const newOrder = Array.from(document.querySelectorAll('.sortable-item'))
        .map(item => item.dataset.fileId);
    appState.mergeOrder = newOrder;
}

function getDragAfterElement(y) {
    const draggableElements = [...document.querySelectorAll('.sortable-item:not(.dragging)')];

    return draggableElements.reduce((closest, child) => {
        const box = child.getBoundingClientRect();
        const offset = y - box.top - box.height / 2;

        if (offset < 0 && offset > closest.offset) {
            return { offset: offset, element: child };
        } else {
            return closest;
        }
    }, { offset: Number.NEGATIVE_INFINITY }).element;
}

// Open page selection modal
function openPageModal(fileId, totalPages) {
    const file = appState.files.find(f => f.id === fileId);
    if (!file) return;

    document.getElementById('modalTitle').textContent = `Select Pages - ${escapeHtml(file.name)}`;
    document.getElementById('totalPagesInfo').innerHTML = `
        <strong>Total pages:</strong> ${totalPages}
    `;

    // Set current selection
    const currentSelection = appState.pageSelections[fileId];
    if (currentSelection && currentSelection.pages !== 'all') {
        document.querySelector('input[value="custom"]').checked = true;
        document.getElementById('customPageInput').style.display = 'block';
        document.getElementById('pageNumbers').value = currentSelection.pages.join(',');
    } else {
        document.querySelector('input[value="all"]').checked = true;
        document.getElementById('customPageInput').style.display = 'none';
    }

    appState.currentFileId = fileId;
    appState.currentTotalPages = totalPages;

    document.getElementById('pageModal').style.display = 'flex';
}

function closeModal() {
    document.getElementById('pageModal').style.display = 'none';
}

function confirmPageSelection() {
    const fileId = appState.currentFileId;
    const totalPages = appState.currentTotalPages;

    const option = document.querySelector('input[name="pageOption"]:checked').value;

    if (option === 'all') {
        appState.pageSelections[fileId] = { pages: 'all' };
    } else {
        const input = document.getElementById('pageNumbers').value.trim();

        if (!input) {
            showError('Please enter page numbers');
            return;
        }

        const pages = parsePageNumbers(input, totalPages);
        if (pages === null) {
            showError('Invalid page format. Use: 1,3,5-7');
            return;
        }

        appState.pageSelections[fileId] = { pages: pages };
    }

    closeModal();
    updateSortableList();
    updatePreview();
}

function parsePageNumbers(input, totalPages) {
    const pages = new Set();

    const parts = input.split(',').map(p => p.trim());

    for (const part of parts) {
        if (part.includes('-')) {
            const [start, end] = part.split('-').map(p => parseInt(p.trim()) - 1);

            if (isNaN(start) || isNaN(end) || start < 0 || end >= totalPages || start > end) {
                return null;
            }

            for (let i = start; i <= end; i++) {
                pages.add(i);
            }
        } else {
            const page = parseInt(part) - 1;

            if (isNaN(page) || page < 0 || page >= totalPages) {
                return null;
            }

            pages.add(page);
        }
    }

    return Array.from(pages).sort((a, b) => a - b);
}

// Update preview
function updatePreview() {
    const preview = document.getElementById('mergePreview');
    preview.innerHTML = '';

    let pageCount = 0;

    appState.mergeOrder.forEach((fileId, index) => {
        const file = appState.files.find(f => f.id === fileId);
        if (!file) return;

        const pageSelection = appState.pageSelections[fileId];
        let pages = [];

        if (!pageSelection || pageSelection.pages === 'all') {
            pages = Array.from({ length: file.pages }, (_, i) => i + 1);
        } else {
            pages = pageSelection.pages.map(p => p + 1);
        }

        const previewItem = document.createElement('div');
        previewItem.className = 'preview-item';
        previewItem.innerHTML = `
            <span class="preview-item-number">${index + 1}.</span>
            ${escapeHtml(file.name)} (${pages.length} page${pages.length !== 1 ? 's' : ''})
        `;
        preview.appendChild(previewItem);

        pageCount += pages.length;
    });

    const summary = document.createElement('div');
    summary.className = 'preview-item';
    summary.style.background = '#f0f4ff';
    summary.style.marginTop = '15px';
    summary.innerHTML = `
        <strong>Total pages in merged file: ${pageCount}</strong>
    `;
    preview.appendChild(summary);
}

// Merge PDFs
async function mergePdfs() {
    if (appState.mergeOrder.length === 0) {
        showError('No files selected for merging');
        return;
    }

    // Get custom filename
    let outputFilename = document.getElementById('outputFilename').value.trim();
    if (!outputFilename) {
        showError('Please enter a file name for the merged PDF');
        return;
    }

    // Sanitize filename (remove .pdf if user added it, remove special characters)
    outputFilename = outputFilename.replace(/\.pdf$/i, '').replace(/[<>:"|?*\/\\]/g, '_');

    if (outputFilename.length === 0) {
        showError('Invalid file name');
        return;
    }

    document.getElementById('loadingSpinner').style.display = 'flex';

    try {
        const config = appState.mergeOrder.map(fileId => {
            const pageSelection = appState.pageSelections[fileId];
            let pages = 'all';

            if (pageSelection && pageSelection.pages !== 'all') {
                pages = pageSelection.pages;
            }

            return {
                file_id: fileId,
                pages: pages
            };
        });

        const response = await fetch('/api/merge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: appState.sessionId,
                config: config,
                output_filename: outputFilename
            })
        });

        const data = await response.json();

        document.getElementById('loadingSpinner').style.display = 'none';

        if (data.success) {
            appState.mergedFile = data.file;
            showResultSuccess('PDF merged successfully!');
        } else {
            showError(data.error || 'Merge failed');
        }
    } catch (error) {
        document.getElementById('loadingSpinner').style.display = 'none';
        showError('Error merging PDFs: ' + error.message);
    }
}

// Download merged PDF
function downloadPdf() {
    if (!appState.mergedFile) {
        showError('No merged file available');
        return;
    }

    window.location.href = `/api/download/${appState.mergedFile}`;
}

// Show result success
function showResultSuccess(message) {
    document.getElementById('errorSection').style.display = 'none';
    document.getElementById('uploadArea').parentElement.style.display = 'none';
    document.getElementById('sortingSection').style.display = 'none';
    document.getElementById('actionsSection').style.display = 'none';

    document.getElementById('resultMessage').textContent = message;
    document.getElementById('resultSection').style.display = 'block';

    window.scrollTo(0, 0);
}

// Show error
function showError(message) {
    document.getElementById('resultSection').style.display = 'none';

    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorSection').style.display = 'block';

    window.scrollTo(0, 0);
}

// Reset app
async function resetApp() {
    try {
        await fetch('/api/cleanup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: appState.sessionId })
        });
    } catch (error) {
        // Ignore cleanup errors
    }

    // Reset state
    appState = {
        sessionId: null,
        files: [],
        mergeOrder: [],
        pageSelections: {},
        mergedFile: null
    };

    // Hide sections
    document.getElementById('fileList').style.display = 'none';
    document.getElementById('sortingSection').style.display = 'none';
    document.getElementById('actionsSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
    document.getElementById('uploadArea').parentElement.style.display = 'block';

    // Show upload area
    document.getElementById('uploadArea').style.opacity = '1';
    document.getElementById('uploadArea').style.display = 'block';

    // Reinitialize
    initializeApp();

    window.scrollTo(0, 0);
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
