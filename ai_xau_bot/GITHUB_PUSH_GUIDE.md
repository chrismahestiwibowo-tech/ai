# GitHub Push Guide

## Repository Details
- **Repository URL:** https://github.com/chrismahestiwibowo-tech/ai.git
- **Target Subfolder:** ai_aux_bot
- **Username:** chrismahestiwibowo-tech
- **Email:** chrismahestiwibowo.ae@gmail.com

---

## Option 1: Install Git and Push via Command Line (Recommended)

### Step 1: Install Git
Download and install Git for Windows from: https://git-scm.com/download/win

### Step 2: Configure Git
After installation, open a new terminal and run:
```bash
git config --global user.name "chrismahestiwibowo-tech"
git config --global user.email "chrismahestiwibowo.ae@gmail.com"
```

### Step 3: Clone Your Repository
```bash
cd D:\chrisma\ai-botbos
git clone https://github.com/chrismahestiwibowo-tech/ai.git
```

### Step 4: Create ai_aux_bot Folder (if it doesn't exist)
```bash
cd ai
mkdir -p ai_aux_bot
```

### Step 5: Copy Project Files
```powershell
# Copy all files from current project to ai_aux_bot folder
Copy-Item -Path "D:\chrisma\ai-botbos\20260107_ai_xauusd_bot\*" -Destination "D:\chrisma\ai-botbos\ai\ai_aux_bot\" -Recurse -Exclude "venv"
```

### Step 6: Commit and Push
```bash
cd D:\chrisma\ai-botbos\ai
git add ai_aux_bot/
git commit -m "Add XAU/USD AI prediction bot with XGBoost and Monte Carlo simulation"
git push origin main
```

*Note: If the default branch is 'master' instead of 'main', use `git push origin master`*

---

## Option 2: Use GitHub Desktop (Easier for Beginners)

### Step 1: Install GitHub Desktop
Download from: https://desktop.github.com/

### Step 2: Sign In
Sign in with your GitHub account in GitHub Desktop

### Step 3: Clone Repository
1. Click "File" → "Clone Repository"
2. Enter: `chrismahestiwibowo-tech/ai`
3. Choose location: `D:\chrisma\ai-botbos\ai`
4. Click "Clone"

### Step 4: Copy Files
1. Open File Explorer
2. Navigate to `D:\chrisma\ai-botbos\ai`
3. Create folder `ai_aux_bot` if it doesn't exist
4. Copy all files from `D:\chrisma\ai-botbos\20260107_ai_xauusd_bot\` to `ai_aux_bot\`
   - **IMPORTANT:** Do NOT copy the `venv` folder

### Step 5: Commit and Push
1. GitHub Desktop will show all changes
2. In the summary field, type: "Add XAU/USD AI prediction bot"
3. In the description field, add details about the project
4. Click "Commit to main" (or master)
5. Click "Push origin"

---

## Option 3: Upload via GitHub Web Interface

### Step 1: Navigate to Repository
Go to: https://github.com/chrismahestiwibowo-tech/ai

### Step 2: Create Folder
1. Click "Add file" → "Create new file"
2. Type: `ai_aux_bot/README.md`
3. Add a simple description
4. Click "Commit new file"

### Step 3: Upload Files
1. Navigate to the `ai_aux_bot` folder in your repo
2. Click "Add file" → "Upload files"
3. Drag and drop all files from `D:\chrisma\ai-botbos\20260107_ai_xauusd_bot\`
   - **EXCLUDE:** venv folder
4. Add commit message: "Add XAU/USD AI prediction bot with XGBoost and Monte Carlo"
5. Click "Commit changes"

---

## Files to Upload

Make sure to upload these files:
- ✅ config.py
- ✅ data_loader.py
- ✅ indicators.py
- ✅ model.py
- ✅ monte_carlo.py
- ✅ main.py
- ✅ requirements.txt
- ✅ README.md
- ✅ QUICKSTART.py
- ✅ .env.example
- ❌ venv/ (DO NOT UPLOAD - exclude virtual environment)

---

## Alternative: PowerShell Script (If Git is Installed)

Save this as `push_to_github.ps1` and run it:

```powershell
# Navigate to parent directory
cd D:\chrisma\ai-botbos

# Clone repository if not exists
if (-not (Test-Path "ai")) {
    git clone https://github.com/chrismahestiwibowo-tech/ai.git
}

# Navigate to repository
cd ai

# Pull latest changes
git pull origin main

# Create ai_aux_bot folder if not exists
if (-not (Test-Path "ai_aux_bot")) {
    New-Item -ItemType Directory -Path "ai_aux_bot"
}

# Copy files (excluding venv)
$source = "D:\chrisma\ai-botbos\20260107_ai_xauusd_bot"
$destination = "D:\chrisma\ai-botbos\ai\ai_aux_bot"

Get-ChildItem -Path $source | Where-Object { $_.Name -ne 'venv' } | Copy-Item -Destination $destination -Recurse -Force

# Add, commit, and push
git add ai_aux_bot/
git commit -m "Add XAU/USD AI prediction bot with XGBoost and Monte Carlo simulation

Features:
- Fetches 3 years of XAU/USD data from MetaTrader5
- Calculates 11 technical indicators
- XGBoost regression model with 80/20 train-test split
- Monte Carlo simulation with 10,000 paths
- Comprehensive evaluation metrics (MSE, RMSE, MAE, R²)
- Automated report generation and visualization"

git push origin main

Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
```

---

## Troubleshooting

### If you get authentication errors:
1. Generate a Personal Access Token on GitHub:
   - Go to: Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Click "Generate new token"
   - Select scopes: `repo`
   - Copy the token

2. When prompted for password, use the token instead

### If branch is 'master' instead of 'main':
Replace `main` with `master` in all commands above

### If you need to update existing files:
The commands above will overwrite existing files in `ai_aux_bot` folder

---

## Quick Command Reference

After Git is installed and repository is cloned:

```bash
# One-time setup
cd D:\chrisma\ai-botbos\ai
git config user.name "chrismahestiwibowo-tech"
git config user.email "chrismahestiwibowo.ae@gmail.com"

# Copy files (PowerShell)
Copy-Item -Path "D:\chrisma\ai-botbos\20260107_ai_xauusd_bot\*" -Destination "D:\chrisma\ai-botbos\ai\ai_aux_bot\" -Recurse -Exclude "venv"

# Add, commit, push (Git Bash or PowerShell after Git installation)
cd D:\chrisma\ai-botbos\ai
git add ai_aux_bot/
git commit -m "Add XAU/USD AI prediction bot"
git push origin main
```

---

**Recommended:** Option 1 (Command Line) or Option 2 (GitHub Desktop) for best control and ease of use.
