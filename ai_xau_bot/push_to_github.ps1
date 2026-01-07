# PowerShell Script to Push XAU/USD Bot to GitHub
# Run this script after installing Git

# Configuration
$repoUrl = "https://github.com/chrismahestiwibowo-tech/ai.git"
$userName = "chrismahestiwibowo-tech"
$userEmail = "chrismahestiwibowo.ae@gmail.com"
$targetFolder = "ai_aux_bot"
$sourceFolder = "D:\chrisma\ai-botbos\20260107_ai_xauusd_bot"
$parentDir = "D:\chrisma\ai-botbos"
$repoDir = "D:\chrisma\ai-botbos\ai"

Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "GitHub Push Script - XAU/USD AI Bot" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# Check if Git is installed
try {
    $gitVersion = git --version 2>&1
    Write-Host "`n✓ Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "`n✗ Git is not installed!" -ForegroundColor Red
    Write-Host "`nPlease install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "After installation, restart PowerShell and run this script again.`n" -ForegroundColor Yellow
    exit 1
}

# Navigate to parent directory
Write-Host "`nNavigating to parent directory..." -ForegroundColor Yellow
Set-Location $parentDir

# Configure Git
Write-Host "`nConfiguring Git user..." -ForegroundColor Yellow
git config --global user.name $userName
git config --global user.email $userEmail
Write-Host "✓ Git configured for $userName <$userEmail>" -ForegroundColor Green

# Clone repository if it doesn't exist
if (-not (Test-Path $repoDir)) {
    Write-Host "`nCloning repository from GitHub..." -ForegroundColor Yellow
    git clone $repoUrl
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n✗ Failed to clone repository!" -ForegroundColor Red
        Write-Host "Please check your internet connection and repository URL.`n" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "✓ Repository cloned successfully" -ForegroundColor Green
} else {
    Write-Host "`n✓ Repository already exists locally" -ForegroundColor Green
}

# Navigate to repository
Set-Location $repoDir

# Pull latest changes
Write-Host "`nPulling latest changes from GitHub..." -ForegroundColor Yellow
git pull origin main 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    # Try master branch if main doesn't exist
    Write-Host "Trying 'master' branch..." -ForegroundColor Yellow
    git pull origin master 2>&1 | Out-Null
}
Write-Host "✓ Repository is up to date" -ForegroundColor Green

# Create target folder if it doesn't exist
$targetPath = Join-Path $repoDir $targetFolder
if (-not (Test-Path $targetPath)) {
    Write-Host "`nCreating $targetFolder directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $targetPath | Out-Null
    Write-Host "✓ Directory created" -ForegroundColor Green
} else {
    Write-Host "`n✓ Target directory already exists" -ForegroundColor Green
}

# Copy files (excluding venv)
Write-Host "`nCopying project files..." -ForegroundColor Yellow
Write-Host "Source: $sourceFolder" -ForegroundColor Gray
Write-Host "Destination: $targetPath" -ForegroundColor Gray

$itemsToCopy = Get-ChildItem -Path $sourceFolder | Where-Object { 
    $_.Name -ne 'venv' -and 
    $_.Name -ne 'models' -and 
    $_.Name -ne 'plots' -and 
    $_.Name -ne 'results' -and
    $_.Name -ne '__pycache__' -and
    $_.Name -ne '.env'
}

foreach ($item in $itemsToCopy) {
    Copy-Item -Path $item.FullName -Destination $targetPath -Recurse -Force
    Write-Host "  ✓ Copied: $($item.Name)" -ForegroundColor White
}

Write-Host "✓ All files copied successfully" -ForegroundColor Green

# Check Git status
Write-Host "`nChecking Git status..." -ForegroundColor Yellow
$status = git status --short

if ($status) {
    Write-Host "Changes detected:" -ForegroundColor White
    git status --short
    
    # Add files
    Write-Host "`nAdding files to Git..." -ForegroundColor Yellow
    git add "$targetFolder/"
    Write-Host "✓ Files staged for commit" -ForegroundColor Green
    
    # Commit
    Write-Host "`nCreating commit..." -ForegroundColor Yellow
    $commitMessage = @"
Add XAU/USD AI prediction bot with XGBoost and Monte Carlo simulation

Features:
- Fetches 3 years of XAU/USD data from MetaTrader5
- Calculates 11 technical indicators (RSI, MACD, BB, ATR, SMA, EMA, etc.)
- XGBoost regression model with 80/20 train-test split
- Monte Carlo simulation with 10,000 paths for probabilistic forecasting
- Comprehensive evaluation metrics (MSE, RMSE, MAE, R²)
- Automated report generation and visualization
- Modular architecture with separate modules for data, features, model, and simulation
"@
    
    git commit -m $commitMessage
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Commit created successfully" -ForegroundColor Green
        
        # Push to GitHub
        Write-Host "`nPushing to GitHub..." -ForegroundColor Yellow
        Write-Host "This may take a moment..." -ForegroundColor Gray
        
        # Try main branch first
        git push origin main 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            # Try master branch if main doesn't exist
            Write-Host "Trying 'master' branch..." -ForegroundColor Yellow
            git push origin master
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n=================================================================" -ForegroundColor Green
            Write-Host "SUCCESS! Project pushed to GitHub" -ForegroundColor Green
            Write-Host "=================================================================" -ForegroundColor Green
            Write-Host "`nRepository: https://github.com/$userName/ai" -ForegroundColor Cyan
            Write-Host "Project Location: $targetFolder/" -ForegroundColor Cyan
            Write-Host "`n✓ All done!`n" -ForegroundColor Green
        } else {
            Write-Host "`n✗ Failed to push to GitHub!" -ForegroundColor Red
            Write-Host "`nPossible reasons:" -ForegroundColor Yellow
            Write-Host "1. Authentication required - you may need a Personal Access Token" -ForegroundColor White
            Write-Host "2. No internet connection" -ForegroundColor White
            Write-Host "3. Repository permissions issue" -ForegroundColor White
            Write-Host "`nFor authentication, generate a token at:" -ForegroundColor Yellow
            Write-Host "https://github.com/settings/tokens`n" -ForegroundColor Cyan
        }
    } else {
        Write-Host "`n✗ Commit failed!" -ForegroundColor Red
    }
} else {
    Write-Host "✓ No changes detected - files are already up to date" -ForegroundColor Green
}

Write-Host "`nScript completed.`n" -ForegroundColor Cyan
