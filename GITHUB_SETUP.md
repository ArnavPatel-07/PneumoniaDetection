# GitHub Repository Setup Guide

Your project is now ready to be uploaded to GitHub! Follow these steps:

## Step 1: Create a New Repository on GitHub

1. Go to [GitHub](https://github.com) and sign in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the repository details:
   - **Repository name**: `pneumonia-detection-app` (or your preferred name)
   - **Description**: "Medical-grade deep learning web application for detecting pneumonia from chest X-ray images"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

## Step 2: Connect Your Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these commands:

### Option A: If you haven't created the repository yet (recommended)

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/pneumonia-detection-app.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### Option B: If repository already exists

```bash
# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/pneumonia-detection-app.git

# Push to GitHub
git push -u origin main
```

## Step 3: Verify Upload

1. Refresh your GitHub repository page
2. You should see all your files:
   - README.md
   - LICENSE
   - TRAINING.md
   - backend/ folder
   - frontend/ folder
   - .gitignore

## Important Notes

### Model File Exclusion

The model file (`backend/models/pneumonia_resnet50_final.h5`) is **NOT** included in the repository because:
- It's too large for standard Git (>100MB)
- It's already in `.gitignore`

### To Include the Model File (Optional)

If you want to include the model file, you have two options:

#### Option 1: Use Git LFS (Large File Storage)

```bash
# Install Git LFS (if not already installed)
# Download from: https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Track .h5 files
git lfs track "*.h5"

# Add the .gitattributes file
git add .gitattributes

# Add the model file
git add backend/models/pneumonia_resnet50_final.h5

# Commit
git commit -m "Add model file using Git LFS"

# Push
git push origin main
```

#### Option 2: Upload Separately

Upload the model file separately to:
- Google Drive
- Dropbox
- AWS S3
- GitHub Releases (as an asset)

Then add a download link in your README.md

## Next Steps After Upload

1. **Add Topics/Tags** to your repository:
   - `machine-learning`
   - `deep-learning`
   - `medical-ai`
   - `pneumonia-detection`
   - `resnet50`
   - `fastapi`
   - `tensorflow`

2. **Enable GitHub Pages** (optional, for frontend demo):
   - Go to Settings → Pages
   - Source: Deploy from a branch
   - Branch: main, folder: /frontend
   - Save

3. **Add a GitHub Actions workflow** (optional):
   - Create `.github/workflows/ci.yml` for automated testing

4. **Create a Release**:
   - Go to Releases → Create a new release
   - Tag: v1.0.0
   - Title: "Initial Release"
   - Description: Copy from README features section

## Troubleshooting

### Authentication Issues

If you get authentication errors:

```bash
# Use GitHub CLI (recommended)
gh auth login

# Or use Personal Access Token
# 1. Go to GitHub Settings → Developer settings → Personal access tokens
# 2. Generate new token with repo permissions
# 3. Use token as password when pushing
```

### Large File Issues

If you accidentally try to commit large files:

```bash
# Remove from Git cache
git rm --cached backend/models/pneumonia_resnet50_final.h5

# Add to .gitignore (already done)
# Commit the removal
git commit -m "Remove large model file"

# Push
git push origin main
```

## Quick Command Reference

```bash
# Check remote
git remote -v

# Change remote URL
git remote set-url origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push changes
git push origin main

# Pull changes
git pull origin main

# View commit history
git log --oneline
```

---

**Your project is ready! Just follow Step 1 and Step 2 above to upload to GitHub.**
