# Deepfake Detector - Deployment Guide

## 🌍 Deploying to Hugging Face Spaces

This project is configured to be deployed easily to Hugging Face Spaces using Docker.

### 1. Create a New Space
1.  Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2.  **Name**: `deepfake-detector` (or anything you like).
3.  **License**: MIT (or your choice).
4.  **SDK**: Select **Docker**.
5.  **Template**: Leave as "Blank".
6.  **Visibility**: Public.

### 2. Upload Code
You can upload the code directly via the web interface or using Git.

**Using Git (Recommended):**
```bash
# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Copy all files from this project into the Space folder
cp -r Deepfake-detector/* YOUR_SPACE_NAME/

# Push to Hugging Face
cd YOUR_SPACE_NAME
git add .
git commit -m "Initial deploy"
git push
```

### 3. Configure Secrets (CRITICAL)
For the model to load, you need to provide your Hugging Face token.

1.  Go to your Space's **Settings** tab.
2.  Scroll to **Variables and secrets**.
3.  Click **New secret**.
    *   **Name**: `HUGGINGFACE_TOKEN`
    *   **Value**: Your Hugging Face Access Token (get it [here](https://huggingface.co/settings/tokens)).

### 4. Wait for Build
The Space will start building. It may take a few minutes to:
1.  Build the React frontend.
2.  Install Python dependencies.
3.  Download the model (on first run).

Once "Running", your app is live! 🚀

---

## 🐳 Running Locally with Docker

If you just want to run it on your machine:

1.  **Set your token:**
    ```bash
    # PowerShell
    $env:HUGGINGFACE_TOKEN="your_token_here"
    ```

2.  **Run:**
    ```bash
    docker-compose up --build
    ```

3.  **Open:** [http://localhost:3000](http://localhost:3000)
