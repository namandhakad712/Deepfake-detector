# Deepfake Detector - Setup Guide

This project is a full-stack application for detecting deepfakes in videos using a custom Hugging Face model. It consists of a React frontend and a Flask backend.

## 🚀 Quick Start with Docker (Recommended)

The easiest way to run the project is using Docker. This ensures all dependencies (including system libraries for video processing) are correctly installed.

### Prerequisites
- [Docker](https://www.docker.com/products/docker-desktop/) installed on your machine.
- [Git](https://git-scm.com/downloads) installed.
- A **Hugging Face Token** (Read access). Get one [here](https://huggingface.co/settings/tokens).

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/namandhakad712/Deepfake-detector.git
    cd Deepfake-detector
    ```

2.  **Set up Environment Variables:**
    Create a `.env` file in the root directory (or just set the variable in your terminal) to store your Hugging Face token.
    
    **Option A: Create a .env file**
    Create a file named `.env` in the root folder and add:
    ```env
    HUGGINGFACE_TOKEN=your_hf_token_here
    ```

    **Option B: Pass it directly (Linux/Mac/PowerShell)**
    ```bash
    export HUGGINGFACE_TOKEN=your_hf_token_here
    ```

3.  **Run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```

4.  **Access the App:**
    - **Frontend:** Open [http://localhost:3000](http://localhost:3000) in your browser.
    - **Backend API:** Running at [http://localhost:5000](http://localhost:5000).

---

## 🛠️ Manual Installation (Without Docker)

If you prefer to run it manually, you'll need two terminal windows.

### Backend Setup

1.  Navigate to the backend folder:
    ```bash
    cd backend
    ```

2.  Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Create a `.env` file in the `backend` folder:
    ```env
    HUGGINGFACE_TOKEN=your_hf_token_here
    ```

5.  Run the server:
    ```bash
    python app.py
    ```

### Frontend Setup

1.  Navigate to the frontend folder:
    ```bash
    cd frontend
    ```

2.  Install dependencies:
    ```bash
    npm install
    ```

3.  Start the development server:
    ```bash
    npm start
    ```

4.  Open [http://localhost:3000](http://localhost:3000).

---

## 🏗️ Project Structure

- **frontend/**: React application (TypeScript, Styled Components).
- **backend/**: Flask API (PyTorch, Transformers, OpenCV).
- **docker-compose.yml**: Orchestrates the multi-container setup.

## ⚠️ Troubleshooting

- **Model Loading Error:** Ensure your `HUGGINGFACE_TOKEN` is valid and has read permissions. The first run will take some time to download the model (~500MB).
- **CORS Errors:** If the frontend cannot talk to the backend, ensure both are running and the backend allows requests from `http://localhost:3000`.
