# 🚀 Free Deployment Guide

This guide explains how to deploy the **Agentic Workflow Automation** project completely for free using **Vercel** (for the frontend) and **Render** (for the backend).

---

## 🏛️ Deployment Architecture
*   **Frontend**: Hosted on **Vercel** (Fast, global CDN, auto-deploys on push).
*   **Backend**: Hosted on **Render** (FastAPI app running inside the Docker container, auto-deploys on push).

---

## 🛠️ Step 1: Push your Code to GitHub

If you haven't already pushed your updated code containing the Dockerfiles and dynamic API configuration to your GitHub repository, do so by running this command in your local terminal:

```bash
git push -u origin main --force
```

---

## 🐍 Step 2: Deploy Backend to Render (Free)

Render offers a free tier for Web Services. It will automatically detect the `Dockerfile` inside your `backend/` directory.

### 📋 Instructions:
1.  Go to [Render.com](https://render.com/) and sign up / log in with your GitHub account.
2.  Click **New +** and select **Web Service**.
3.  Connect your GitHub repository `Shreena88/agentic-workflow-automation`.
4.  Configure the service settings:
    *   **Name**: `agentic-backend` (or any name you prefer)
    *   **Region**: Select the region closest to you.
    *   **Branch**: `main`
    *   **Root Directory**: `backend` (⚠️ **Important**: Set this so Render only builds the backend)
    *   **Runtime**: `Docker` (Render will automatically detect your `backend/Dockerfile`)
    *   **Instance Type**: `Free`
5.  Click on the **Advanced** button to add **Environment Variables**:
    *   `GROK_API_KEY`: *(Your Groq API key)*
    *   `GROK_BASE_URL`: `https://api.groq.com/openai/v1`
    *   `GROK_MODEL`: `llama-3.3-70b-versatile`
    *   `MAX_TASKS`: `5`
    *   `MEMORY_PERSIST_DIR`: `./memory`
    *   `ALLOWED_FILE_DIRS`: `./data`
6.  Click **Create Web Service**.
7.  Wait for the build and deploy process to complete. Once finished, copy the **live URL** (e.g., `https://agentic-backend.onrender.com`).

> [!WARNING]
> **Render Free Tier Constraints:**
> 1. **Spin Down**: If the backend receives no traffic for 15 minutes, the server will "spin down" (go to sleep). The next request will take about 50 seconds to boot the server back up.
> 2. **Ephemeral Disk**: The Render free tier does not support persistent disks. This means any uploaded files (under `./data/`) and FAISS session memories (under `./memory/`) will be deleted whenever the server restarts or wakes up from sleep. (To persist them, you'd need Render's paid tier with a disk volume, or to connect the app to a remote database/Vector DB like Pinecone).

---

## 💻 Step 3: Deploy Frontend to Vercel (Free)

Vercel provides a free tier for static frontend applications.

### 📋 Instructions:
1.  Go to [Vercel.com](https://vercel.com/) and sign up / log in with your GitHub account.
2.  Click **Add New...** and select **Project**.
3.  Import your GitHub repository `Shreena88/agentic-workflow-automation`.
4.  Configure the project:
    *   **Framework Preset**: `Vite` (Vercel will auto-detect Vite)
    *   **Root Directory**: `frontend` (⚠️ **Important**: Click *Edit* and select the `frontend` folder)
    *   **Build & Development Settings**: Keep defaults (Build command: `npm run build`, Output directory: `dist`)
5.  Open the **Environment Variables** section and add:
    *   **Key**: `VITE_API_BASE_URL`
    *   **Value**: *Your Render backend URL* (e.g., `https://agentic-backend.onrender.com`)
6.  Click **Deploy**.
7.  Vercel will build the frontend and provide you with a live URL (e.g., `https://agentic-workflow-automation.vercel.app`).

---

## 🧪 Step 4: Verify Deployment
1.  Open your Vercel frontend URL.
2.  Try uploading a CSV or running a query (e.g., "Hello, what tools can you run?").
3.  Verify the frontend receives responses from the Render backend.
