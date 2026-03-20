# AQI Web App — Deployment Guide (Free on Render.com)

## What's in this folder

```
aqi-web-app/
├── backend/
│   ├── main.py          ← FastAPI app (the actual server)
│   ├── config.py        ← Path to your AQI-prediction---INDIA folder
│   └── requirements.txt ← Python dependencies
├── frontend/
│   └── index.html       ← The website (served by the backend)
└── Procfile             ← Tells Render how to start the server
```

---

## Step 1 — Test Locally First

```bash
# In your terminal, activate the venv and run:
cd /home/samantdev/Desktop/finalPro/aqi-web-app/backend
/home/samantdev/Desktop/finalPro/AQI-prediction---INDIA/hello/bin/uvicorn main:app --reload --port 8000
```

Open your browser at: **http://localhost:8000**

---

## Step 2 — Put Everything on GitHub

Render.com deploys directly from a GitHub repo.

1. **Create a new GitHub repo** (e.g., `aqi-prediction-app`)
2. Push the following two folders together:
   - `aqi-web-app/` (the new folder we built)
   - `AQI-prediction---INDIA/` (your models, metrics, post-process files)

```bash
# From your finalPro folder:
git init aqi-deploy
cd aqi-deploy
cp -r ../aqi-web-app .
cp -r ../AQI-prediction---INDIA .
git add .
git commit -m "Initial AQI web app"
git remote add origin https://github.com/YOUR_USERNAME/aqi-prediction-app.git
git push -u origin main
```

> ⚠️ The `knn.pkl` model file is ~200 MB. Add it to `.gitignore` — it doesn't affect the   
> prediction since only XGBoost + LightGBM fine-tuned models are used for prediction.

```bash
# Create .gitignore in aqi-deploy/
echo "AQI-prediction---INDIA/models/knn.pkl" >> .gitignore
```

---

## Step 3 — Update config.py for the Server

On Render, the folder structure will be different. Update `backend/config.py`:

```python
import os

MODEL_BASE = os.environ.get(
    "MODEL_BASE",
    "/opt/render/project/src/AQI-prediction---INDIA"  # Render's default path
)
```

Render sets `$MODEL_BASE` from environment variables — you can set this in the dashboard.

---

## Step 4 — Deploy on Render.com (Free)

1. Go to **https://render.com** → Sign up (free)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repo
4. Fill in:
   - **Name**: `aqi-prediction-app`
   - **Root Directory**: `aqi-web-app` ← important!
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Under **Environment Variables**, add:
   - Key: `MODEL_BASE`
   - Value: `/opt/render/project/src/AQI-prediction---INDIA`
6. Click **"Create Web Service"**

Render will build and deploy. You'll get a URL like:  
`https://aqi-prediction-app.onrender.com`

---

## Step 5 — Connect blockify.in

Once your Render app is running:

1. Go to your **domain registrar** (wherever you manage `blockify.in`)
2. Add a **CNAME record**:
   - **Name**: `aqi` (creates `aqi.blockify.in`)
   - **Value**: `aqi-prediction-app.onrender.com`
3. In Render → Settings → **Custom Domains** → add `aqi.blockify.in`

Your app will be live at: **https://aqi.blockify.in** ✅

> 💡 If you want it at the root domain `blockify.in` instead of a subdomain,  
> you'll need to change your domain's A record to Render's IP — Render provides this.

---

## Free Tier Note

Render's free tier **spins down after 15 minutes of inactivity** and takes ~30 seconds to  
wake up on the next request. For a college project/demo this is perfectly fine.  
For always-on, upgrade to Render's Starter plan (~$7/month).
