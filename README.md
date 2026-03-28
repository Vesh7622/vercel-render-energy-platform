# Vercel Frontend + Render Backend Starter

This starter gives you:

- `frontend/` for **Vercel**
- `backend/` for **Render**

## Local development

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn energy_platform_backend:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

The frontend reads the backend URL from:

- `frontend/.env.local`

## Vercel deployment

Deploy only the `frontend/` folder to Vercel.

Set this environment variable in Vercel:

- `VITE_API_BASE_URL=https://your-render-backend.onrender.com`

## Render deployment

Deploy only the `backend/` folder to Render as a Python web service.

Use:

- Build command: `pip install -r requirements.txt`
- Start command: `python -m uvicorn energy_platform_backend:app --host 0.0.0.0 --port $PORT`

## Note

This starter uses SQLite for simplicity. For production, move to PostgreSQL.
