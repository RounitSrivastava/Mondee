# API Server

## Run locally

```bash
pip install -r requirements.api.txt
python api/server.py
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Check index load status |
| `POST` | `/recommend` | Get recommendations for a persona |

## Example requests

```bash
# Health check
curl http://localhost:5000/health

# Recommend
curl -X POST http://localhost:5000/recommend \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "u_001",
       "persona": "truck_driver",
       "signals": ["truck", "gps"],
       "query_string": "truck gps commercial vehicle navigation",
       "top_k": 10
     }'
```

## Deploy

- Use `gunicorn` for production:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5000 api.server:app
  ```
- Mount the `index/` directory as a volume if running in Docker.
- Set `PORT` env var in cloud platforms (Render, Fly, Railway, etc.).
