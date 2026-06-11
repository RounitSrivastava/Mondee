# Checkpoint System

This directory stores runtime checkpoint files used to resume index building after interruptions.

## Files

| File | Purpose |
|------|---------|
| `experiences_checkpoint.json` | Saves parsed catalog records after data ingestion (from `scripts/build_index.py`) |
| `embedding_checkpoint.npy` | Saves partial E5 embeddings during batch encoding (from `core/indexer.py`) |

## Auto-Checkpoint Behavior

### Experiences Checkpoint
- **Created** after all data files are parsed into memory
- **Resume** - if `experiences_checkpoint.json` exists on next run, data ingestion is skipped entirely
- **Removed** automatically after a successful full build

### Embedding Checkpoint
- **Created** after each batch during the E5 encoding loop
- **Resume** - if `embedding_checkpoint.npy` exists, encoding resumes from the last completed batch
- **Removed** automatically after a successful full build

## Manual Control

To force a complete fresh build from scratch:
```powershell
# Remove checkpoints manually
Remove-Item -LiteralPath "index\experiences_checkpoint.json" -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "index\embedding_checkpoint.npy" -ErrorAction SilentlyContinue
```

Then re-run:
```powershell
python scripts/build_index.py
```
