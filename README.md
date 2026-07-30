Basic Docker Compose setup for a PyTorch-enabled Python app that mounts a static `data/` directory.

Files added:
- `docker-compose.yml` — service `app` that builds `./app` and mounts `./data` into `/app/data` (read-only).
- `data/` — sample dataset directory mounted into the container.
- `app/Dockerfile` — Python 3.11 slim image, installs requirements and copies `run.py`.
- `app/requirements.txt` — installs `torch`, `torchvision`, and `the_well`.
- `app/run.py` — small script that imports torch and the_well and lists `/app/data`.

Quick start

1. Build and start the service (in the project root):

```bash
docker compose up --build
```
