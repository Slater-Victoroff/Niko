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

2. The container runs `python run.py` and prints torch/the_well import status and lists files in `/app/data`.

Dataset download

This setup mounts your host `./data` directory into the container at `/app/data` as read-only by default.
If you need the example dataset `rayleigh_benard`, download it on the host (recommended) before starting the container.

Using the-well CLI (on the host):

```bash
# from project root, this downloads into ./data/datasets/rayleigh_benard
the-well-download --base-path ./data --dataset rayleigh_benard
```

If you prefer to download inside the container, run an interactive container with a writable mount and use:

```bash
docker compose run --service-ports --rm app bash
# then inside container
the-well-download --base-path . --dataset rayleigh_benard
```

Note: the original automatic download logic was removed because the host mount may be read-only and
downloads can end up in unexpected places. The README now documents explicit, repeatable commands.

Notes
- The `requirements.txt` pins CPU wheels for `torch`/`torchvision`; if your platform is different or wheel tags
  don't match, edit `requirements.txt` to a compatible torch package (for example, simply `torch`).
- If you need GPU support or a specific CUDA version, you'll need a different base image and matching torch wheels.
