# minimal-diffusion
Some minimal notes and implementations I made while self-studying Diffusion models

## Denoising Diffusion Probabilistic Models
Change to [`infra-local`](./infra-local/) directory:
```console
host$ cd ./infra-local/
```

Start and get inside the container:
```console
host$ docker compose up --build --remove-orphans -d
host$ docker compose exec minimal_ddpm bash
```

Run the program:
```console
minimal_ddpm# uv sync
minimal_ddpm# uv run python main.py
```
