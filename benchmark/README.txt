1. Start with 1 client (PARALLEL=1)

- $ docker compose down
- $ docker compose up sim server -d
- $ docker compose run -it --rm client-xxx

2. Start with N client (PARALLEL=N)

- $ docker compose down
- $ docker compose up sim server client-xxx
- check `logs/client-xxx`

3. Cleanup & Rebuild

- $ docker compose down
- $ docker compose build


Notice:
- You cannot start two clients at the same time
- Clients might stuck at "Waiting for the sim net to be ready...", rerun all commands fix this
