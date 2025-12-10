# DistributedApplication

In directory BookRent call
```bash
docker-compose up --build #OR
docker compose up --build
```
under localhost:port/scalar/v1 the API documentation can be found for the chosen service (e.g. localhost:8080/scalar/v1 shows documentation of the orchestrator)
In production only orchestrator should be able to be called from customer side


ports for services:
- orchestrator
  - api: 8080
- catalog
  - api: 8081
  - db: 1433
- identity
  - api: 8082
  - db: 1434
- renting
  - api: 8083
  - db: 1435
- user
  - api: 8084
  - db: 1436

To only start databases use
```bash
docker compose up -d $(docker compose config --services | grep "db")
```