## Generate Client Readme

First install nswag cli
```bash
  ddotnet tool install --global NSwag.ConsoleCore
```
Then start all dockers with
```bash
docker-compose up
```
Execute this in directory root of solution ./BookRent 

CatalogServiceClient
```bash
nswag openapi2csclient \
  /input:http://localhost:8081/openapi/v1.json \
  /classname:CatalogClient \
  /namespace:BookRent.Orchestrator.Clients \
  /output:./BookRent.Orchestrator/Clients/CatalogClient.g.cs

```

IdentityServiceClient
```bash
nswag openapi2csclient \
  /input:http://localhost:8082/openapi/v1.json \
  /classname:IdentityClient \
  /namespace:BookRent.Orchestrator.Clients \
  /output:./BookRent.Orchestrator/Clients/IdentityClient.g.cs \
  /GenerateExceptionClasses:false
```

RentingServiceClient
```bash
nswag openapi2csclient \
  /input:http://localhost:8083/openapi/v1.json \
  /classname:RentingClient \
  /namespace:BookRent.Orchestrator.Clients \
  /output:./BookRent.Orchestrator/Clients/RentingClient.g.cs \
  /GenerateExceptionClasses:false
```

UserServiceClient
```bash
nswag openapi2csclient \
  /input:http://localhost:8084/openapi/v1.json \
  /classname:UserClient \
  /namespace:BookRent.Orchestrator.Clients \
  /output:./BookRent.Orchestrator/Clients/UserClient.g.cs \
  /GenerateExceptionClasses:false
```

PublicOrchestrationClient
```bash
nswag openapi2tsclient \
  /input:http://localhost:8080/openapi/v1.json \
  /classname:apiClient \
  /output:../BookRent.WebClient/src/Client/apiClient.g.ts
```