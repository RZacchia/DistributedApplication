# Exercise 02 - Robert Zacchia
# Installation
Installation is pretty straight forward in Ubuntu. I used the snap package for kubectl, and downloaded the binary for minikube.
```bash
sudo snap install kubectl --classic

curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
chmod +x minikube
sudo cp minikube /usr/local/bin && rm minikube
```
After installing both programs I verified the installations via checking their versions.
```bash
minikube version

minikube version: v1.37.0
commit: 65318f4cfff9c12cc87ec9eb8f4cdd57b25047f3


kubectl version

Client Version: v1.34.1
Kustomize Version: v5.7.1
Unable to connect to the server: dial tcp 192.168.49.2:8443: connect: no route to
```
# Configuration
The Configuration can be split into 3 parts
1. Creating the images for the application
2. Setting up deployment scripts
3. Applying them to the cluster

## 1. Images for the application
I decided to keep the application BookRent which I used in exercise 01.
BookRent consists of 5 Services and 4 Databases.

| Service Name         | Database Name | External Callable |
|----------------------|---------------|-------------------|
| catalog-service      | catalog-db    | False             |
| identity-service     | identity-db   | False             |
| renting-service      | renting-db    | False             |
| user-service         | user-db       | False             |
| orchestrator-service | None          | True              |

Only the orchestrator-service is able to call the other services and be called from external sources.
Since I already had Dockerfiles and a docker-compose file set up for the project I could just build the images via.
```bash
docker-compose build --no-cache --parallel 
```

## 2. Setting up deployment
### 2.1 Resource setup
Before I can setup the deployment I first needed to setup an environment and checked the status.
```bash
minikube start --profile=bookrent --driver=docker --cpus=4 --memory=12g

😄  [bookrent] minikube v1.37.0 on Ubuntu 24.04
✨  Using the docker driver based on user configuration
...
...
🏄  Done! kubectl is now configured to use "bookrent" cluster and "default" namespace by default


minikube status --profile=bookrent
bookrent
type: Control Plane
host: Running
kubelet: Running
apiserver: Running
kubeconfig: Configured
```
After the minikube cluster was created I could set it up to be used by kubectl via
```bash
kubectl config use-context bookrent
Switched to context "bookrent".
kubectl config set-context --current --namespace=bookrent


kubectl get nodes
NAME       STATUS   ROLES           AGE     VERSION
bookrent   Ready    control-plane   2m12s   v1.34.0
```

### 2.2 Setup Databases
Before I setup the database, I made an sql-secrety.yaml file to store the password for the databases.
In real applications this file should not be in the source control but I included it for that example.
I also use the same credentials for all databases.
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mssql-secret
type: Opaque
stringData:
  SA_PASSWORD: "Your_SA_Password_123!"
```
This helps me to set the password in the databases and services.

The databases consist of multiple statefulsets which are defined in the sql.yaml file



## 3. After the