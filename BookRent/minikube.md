## Setup Cluster

```bash
minikube start --profile=bookrent --driver=docker --cpus=4 --memory=12g

😄  [bookrent] minikube v1.37.0 on Ubuntu 24.04
✨  Using the docker driver based on user configuration
📌  Using Docker driver with root privileges
👍  Starting "bookrent" primary control-plane node in "bookrent" cluster
🚜  Pulling base image v0.0.48 ...
💾  Downloading Kubernetes v1.34.0 preload ...
    > gcr.io/k8s-minikube/kicbase...:  488.52 MiB / 488.52 MiB  100.00% 2.81 Mi
    > preloaded-images-k8s-v18-v1...:  337.07 MiB / 337.07 MiB  100.00% 1.85 Mi
🔥  Creating docker container (CPUs=4, Memory=12288MB) ...
🐳  Preparing Kubernetes v1.34.0 on Docker 28.4.0 ...
🔗  Configuring bridge CNI (Container Networking Interface) ...
🔎  Verifying Kubernetes components...
    ▪ Using image gcr.io/k8s-minikube/storage-provisioner:v5
🌟  Enabled addons: storage-provisioner, default-storageclass
🏄  Done! kubectl is now configured to use "bookrent" cluster and "default" namespace by default
```
## Check Cluster
```bash
minikube status --profile=bookrent
bookrent
type: Control Plane
host: Running
kubelet: Running
apiserver: Running
kubeconfig: Configured


kubectl config use-context bookrent
Switched to context "bookrent".
kubectl config set-context --current --namespace=bookrent


kubectl get nodes
NAME       STATUS   ROLES           AGE     VERSION
bookrent   Ready    control-plane   2m12s   v1.34.0

```
## Build Docker Images
```bash
docker-compose build --no-cache
```
sql-secret.yaml
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mssql-secret
type: Opaque
stringData:
  SA_PASSWORD: "Your_SA_Password_123!"
```
apply
```bash
kubectl apply -f k8s/secret-sql.yaml -n bookrent

kubectl config use-context bookrent
kubectl config set-context --current --namespace=bookrent

kubectl apply -f k8s/sqls.yaml -n bookrent

```

- catalog-sql:1433
- renting-sql:1433
- user-sql:1433
- identity-sql:1433

```bash
kubectl apply -f k8s/services.yaml -n bookrent
kubectl get pods -n bookrent
kubectl get svc -n bookrent
```

Now, inside the cluster, you have:

- catalog-api → http://catalog-api
- renting-api → http://renting-api
- user-api → http://user-api
- identity-api → http://identity-api

```bash
minikube addons enable ingress --profile=bookrent
minikube ip --profile=bookrent
```

Edit /etc/hosts
```
192.168.49.2   api.bookrent.local
192.168.49.2   dev.bookrent.local

```

self signed ssl and create the secret in bookrent cluster
```bash
openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout api-bookrent.key \
  -out api-bookrent.crt \
  -subj "/CN=api.bookrent.local/O=api.bookrent.local"


kubectl create secret tls bookrent-tls \
  --cert=api-bookrent.crt \
  --key=api-bookrent.key \
  -n bookrent

```
