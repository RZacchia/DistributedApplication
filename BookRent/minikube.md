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


kubectl get nodes
NAME       STATUS   ROLES           AGE     VERSION
bookrent   Ready    control-plane   2m12s   v1.34.0

```
## Build Catalog Docker Image
```bash
docker build -t bookrent-catalog:0.1 \
  -f Services/BookRent.Catalog/Dockerfile \
  Services/BookRent.Catalog

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
```