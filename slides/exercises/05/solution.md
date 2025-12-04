# Task 05 - Autoscaling - Robert Zacchia

For the Autoscaling task, I made a small custom application with 2 endpoints.
```flaky_load(n: int):``` and ```fib_stress(n: int):```.

```flaky_load``` makes a loop with n iterations where each iteration it gets randomly choosen if the function sleeps or makes heavy square root calculations. Each iteration takes between 8 and 12 seconds. I purposefully chose this timeframe to exploit the 10 second wait time before scaling happens. (Resource Fluctuations)

```fib_stress``` is just a naive recursive fibonacchi implementation to stress the cpu. (Burst Scaling)

## Minikube Setup

```bash
minikube start -p task05 --nodes=3 --cpus=2 --memory=2g
kubectl config use-context task05

cd app
docker build -t task05:latest
cd ..
minikube -p task05 image load task05:latest

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/app.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/horizontal.yaml
```

```bash
while true; do
  echo "=== $(date -Iseconds) ===" >> scaling.log
  kubectl get hpa -A >> scaling.log
  kubectl get deploy -A >> scaling.log
  kubectl top pods -A >> scaling.log  # needs metrics-server
  echo "" >> scaling.log
  sleep 10
done
```