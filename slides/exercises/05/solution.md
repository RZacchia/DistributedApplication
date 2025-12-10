# Task 05 - Autoscaling - Robert Zacchia

For the Autoscaling task, I made a small custom application with 2 endpoints.
```flaky_load(n: int):``` and ```fib_stress(n: int):```.

```flaky_load``` makes a loop with n iterations where each iteration it gets randomly choosen if the function sleeps or makes heavy square root calculations. Each iteration takes between 8 and 12 seconds. I purposefully chose this timeframe to exploit the 10 second wait time before scaling happens. (Resource Fluctuations)

```fib_stress``` is just a naive recursive fibonacchi implementation to stress the cpu. (Burst Scaling)

## Minikube Setup

In the following section you can see the setup of my minikube cluster and how to deploy the application into it. The addon ingress is used to not have to forward ports during initiating cpu load. The metric server is needed so minikube has access to the metrics of the single deployments and nodes. After you have applied the ingress you need to call ```minikube ip``` and add that to the /ect/hosts file.

```bash
minikube start -p task05 --nodes=3 --cpus=2 --memory=2g
# addons
minikube -p task05 addons enable ingress
minikube -p task05 addons enable metrics-server

kubectl config use-context task05

cd app
docker build -t task05:latest .
cd ..
minikube -p task05 image load task05:latest

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/app.yaml
kubectl apply -f k8s/ingress.yaml
```
## Testing setup

To monitor the application I made a quick bash [script](./monitor.sh). I then used the log file to get a time graph of the scaling. To initiate calls I used [Insomnia](https://insomnia.rest) and automated repeated calling of the cluster.

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

### Horizontal Scaling

For horizontal scaling you need to have a HorizontalPodAutoscaler yaml file.
There you can describe the metrics which should be observed, what deployment should be observed and min/max replicas. With ```kubectl apply -f k8s/horizontal.yaml``` the autoscaler gets added to the cluster.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: task05
  namespace: task05
spec:
  minReplicas: 1
  maxReplicas: 5
  metrics:
    - resource:
        name: cpu
        target:
          averageUtilization: 80
          type: Utilization
      type: Resource
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: task05
```

### Vertical Scaling

The vertical autoscaler works different and needs more setup. Before vertical scaling can be used I had to...

With ```kubectl apply -f k8s/vertical.yaml``` the autoscaler gets added to the cluster.

```yaml
apiVersion: autoscaling/v2
kind: VerticalPodAutoscaler
metadata:
  name: task05
  namespace: task05
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: task05
  updatePolicy:
    updateMode: Off

```


## Results

