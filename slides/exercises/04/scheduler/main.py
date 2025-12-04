import os
import time
import random
from typing import List

from kubernetes import client, config, watch
from kubernetes.client import V1Node, V1Pod, V1Binding, V1ObjectMeta, V1ObjectReference


def load_kube_config():
    """
    Try in-cluster config first, fall back to KUBECONFIG if present.
    """
    try:
        config.load_incluster_config()
        print("Loaded in-cluster Kubernetes configuration")
    except Exception:
        kubeconfig_path = os.getenv("KUBECONFIG")
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
            print(f"Loaded kubeconfig from {kubeconfig_path}")
        else:
            config.load_kube_config()
            print("Loaded kubeconfig from default location")


# ------------ Genetic Algorithm types and helpers ------------

class Individual:
    def __init__(self, node_index: int, fitness: float):
        self.node_index = node_index
        self.fitness = fitness


def compute_fitness(node: V1Node) -> float:
    """
    Use node label 'latency-score' as a proxy for response time.
    Lower latency-score => higher fitness.
    """
    labels = node.metadata.labels or {}
    latency_val = 10  # default if label missing or bad

    val = labels.get("latency-score")
    if val is not None:
        try:
            latency_val = int(val)
        except ValueError:
            pass

    # simple transform, avoid division by zero
    return 1.0 / (1.0 + float(latency_val))


def tournament_select(population: List[Individual], tournament_size: int = 3) -> Individual:
    best = random.choice(population)
    for _ in range(1, tournament_size):
        contender = random.choice(population)
        if contender.fitness > best.fitness:
            best = contender
    return best


def run_genetic_algorithm(nodes: List[V1Node]) -> V1Node:
    """
    GA over node indices. Each individual has a single 'gene' = node index.
    """
    population_size = 20
    generations = 15
    mutation_rate = 0.1

    # Initialize population
    population: List[Individual] = []
    for _ in range(population_size):
        idx = random.randrange(len(nodes))
        population.append(Individual(node_index=idx, fitness=compute_fitness(nodes[idx])))

    # Evolve
    for _ in range(generations):
        new_pop: List[Individual] = []

        while len(new_pop) < population_size:
            # Tournament selection
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)

            # Crossover (trivial with one gene: pick parent1 or parent2)
            if random.random() < 0.5:
                child_idx = parent1.node_index
            else:
                child_idx = parent2.node_index

            # Mutation
            if random.random() < mutation_rate:
                child_idx = random.randrange(len(nodes))

            child = Individual(node_index=child_idx, fitness=compute_fitness(nodes[child_idx]))
            new_pop.append(child)

        population = new_pop

    # Pick best individual
    best = max(population, key=lambda ind: ind.fitness)
    return nodes[best.node_index]


# ------------ Scheduling logic ------------

def schedule_pod(core_v1: client.CoreV1Api, pod: V1Pod, scheduler_name: str):
    # List all nodes
    nodes_resp = core_v1.list_node()
    nodes = nodes_resp.items
    if not nodes:
        print("No nodes available to schedule pod")
        return

    # Run GA to choose best node
    best_node = run_genetic_algorithm(nodes)
    node_name = best_node.metadata.name
    print(f"Selected node {node_name} for pod {pod.metadata.namespace}/{pod.metadata.name}")

    # Create Binding object
    target = V1ObjectReference(
        api_version="v1",
        kind="Node",
        name=node_name,
    )

    metadata = V1ObjectMeta(
        name=pod.metadata.name,
        namespace=pod.metadata.namespace,
        uid=pod.metadata.uid,
    )

    binding = client.V1Binding(
        metadata=metadata,
        target=target
    )

    try:
        core_v1.create_namespaced_pod_binding(
            name=pod.metadata.name,
            namespace=pod.metadata.namespace,
            body=binding
        )
        print(f"Successfully bound pod {pod.metadata.namespace}/{pod.metadata.name} to node {node_name}")
    except Exception as e:
        print(f"Failed to bind pod {pod.metadata.namespace}/{pod.metadata.name}: {e}")



def watch_pods(scheduler_name: str):
    load_kube_config()
    core_v1 = client.CoreV1Api()
    w = watch.Watch()

    print(f"Starting scheduler '{scheduler_name}' watch loop")

    while True:
        try:
            stream = w.stream(
                core_v1.list_pod_for_all_namespaces,
                timeout_seconds=0,
            )

            for event in stream:
                pod: V1Pod = event["object"]
                event_type = event["type"]

                # Basic filters
                if pod.spec is None:
                    continue
                if pod.spec.scheduler_name != scheduler_name:
                    continue
                if pod.status is None or pod.status.phase != "Pending":
                    continue
                if pod.spec.node_name:  # already scheduled
                    continue

                print(
                    f"Event {event_type}: Pod {pod.metadata.namespace}/{pod.metadata.name} "
                    f"is Pending and uses scheduler '{scheduler_name}'"
                )
                schedule_pod(core_v1, pod, scheduler_name)

        except Exception as e:
            print(f"Error in watch loop: {e}")
            time.sleep(2)


if __name__ == "__main__":
    random.seed(int(time.time()))

    scheduler_name = os.getenv("SCHEDULER_NAME", "metaheuristic-scheduler")
    print(f"Scheduler name: {scheduler_name}")

    try:
        watch_pods(scheduler_name)
    except Exception as e:
        # Ensure any fatal error shows up in logs before exit
        import traceback
        print("Fatal error in scheduler:", e)
        traceback.print_exc()
        time.sleep(5)
