import math
import random
import time
from fastapi import FastAPI

app = FastAPI()


@app.get("/flaky/{n}")
def flaky_load(n: int):
    """
    Endpoint that behaves "flakily":
    - It runs through a random number of phases.
    - Each phase is either:
        * CPU busy loop for a random duration, or
        * time.sleep for a random duration.
    This makes CPU utilization jump around during a single call.
    """
    trace = []

    for i in range(n):
        phase_type = random.choice(["busy", "sleep"])

        if phase_type == "busy":
            # Busy-loop for a random duration (CPU spike)
            duration = random.randint(8, 12)  # seconds
            end = time.time() + duration
            ops = 0

            # Naive CPU burner: do some math in a tight loop
            while time.time() < end:
                _ = math.sqrt(random.random())
                ops += 1

            trace.append({
                "phase": i,
                "type": "busy",
                "duration": round(duration, 3),
                "ops": ops,
            })

        else:
            # Sleep for a random duration (low CPU)
            duration = random.randint(8, 12)  # seconds
            time.sleep(duration)
            trace.append({
                "phase": i,
                "type": "sleep",
                "duration": round(duration, 3),
            })

    total_duration = sum(p["duration"] for p in trace)

    return {
        "phases": trace,
        "total_duration_seconds": round(total_duration, 3),
    }




@app.get("/fib/{n}")
def fib_stress(n: int):
    start = time.time()
    result = fib(n)
    duration=time.time() - start
    
    return {"result": result, "duration": round(duration, 3)}


def fib(n: int) -> int:
    if(n == 0): return 0
    if(n == 1): return 1
    
    return fib(n - 1) + fib(n - 2)