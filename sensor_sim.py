# sensor_sim.py
import random, time

def simulate_pir(interval=2.0):
    """Yield True when motion detected."""
    while True:
        motion = random.random() < 0.2   # ~20 % chance of motion
        yield motion
        time.sleep(interval)

if __name__ == "__main__":
    for tick, motion in enumerate(simulate_pir(1.0)):
        print(f"[Tick {tick}] Motion detected? {motion}")
