import time


def stage(name):
    print(f"\n==== Stage: {name} ====")
    return time.time()

def done(start_time):
    elapsed = time.time() - start_time
    print(f"✔ Done ({elapsed:.2f}s)")
