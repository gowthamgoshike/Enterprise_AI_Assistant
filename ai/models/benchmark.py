import time

from ai.models.ollama_client import (
    generate,
)


prompt = """
What was our company's revenue today?
"""


start = time.perf_counter()


answer = generate(
    prompt
)


duration = (
    time.perf_counter()
    -
    start
)


print(answer)

print(
    f"\nLatency: "
    f"{duration:.2f} seconds"
)