import subprocess
import time
import psutil
import pandas as pd

models = ["tinyllama", "llama3.2", "deepseek-r1:7b"]

basic_prompts = [
    ("General QA", "What is the capital of Argentina?"),
    ("Summarization", "Summarize the following paragraph: 'Artificial intelligence is a branch of computer science that aims to create machines capable of intelligent behavior. This includes learning, reasoning, problem-solving, and language understanding.'"),
    ("Code Generation", "Write a Python function that checks if a number is prime."),
    ("Creative Writing", "Write a short story about a robot discovering a forest for the first time.")
]

results = []

for model in models:
    for task_type, prompt in basic_prompts:
        print(f"Running '{task_type}' on {model}...")

        # Start time + resource snapshot
        start = time.time()
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Feed prompt
        stdout, stderr = process.communicate(input=prompt.encode(), timeout=120)

        # CPU & memory after run
        end = time.time()
        cpu_percent = psutil.cpu_percent(interval=1)  # avg over 1 second
        memory_info = psutil.virtual_memory()

        results.append({
            "model": model,
            "task": task_type,
            "prompt": prompt,
            "response": stdout.decode().strip(),
            "error": stderr.decode().strip(),
            "time_taken_sec": round(end - start, 2),
            "cpu_percent": cpu_percent,
            "ram_used_mb": round((memory_info.total - memory_info.available) / (1024 * 1024), 2)
        })

# Save results
df = pd.DataFrame(results)
df.to_csv("ollama_with_resource_usage.csv", index=False)

print("All results (with CPU/RAM) saved to .csv")
