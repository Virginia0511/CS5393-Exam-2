import time
import psutil
import pandas as pd
import ollama

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

        try:
            start = time.time()
            # Generate response using Ollama Python client
            response = ollama.generate(
                model=model,
                prompt=prompt,
                stream=False
            )
            end = time.time()

            # Collect CPU & memory after run
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()

            results.append({
                "model": model,
                "task": task_type,
                "prompt": prompt,
                "response": response['response'].strip(),
                "error": "",
                "time_taken_sec": round(end - start, 2),
                "cpu_percent": cpu_percent,
                "ram_used_mb": round((memory_info.total - memory_info.available) / (1024 * 1024), 2)
            })

        except Exception as e:
            results.append({
                "model": model,
                "task": task_type,
                "prompt": prompt,
                "response": "[Error]",
                "error": str(e),
                "time_taken_sec": "N/A",
                "cpu_percent": "N/A",
                "ram_used_mb": "N/A"
            })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("ollama_with_resource_usage.csv", index=False)

print("Completed and saved to .csv")
