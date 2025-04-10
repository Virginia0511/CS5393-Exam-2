import time
import pandas as pd
import psutil
import ollama

# List of HR task prompts
prompts = [
    "Write a job description for a mid-level software engineer at a startup.",
    "Create five behavioral interview questions to assess teamwork skills.",
    "Create a Resume template for a Software Engineer candidate.",
    "An employee is consistently late and their teammate is frustrated. How should HR handle this?",
    "List best practices for promoting diversity in hiring.",
    "Write constructive feedback for an employee who meets deadlines but struggles with team communication.",
    "Draft a welcome email for a new hire joining the marketing department, including their first-day schedule.",
    "Write a professional and empathetic message to notify an employee of a layoff due to organizational restructuring.",
    "Create a remote work policy that outlines expectations, communication guidelines, and eligibility criteria.",
    "Explain how to conduct a salary benchmarking analysis for a new role in the tech industry."
]

# Models to test
models = ["tinyllama", "llama3.2", "deepseek-r1:7b"]

# Store results
results = []

# Loop through each prompt then each model
for prompt in prompts:
    for model in models:
        print(f"prompt: '{prompt[:40]}...' | Model: {model}")
        start = time.time()
        try:
            # Generate response using Ollama's Python client
            response = ollama.generate(
                model=model,
                prompt=prompt,
            )
            end = time.time()

            # Capture CPU and RAM usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            ram_used_mb = round((memory_info.total - memory_info.available) / (1024 * 1024), 2)

            results.append({
                "prompt": prompt,
                "model": model,
                "response": response['response'].strip(),
                "error": "",
                "time_taken_sec": round(end - start, 2),
                "cpu_percent": cpu_percent,
                "ram_used_mb": ram_used_mb
            })

        except Exception as e:
            results.append({
                "prompt": prompt,
                "model": model,
                "response": "[Error]",
                "error": str(e),
                "time_taken_sec": "N/A",
                "cpu_percent": "N/A",
                "ram_used_mb": "N/A"
            })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("ollama_hr_test_with_resources.csv", index=False)

print('Completed and saved to .csv')
