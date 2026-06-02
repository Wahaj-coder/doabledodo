import json

# load your file
with open("eval_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

false_hits = []

for r in data["results"]:
    if r.get("hit") is False:
        false_hits.append({
            "query": r["query"],
            "expected_file": r["expected_file"],
            "expected_name": r["expected_name"],
            "retrieved": [
                {
                    "file_path": x["file_path"],
                    "name": x["name"]
                }
                for x in r.get("retrieved", [])
            ]
        })

# print nicely
for i, item in enumerate(false_hits, 1):
    print(f"\n--- False Hit #{i} ---")
    print("Query:", item["query"])
    print("Expected File:", item["expected_file"])
    print("Expected Name:", item["expected_name"])
    print("\nRetrieved:")
    for j, r in enumerate(item["retrieved"], 1):
        print(f"  {j}. {r['file_path']} — {r['name']}")

with open("false_hits.json", "w", encoding="utf-8") as f:
    json.dump(false_hits, f, indent=2)