import json
import sys

def transform_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    domain = input_path.split("_bird_")[1].replace(".json", "")

    for row in data:
        try:
            sample_id = row["sample_id"]
            query = row["input"]
            answer = row["gold_answer"]
            arguments = row["output"][0]["arguments"]
            result[domain+"_"+str(sample_id)] = {
                "init_args": arguments,
                "query": query, 
                "answer": answer
            }
        except (KeyError, IndexError, TypeError) as e:
            raise Exception(f"Skipping row due to error: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(result)} entries to {output_path}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    input_json = sys.argv[1]
    output_json = sys.argv[2]

    transform_json(input_json, output_json)
