import yaml
import os


def load_config(yaml_dir):
    with open(yaml_dir, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)  # use safe_load for safety

    ResultsDir = config['ResultsDir']
    ModelName = config['ModelName']
    KnowledgeBenchDir = config['KnowledgeBenchDir']
    TestingScope = config['TestingScope']

    result_file_dir = os.path.join(ResultsDir, ModelName)
    os.makedirs(result_file_dir, exist_ok=True)

    tested_file_name = []
    for filename in os.listdir(result_file_dir):
        if filename.endswith(".json"):
            tested_file_name.append(filename)
    print(f"there are {len(tested_file_name)} files have been tested.")
    knowledge_bench_file = []
    for root, _, files in os.walk(KnowledgeBenchDir):
        for filename in files:
            if filename.endswith(".json"):
                knowledge_bench_file.append(os.path.join(root, filename))
    print(f"load total number of files: {len(knowledge_bench_file)}")

    # Exclude files that have already been tested
    results_file = [item for item in knowledge_bench_file if os.path.basename(item) not in tested_file_name]

    print(results_file)
    print(f"load total number of untested files: {len(results_file)}")
    # Filter by TestingScope

    flat_TestingScope = {}

    def flatten(d):
        for k, v in d.items():
            if isinstance(v, dict):
                flatten(v)
            else:
                flat_TestingScope[k] = bool(v)

    flatten(TestingScope)
    print("detailed testing scope: ", flat_TestingScope)

    kept = []
    for path in results_file:
        filename = os.path.basename(path)  # get file name
        for key, enabled in flat_TestingScope.items():
            if enabled and filename.startswith(key):
                kept.append(path)
                break

    print(kept)
    print(f"load total number of filtered untested files: {len(kept)}")
    return config, kept


if __name__ == "__main__":
    result = load_config('/Users/kendrickstein/Code/GUI_Knowledge_Bench/Inference/configs/thinking_gpt-5-2025-08-07.yaml')
    # print(result)
