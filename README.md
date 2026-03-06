# M3Benchmark

> **Getting started?** See [setup.md](setup.md) for installation, container setup, and how to run benchmarks and e2e tests.

## 📊 Dataset

Please find more details about the dataset (download, schema, etc.) in [docs/dataset.md](docs/dataset.md) and APIs in [apis](apis).


## Data Schema

| Field Name             | Type          | Description                                                                                                                                                           |
|------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `sample_id`       | string        | A unique identifier for each dialogue. |
| `domain`          | string        | Domain label for the dialogue. Possible values: "finance", "music", "movie3", "sports" ...... |
| `num_turns`       | int           | Number of turns in the dialogue |
| `turns`           | list          | List of dictionaries containing the question for each turn, answer, and API call|
| `tool_list`       | list          | List of available tools for answering questions within the dialogue.|
| `alt_ans`         | list          | Other valid gold standard answers to the question.|
| `scenarios`       | dict          | Description present in [docs/scenarios.md](docs/dataset.md)|

## 📏 Evaluation Metrics
M3 systems are evaluated using a scoring method that measures response quality to questions in the evaluation set. Responses are rated as perfect, acceptable, missing, or incorrect:

- Perfect: The response correctly answers the user question and contains no hallucinated content.

- Acceptable: The response provides a useful answer to the user question, but may contain minor errors that do not harm the usefulness of the answer.

- Missing: The answer does not provide the requested information. Such as “I don’t know”, “I’m sorry I can’t find …” or similar sentences without providing a concrete answer to the question.

- Incorrect: The response provides wrong or irrelevant information to answer the user question


Auto-evaluation: 
- Automatic evaluation employs rule-based matching and LLM assessment to check answer correctness. It will assign three scores: correct (1 point), missing (0 points), and incorrect (-1 point).


Please refer to [evaluation.py](evaluation.py) for more details on how the evaluation was implemented.



## 🏁 Baselines
We include three baselines for demonstration purposes, and you can read more about them in [docs/baselines.md](docs/baselines.md).


## Citations


## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE). This license permits sharing and adapting the work, provided it's not used for commercial purposes and appropriate credit is given. For a quick overview, visit [Creative Commons License](https://creativecommons.org/licenses/by-nc/4.0/).
