# Apache Jira Evaluation Dataset

## Source of truth

- `corpus.jsonl`: authoritative indexing corpus containing 600 historical cases.
- `queries.jsonl`: authoritative evaluation set containing 30 held-out duplicate reports with 0%, 30%, and 60% progress queries.

Each file is UTF-8 JSON Lines: every non-empty line is one JSON object.

## Data structure

### `corpus.jsonl`

Each line is one historical case to index:

- `case_id`, `issue_id`: Jira's numeric issue identifier.
- `key`: public Jira key, such as `CASSANDRA-10233`.
- `project`: `CASSANDRA`, `HADOOP`, `HBASE`, or `SPARK`.
- `cluster`: duplicate-group identifier for a gold target, or a unique distractor identifier.
- `role`: `gold_target` or `fixed_distractor`.
- `summary`: Jira issue title.
- `description`: initial Jira issue description.
- `comments`: chronological list of objects containing `body` and `created`.
- `conversations`: normalized ordered messages derived from the description and comments. Each message contains `from`, `body`, and, for the initial report, `subject`.
- `created`, `resolution`, `resolution_date`: Jira lifecycle metadata.
- `metadata`: source metadata identifying Apache Jira.

### `queries.jsonl`

Each line is one held-out duplicate report:

- `issue_id`, `key`, `project`, `cluster`: query identity and gold duplicate group.
- `target_key`: exact historical target expected in the retrieval result.
- `n_comments`: number of usable pre-disclosure comments.
- `progress_valid`: validity flags for the 0%, 30%, and 60% query stages.
- `query_0`: issue summary and description.
- `query_30`: query text with the first 30% of usable comments.
- `query_60`: query text with the first 60% of usable comments.

## Data source

The dataset was built from public issue records and human-created duplicate links retrieved directly from the [Apache Software Foundation Jira issue tracker](https://issues.apache.org/jira/) using the Jira REST API.

From the issue trackers for [Apache Cassandra](https://issues.apache.org/jira/projects/CASSANDRA), [Apache Hadoop](https://issues.apache.org/jira/projects/HADOOP), [Apache HBase](https://issues.apache.org/jira/projects/HBASE), and [Apache Spark](https://issues.apache.org/jira/projects/SPARK), we used contributor-created duplicate links to pair later reports with historical issues already resolved as `Fixed` before those reports were opened. The later reports form the held-out queries; their linked historical issues form the gold corpus cases. Other `Fixed` issues from these same four sources provide distractors, while issue titles, descriptions, and chronological comments provide the case and query histories.

## How to use the dataset

**Build the index with `corpus.jsonl`.** Treat each row as one historical case. Use `conversations` as its ordered content, or render the equivalent `summary`, `description`, and `comments`; these are alternative representations of the same content, not additional text to concatenate. Keep a mapping from indexed case or chunk IDs to the case's Jira `key` so retrieval results can be matched back to source cases.

**Query the index with `queries.jsonl`.** Each row represents a separate held-out report at different stages of its investigation. Submit the supplied `query_0`, `query_30`, or `query_60` text only when the matching `progress_valid` flag (`"0"`, `"30"`, or `"60"`) is true. Skip invalid stages even if query text is present. Keep these reports out of the index, and use `target_key`, `cluster`, and corpus `role` only for evaluation, not as retrieval signals.

**Evaluate the retrieved cases.** Match the returned cases' Jira `key` values against the query's `target_key`. A Case Hit is 1 when the expected historical case is returned and 0 otherwise. Average this score separately for each progress stage over its valid queries. When comparing retrieval methods, use the same corpus, valid queries, and retrieval constraints.
