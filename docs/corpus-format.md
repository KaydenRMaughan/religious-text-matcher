# Corpus format

The app loads any collection of texts saved as a JSON array of documents (or JSON Lines, one document per line).

| Field | Required | Type | Notes |
|---|---|---|---|
| `id` | yes | string | Unique, stable across rebuilds |
| `title` | yes | string | Shown in results |
| `text` | yes | string | Full text; chunked at load |
| `source` | no | string | e.g. collection or book name, used as a filter |
| `year` | no | integer | Enables the year filter |
| `author` | no | string | Shown in results |
| `url` | no | string | Linked from results |

```json
[
  {"id": "doc-001", "title": "Example", "text": "Full text here...", "source": "My notes", "year": 2025}
]
```

To point the app at a local corpus, create `config.local.toml` (gitignored) in the repo root:

```toml
[corpus]
path = "data/private/my_corpus.json"
name = "My corpus"
```
