# Community Notes 2025 Scoring Code

This folder contains the Community Notes scoring code snapshot used for the 2025-style weekly outputs.

The X topic-modeling code lives in:

```text
scoring/topic_model.py
```

That file defines `TopicModel` and returns note-topic assignments, but it does not directly write `df_notes_with_topics.csv`. The CSV is written by:

```text
../../analysis/src/data_loader.py
```

in the `process_note_topics(...)` helper, which calls `TopicModel.get_note_topics(...)`, merges the resulting topics onto the notes dataframe, and saves `df_notes_with_topics.csv`.

Input data and generated outputs are not included in this repository.
