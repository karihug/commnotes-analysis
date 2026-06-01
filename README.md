# Community Notes Analysis

This repository contains the notebooks, helper code, and data-generation scripts used for
the Community Notes analysis.

Data files and generated outputs are not included. The notebooks and scripts currently
preserve paths from the original analysis environment; update those paths before rerunning
elsewhere.

## Structure

- `analysis/`: final analysis notebooks and shared analysis helpers. The main notebooks
  should be run top to bottom after the generated data files have been placed at the
  expected paths.
- `data_generation/communitynotes_2022/`: 2022 Community Notes scoring snapshot used to
  generate historical weekly matrix-factorization outputs. Run:

  ```bash
  cd data_generation/communitynotes_2022
  python weekly2022.py
  ```

- `data_generation/communitynotes_2025/`: 2025 Community Notes scoring snapshot,
  including the X topic model used for note-topic assignments. To run the scoring CLI,
  use:

  ```bash
  cd data_generation/communitynotes_2025
  python -m scoring.runner --notes /path/to/notes.tsv --ratings /path/to/ratings.tsv --status /path/to/noteStatusHistory.tsv --enrollment /path/to/userEnrollment.tsv --outdir /path/to/output
  ```

  To generate the note-topic CSV used by the analysis, use the
  `process_note_topics(...)` helper in `analysis/src/data_loader.py`, which calls
  `scoring/topic_model.py` and writes `df_notes_with_topics.csv`.

- `data_generation/communitynotes_twostage/`: two-stage matrix-factorization analysis
  code. On SLURM, run:

  ```bash
  cd data_generation/communitynotes_twostage
  sbatch run_mf.sh
  ```

  Without SLURM, run:

  ```bash
  cd data_generation/communitynotes_twostage
  python run_mf.py --output_dir /path/to/output
  ```

- `data_generation/llm_topic_modeling/`: placeholder for code that
  generated `notes_llm_topic.csv`. The CSV itself is not included in this repository.
