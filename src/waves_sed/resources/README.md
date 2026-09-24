# AudioSet strong class metadata

Source: [PretrainedSED at `1aa47e482f7e89904cba2338999345025d8b4e36`](https://github.com/fschmid56/PretrainedSED/tree/1aa47e482f7e89904cba2338999345025d8b4e36).
The upstream MIT license is retained in `../_vendor/pretrained_sed/LICENSE`.

`audioset_strong.json` contains 447 objects with `index`, `id`, and `name`.
`index` is the classifier output index (zero based), in the exact declaration
order of `as_strong_train_classes` from
[`data_util/audioset_classes.py`](https://github.com/fschmid56/PretrainedSED/blob/1aa47e482f7e89904cba2338999345025d8b4e36/data_util/audioset_classes.py).
This order must be preserved when interpreting model probabilities.

Actual AudioSet IDs were joined by exact class name against
[`hf_dataset_gen/metadata/class_labels_indices_strong.csv`](https://github.com/fschmid56/PretrainedSED/blob/1aa47e482f7e89904cba2338999345025d8b4e36/hf_dataset_gen/metadata/class_labels_indices_strong.csv).
That CSV is preserved verbatim here as `class_labels_indices_strong.csv`.
It has two columns, AudioSet ID and class name, with no header; its row order is
not the model output order. No synthetic label IDs are used.

`audioset_strong.provenance.json` records the source revision, original SHA-256
digests, generated JSON digest, and transformation. Extraction uses Python AST
literal parsing and CSV parsing without importing upstream code or PyTorch.
Generation checks require 447 distinct names and 447 distinct IDs.
