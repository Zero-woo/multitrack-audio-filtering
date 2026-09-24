# PretrainedSED ATST implementation

Source: [https://github.com/fschmid56/PretrainedSED](https://github.com/fschmid56/PretrainedSED)  
Pinned revision: [`1aa47e482f7e89904cba2338999345025d8b4e36`](https://github.com/fschmid56/PretrainedSED/tree/1aa47e482f7e89904cba2338999345025d8b4e36)  
Upstream license: MIT, retained verbatim in `LICENSE`.

Only the four Python files required for the ATST backbone and its wrapper are
vendored. Files were read from the pinned Git objects, preserving their original
bytes except for these two import changes in `models/atstframe/ATSTF_wrapper.py`:

```diff
-from models.atstframe.audio_transformer import FrameASTModel
-from models.transformer_wrapper import BaseModelWrapper
+from .audio_transformer import FrameASTModel
+from ..transformer_wrapper import BaseModelWrapper
```

No model computation or preprocessing was changed. Empty package initializers
with docstrings were added to `_vendor`, `pretrained_sed`, `models`, and
`models/atstframe`. This provenance document and `MANIFEST.sha256.json` were added
locally. The manifest records SHA-256 digests for the original and vendored bytes
of every copied file, including the license. Package initializers and provenance
files are local additions rather than upstream files.

AudioSet strong label resources are documented separately in
`../../resources/README.md`.
