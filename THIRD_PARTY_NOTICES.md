# Third-party sources

ATST model code is derived from [Florian Schmid's PretrainedSED](https://github.com/fschmid56/PretrainedSED)
at revision `1aa47e482f7e89904cba2338999345025d8b4e36` (MIT).
The original [license](src/waves_sed/_vendor/pretrained_sed/LICENSE),
[exact source and changes](src/waves_sed/_vendor/pretrained_sed/UPSTREAM.md), and
[file hashes](src/waves_sed/_vendor/pretrained_sed/MANIFEST.sha256.json) are included.
Only package imports were changed in the copied model files.

The local linear-head adapter follows the default `PredictionsWrapper` inference path;
it validates all checkpoint weights and explicitly permits only two absent, deterministic
torchaudio mel buffers. No model training is implemented.

The 447 class names/output ordering and corresponding AudioSet MIDs come from the same
upstream revision. See [label provenance](src/waves_sed/resources/README.md).
This is the model's output vocabulary, not a full AudioSet ontology or a source mapping.

The `ATST-F_strong_1.pt` weights are downloaded from the author's
[v0.0.1 release](https://github.com/fschmid56/PretrainedSED/releases/tag/v0.0.1).
They are not redistributed in this repository. The pinned SHA-256 was computed from
that download; it is not represented as a publisher-signed checksum.

Optional verification uses the upstream test WAV
`752547__iscence__milan_metro_coming_in_station.wav`.
Attribution and licensing are in the upstream `test_files/freesound_attributions.txt`.
That WAV is not redistributed here.

## AudioSet ontology

The bundled `resources/audioset_ontology.json` is an unchanged copy of the official
[Google AudioSet ontology](https://github.com/audioset/ontology), revision
`d417d32bf59c711abb5910fd2f76a0eb44697991`, attributed to Google Inc. / Dan Ellis.
The ontology is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Its [attribution](src/waves_sed/resources/audioset_ontology.LICENSE.md) and
[source/hash metadata](src/waves_sed/resources/audioset_ontology.provenance.json) are included.
This data license applies to the bundled ontology, separately from the model code's MIT license.

The archived ontology has 632 nodes. It covers 416 of the current model's 447 IDs;
31 model IDs have no entry and 11 shared IDs have different display names. Neither source
is rewritten to hide these differences. Model-only IDs can be explicitly mapped using
their real model metadata, but their ontology relationships are unavailable.
