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
