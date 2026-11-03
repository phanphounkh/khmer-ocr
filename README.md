# Reproducible Khmer OCR evaluation

Evaluate all **100 human-verified test pages** from the Khmer Sala OCR dataset.
The evaluator separates dataset export, model inference and metric calculation.
No dataset images, transcriptions, model weights or published accuracy numbers
are included. No actual OCR benchmark has been run for this deliverable.

## Evidence and version discrepancy

Reviewed sources on November 3, 2026:

- [Drive dataset README, October 25](https://drive.google.com/file/d/168WPcYfoH6cMj6dtc-HhF1bCFraDWHGW/view):
  17,516 pages; dataset `CADT-IDRI/khmer-sala-ocr-17k`; adapter
  `CADT-IDRI/deepseek-ocr-2-lora-khmer-sala-ocr-17k`.
- [Implementation README at the inspected commit](https://github.com/IDRI-LAB/khmer-sala-ocr/blob/a2b7e25fbe8b3f46f927ac3f8290debec3ea5920/README.md):
  instead names dataset `CADT-IDRI/khmer-sala-ocr-16k` and adapter
  `CADT-IDRI/deepseek-ocr-2-khmer-lora-16k`.
- [Metric implementation](https://github.com/IDRI-LAB/khmer-sala-ocr/blob/a2b7e25fbe8b3f46f927ac3f8290debec3ea5920/src/utils/cer_wer_calculation.py),
  [normalizer](https://github.com/IDRI-LAB/khmer-sala-ocr/blob/a2b7e25fbe8b3f46f927ac3f8290debec3ea5920/src/utils/khmer_norm.py),
  [precleaner](https://github.com/IDRI-LAB/khmer-sala-ocr/blob/a2b7e25fbe8b3f46f927ac3f8290debec3ea5920/src/utils/preprocessing.py),
  and [inference engine](https://github.com/IDRI-LAB/khmer-sala-ocr/blob/a2b7e25fbe8b3f46f927ac3f8290debec3ea5920/src/utils/inference_engine.py).

The default dataset follows the supplied **17k** documentation. It never silently
falls back to 16k. Hub availability, model access and dataset equivalence were not
verified here. If 17k is unavailable, obtain access or explicitly choose 16k and
record it as a different benchmark until the authors confirm equivalence. Pin its
own revision. Do not claim reproduction of a paper score based on the name alone.

## Interpretation warnings

- **Possible contamination / limited generalization:** splitting was by page;
  all 55 test books are represented in training. The documentation says exact
  pages do not overlap, but related pages can share content, typography and layout.
  This does not test unseen-book performance. Pretraining or gold-test exposure
  for SORYA-AI and other models remains unknown unless their training data is audited.
- **No handwriting, scene text or tables:** scores describe printed educational
  pages, not Khmer handwriting recognition or table extraction. Use a separate,
  consented, human-transcribed handwriting set with writer-disjoint splits for that.
- **Only 100 gold pages, one annotator:** no inter-annotator agreement, and page
  selection was not stratified random sampling. Ten pages have equations. Test
  composition is more Khmer-heavy than training. Avoid broad accuracy claims.
- Training, validation and silver labels are Gemini-generated and unverified.
  Scores on those splits measure teacher agreement. This tool rejects them.
- Freeze prompts/settings before scoring. Tune on a development set, not these
  100 references. Keep a separate book-disjoint gold holdout for final validation.

## Install

Python 3.10+; use a dedicated environment. No GPU is required for scoring saved
predictions. The optional DeepSeek adapter does require a compatible CUDA GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

git clone https://github.com/IDRI-LAB/khmer-sala-ocr.git upstream
git -C upstream checkout a2b7e25fbe8b3f46f927ac3f8290debec3ea5920
```

The requirements use evaluation-related versions listed in the inspected upstream
requirements. They have **not** been installed or dependency-resolved in this
workspace. The toolkit's dependency constraints may conflict, as the upstream
README notes. Do not silently substitute versions or bypass dependencies; resolve
in this isolated environment, run the checks, and preserve the resulting lock:

```bash
python -m pip check
python -m pip freeze > environment.lock.txt
```

The normalizer and precleaner are loaded from the pinned checkout and verified by
SHA-256 before execution. No upstream code is redistributed in this package; its
LICENSE file was empty at the inspected version. Preserve upstream/KHCWER
attribution and check their terms before redistribution. The dataset README also
leaves image rights unresolved. This package does not redistribute dataset content.

## 1. Prepare the 100 gold pages

If private/gated, authenticate using your own Hub access, for example `hf auth
login`. Do not put credentials in configuration or reports. Resolve and retain
an immutable dataset commit, then export **only** `test`:

```bash
python -c 'from huggingface_hub import HfApi; print(HfApi().dataset_info("CADT-IDRI/khmer-sala-ocr-17k").sha)'
python evaluate.py prepare \
  --dataset CADT-IDRI/khmer-sala-ocr-17k \
  --dataset-revision YOUR_40_CHARACTER_DATASET_COMMIT \
  --out gold100
```

`prepare` requires exactly 100 rows, all `label_source="human"`, unique
`(book_id,page)` pairs and nonempty references. It does not slice a larger split
or offer a train/validation/silver override. The model never receives reference
text from the evaluator. Local PNGs are exports of decoded RGB images, not the
original compressed bytes; use the same export for all models.

Outputs: `dataset.json` with provenance and manifest checksum,
`manifest.jsonl` with references and image checksums, and `images/`.
Human verification is asserted by dataset metadata, not independently repeated.
Do not edit these files. Re-run preparation to a new directory after an export
failure; existing directories are rejected to prevent mixing snapshots.

## 2. Choose how to obtain predictions

### A. Score saved predictions (no inference, GPU or image upload)

Create a UTF-8 JSONL file with exactly one prediction for each manifest `id`.
Use the manifest IDs verbatim. IDs encode a JSON book/page pair as a string;
this example is illustrative, not a real dataset page:

```json
{"id":"[\"example-book\",12]","prediction":"អត្ថបទដែលម៉ូដែលស្គាល់"}
```

Prediction order does not matter. Missing, extra or duplicate IDs and nonstring
outputs are rejected. An empty string is valid OCR output and incurs deletion
errors. Ground truth present in upstream output files is ignored; only the gold
manifest supplies references. Upstream prediction IDs may use a different format;
map them explicitly through book_id/page rather than assuming row-order alignment.

Copy `model-info.example.json` to `model-info.json`. Fill in the actual model or
service version and settings. Record known/unknown training-data exposure. These
fields document your run; they do not configure a model automatically.

```bash
python evaluate.py run \
  --data gold100 --upstream upstream \
  --predictions sorya-predictions.jsonl \
  --model-info model-info.json --out runs/sorya-v1
```

### B. Run your own SORYA-AI adapter

Create a Python module such as `my_sorya.py` in this directory:

```python
def create(config):
    client = YourActualClient(config)  # initialize once, outside the per-page loop
    class Adapter:
        def predict(self, image, metadata):
            # image is a PIL RGB image. metadata contains id, book_id and page only.
            # Return the complete OCR text in reading order; no explanations.
            return client.recognize(image)
        def close(self):
            client.close()  # optional
    return Adapter()
```

```bash
python evaluate.py run \
  --data gold100 --upstream upstream \
  --adapter my_sorya:create --config my-sorya.json \
  --model-info model-info.json --out runs/sorya-v1
```

`adapters:http` is an included generic HTTP example. Its contract is POST JSON
`{image_base64,mime_type,prompt}` and response JSON `{text}`. **This is not a claim
about SORYA-AI's existing API.** Adapt it to the real service and explicitly set
that endpoint in a copy of `http.example.json`. It transmits page images to the
configured endpoint only when you invoke it. `token_env` names an environment
variable, keeping the credential out of configuration. Omit it for unauthenticated
local endpoints. No API request was made here.

### C. Optional DeepSeek baseline / LoRA wrapper

`adapters:deepseek` wraps the reviewed upstream `DeepSeekOCREngine`. Follow the
upstream instructions for its separate GPU environment, including the PyTorch,
Transformers and PEFT versions; evaluation dependencies alone are insufficient.
Download base weights and optional LoRA to **local immutable snapshots**, recording
their Hub commits in model-info.json. Fill in `deepseek.example.json`. Omit
`adapter_path` for the baseline.

```bash
python evaluate.py run \
  --data gold100 --upstream upstream \
  --adapter adapters:deepseek --config deepseek-config.json \
  --model-info deepseek-model-info.json --out runs/deepseek-lora
```

The wrapper checks the upstream HEAD and inference source hash. The upstream
engine uses `trust_remote_code=True`, CUDA BF16 and eager attention. Review that
model code and pin the local snapshots before using it. Its `result.mmd` output
is returned as text; there is no model-specific cleanup here. The wrapper has
not been exercised against a GPU or weights in this workspace.

## Metrics and normalization

For each page, calculate exact unit-cost Levenshtein edit distance. Corpus scores
are **micro-averages**:

`CER = sum(character edits per page) / sum(reference character counts per page)`

`WER = sum(word-token edits per page) / sum(reference word counts per page)`

Page boundaries are preserved. Rates are not averaged per page or computed by
concatenating pages, and are not capped at 100%. CER counts Unicode code points,
not grapheme clusters. Ratios and percentages are both saved, with edit counts
and denominators for auditing.

| Metric | Processing on both reference and hypothesis |
| --- | --- |
| `cer_upstream` | Upstream precleaner, then `khmer_norm.clean_and_normalize`, then code-point distance |
| `wer_upstream` | Upstream precleaner, `word_segment(word_type="com", rt_type="list")`, normalize each token, then token distance |
| `cer_nfc_conservative` | NFC and whitespace collapse only, then code-point distance; retains zero-width characters and Khmer spelling distinctions |

The upstream normalizer does more than Unicode NFC: it maps `្ត` to `្ដ`, removes
whitespace/U+200B/U+200C/U+200D, reorders Khmer syllables, merges certain vowel
sequences and can collapse repeated chunks. Its precleaner also converts actual
and literal `\n` sequences to spaces and removes runs of periods. This can hide
some recognition differences or alter literal LaTeX sequences. The conservative
CER diagnostic exposes some of this effect; it is a separate protocol, not a
replacement paper score. No case folding or Khmer-to-Arabic digit mapping is added.

For WER, segmentation occurs **before** whitespace removal. Splitting Khmer only
on spaces is not used as a real WER implementation. The upstream token-count
reference denominator is preserved even when normalization empties tokens. The
reported toolkit version and any separately downloaded segmenter assets must be
held fixed across models. There is no whitespace-token fallback if it is missing.

LaTeX, Latin text, punctuation and `(image|figure)` placeholders stay in the
full-text task. This follows the code's evaluation pipeline, not the README's
natural-language-only descriptive word statistics. Do not compare denominators
with those statistics as a correctness test. No layout/box metric is computed.

If a normalized reference is empty, a differing prediction produces JSON `null`
for that page rate rather than a fabricated percentage; its insertion edits still
contribute to the corpus numerator. This explicit edge-case convention differs
from upstream's zero-denominator fallback of 1.0. Nonempty-reference behavior
matches the inspected algorithm. No error is hidden by clipping.

## Outputs and fair comparisons

Each new run directory contains:

- `predictions.jsonl`: complete raw model text, keyed by page ID; reusable for rescoring.
- `per_page.jsonl`: raw references/predictions, edit counts, denominators and rates.
- `report.json`: corpus metrics, 100-page coverage, dataset SHA/fingerprint,
  manifest and prediction hashes, evaluator/upstream/adapter source hashes,
  environment/package versions, model metadata, seed and warnings.

A failed inference run may leave `predictions.partial.jsonl`. It never writes a
complete report. Fix the issue and run to a **new** directory. No page is silently
skipped; no partial-run CER is presented as a benchmark. Empty OCR output is
scored rather than excluded. Scoring supplied predictions sets
`inference_executed=false` explicitly.

To compare SORYA-AI with another model, use the same prepared manifest, segmenter,
metric source and image exports. Run each with separate model metadata and output
directory. Compare the `metrics` objects only after matching manifest hashes.
Record prompt, image resizing/cropping, quantization, max output length, runtime
seed, base/adapter commits and test exposure. Inspect high-error pages for
truncation, reading-order errors, equations and normalizer effects. Freeze choices
before comparing models; avoid selecting a prompt on these gold references.

`--seed` seeds Python's random module only. Each adapter must set its own NumPy,
Torch and generation controls; remote inference may remain nondeterministic.
Reproducibility here means pinned inputs, explicit protocol, saved predictions and
auditable rescoring, not guaranteed bit-identical inference across GPU hardware
or changing hosted APIs. Preserve the source, environment lock, model snapshots
and any tokenizer resources with your internal experiment record.

## Verification performed for this deliverable

```bash
SALA_UPSTREAM="$PWD/upstream" python -m unittest -v
python evaluate.py --help
```

**13 tests passed** in the authoring workspace, including exact distance cases,
empty outputs, rates above 100%, micro-aggregation, gold-only guards, duplicate and
missing IDs, manifest tampering, a synthetic 100-row scoring flow, inference
failure handling and inspected upstream Khmer-normalization behavior. The tests
use synthetic inputs and a stub segmenter, not the real toolkit. The source files
used for normalization checks matched upstream Git blob hashes at the recorded
commit. Without `SALA_UPSTREAM`, those two source-dependent tests are skipped.
Pillow is needed for the small synthetic image fixtures.

The real dataset, Khmer tokenizer package, model weights, CUDA inference and
HTTP service were **not** executed. Consequently this package contains **no
measured Khmer OCR accuracy**, and real-tokenizer integration must be checked in
the target environment. A completed real run should contain 100 pages and a
nonzero WER reference count. Inspect tokenizer outputs on a few Khmer sentences
before running inference, without tuning OCR against gold references.
