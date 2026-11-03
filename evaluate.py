#!/usr/bin/env python3
"""Gold-only Khmer Sala OCR benchmark. Python 3.10+. See README.md."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

UPSTREAM_COMMIT = 'a2b7e25fbe8b3f46f927ac3f8290debec3ea5920'
UPSTREAM_HASHES = {'khmer_norm.py': 'cf63b82f7bd5ff1841468ca514b25ea355158c352ae36e9e07b9aa88492e656e', 'preprocessing.py': '956544f71b9d04c98e21eefc21df5bdcdd14fa8a087034a69ed74b2bbff1b803'}
WARNINGS = [
    'CONTAMINATION RISK: page-level splits; all 55 test books also occur in train. '
    'No exact page overlap is documented, but this is not unseen-book evaluation. '
    'Model pretraining/test exposure is unknown; record it for each model.',
    'DOMAIN LIMIT: printed educational textbooks only; no handwriting, scene text, '
    'or tables. These scores cannot establish SORYA-AI handwriting accuracy.',
    'GOLD LIMIT: 100 pages corrected by one annotator; no inter-annotator agreement '
    'or stratified random sampling. Only 10 pages contain equations.',
    'NORMALIZATION: upstream changes some Khmer spellings, removes whitespace and '
    'zero-width characters, reorders syllables and removes repeated periods. '
    'Inspect conservative NFC CER too. Scores are full-text, including LaTeX.',
    'VERSION MISMATCH: Drive README names 17k; GitHub names 16k. '
    'Do not substitute silently or compare scores across different manifests.',
]


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return digest_bytes(Path(path).read_bytes())


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(canonical(r) + '\n' for r in rows), encoding='utf-8')


def page_id(row):
    # A JSON pair avoids collisions between underscores/other separators in book IDs.
    return canonical([str(row['book_id']), int(row['page'])])


def validate_rows(rows, expected=100):
    if len(rows) != expected:
        raise ValueError(f'Require exactly {expected} test pages, got {len(rows)}')
    ids = set()
    for row in rows:
        if row.get('split') != 'test' or row.get('label_source') != 'human':
            raise ValueError('Only test rows with label_source="human" are allowed')
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Missing or empty reference')
        if not isinstance(row.get('book_id'), str) or not row['book_id']:
            raise ValueError('Missing book_id')
        if not isinstance(row.get('page'), int) or isinstance(row['page'], bool):
            raise ValueError('page must be an integer')
        identity = page_id(row)
        if identity in ids:
            raise ValueError(f'Duplicate page: {identity}')
        ids.add(identity)
    return ids


def prepare(args):
    """Download ONLY the test split; never select 100 rows from another split."""
    if not re.fullmatch(r'[0-9a-fA-F]{40}', args.dataset_revision):
        raise ValueError('--dataset-revision must be an immutable 40-digit Hub commit SHA')
    from datasets import load_dataset
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    ds = load_dataset(args.dataset, name='default', split='test',
                      revision=args.dataset_revision)
    # Validate before writing any data. PIL images remain embedded in the Hub files.
    rows = [{k: row[k] for k in ('book_id', 'page', 'text', 'label_source')}
            | {'split': 'test'} for row in ds]
    validate_rows(rows)
    for index, row in enumerate(rows):
        image = ds[index]['image'].convert('RGB')
        rel = f'images/{index:03d}.png'
        image.save(out / rel, format='PNG')
        row.update(image_file=rel, image_sha256=file_hash(out / rel),
                   id=page_id(row))
    write_jsonl(out / 'manifest.jsonl', rows)
    dump(out / 'dataset.json', {
        'schema': 1, 'origin': 'huggingface-test-export', 'dataset': args.dataset,
        'dataset_revision': args.dataset_revision, 'config': 'default', 'split': 'test',
        'pages': len(rows), 'books': len({r['book_id'] for r in rows}),
        'manifest_sha256': file_hash(out / 'manifest.jsonl'),
        'datasets_fingerprint': getattr(ds, '_fingerprint', None),
        'label_verification': 'Dataset label_source metadata; not independently re-annotated',
        'warnings': WARNINGS,
    })
    print(f'Prepared {len(rows)} human-labelled test pages at {out}')


def load_manifest(folder, verify_images=False):
    folder = Path(folder)
    meta = json.loads((folder / 'dataset.json').read_text(encoding='utf-8'))
    if meta.get('origin') != 'huggingface-test-export' or meta.get('split') != 'test':
        raise ValueError('Use the prepare command to create the gold-test manifest')
    if file_hash(folder / 'manifest.jsonl') != meta['manifest_sha256']:
        raise ValueError('Manifest checksum mismatch')
    rows = jsonl(folder / 'manifest.jsonl')
    validate_rows(rows)
    for row in rows:
        if row.get('id') != page_id(row):
            raise ValueError('Page ID mismatch')
        if verify_images:
            path = (folder / row['image_file']).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError('Image path escapes dataset folder')
            if file_hash(path) != row['image_sha256']:
                raise ValueError(f'Image checksum mismatch for {row["id"]}')
    return rows, meta


def levenshtein(a, b):
    """Exact unit-cost edit distance over code points or word tokens."""
    try:
        from Levenshtein import distance
        return int(distance(a, b))
    except ImportError:
        pass
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def stats(ref, hyp):
    distance, length = levenshtein(ref, hyp), len(ref)
    # No 100% cap. Undefined empty-reference rates are explicit JSON null.
    ratio = distance / length if length else (0.0 if distance == 0 else None)
    return {'edits': distance, 'reference_units': length, 'ratio': ratio,
            'percent': ratio * 100 if ratio is not None else None}


def aggregate(records):
    return {name: rate_from_counts(sum(r[name]['edits'] for r in records),
                                  sum(r[name]['reference_units'] for r in records))
            for name in records[0]}


def rate_from_counts(edits, units):
    ratio = edits / units if units else (0.0 if edits == 0 else None)
    return dict(edits=edits, reference_units=units, ratio=ratio,
                percent=100 * ratio if ratio is not None else None)


def conservative(text):
    """NFC + whitespace collapse only; retain Khmer distinctions and zero-width chars."""
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Metrics:
    def __init__(self, upstream):
        root = Path(upstream) / 'src' / 'utils'
        for filename, expected in UPSTREAM_HASHES.items():
            if file_hash(root / filename) != expected:
                raise ValueError(f'Upstream source differs from inspected version: {filename}')
        self.normalize = load_source(root / 'khmer_norm.py', '_sala_norm').clean_and_normalize
        self.clean = load_source(root / 'preprocessing.py', '_sala_clean').clean_text_pipeline
        from khmer_nlp_toolkits.text.segmentation import word_segment
        self.segment = lambda t: word_segment(t, word_type='com', rt_type='list')

    def words(self, text):
        # Match upstream: segment cleaned text FIRST, normalize each token SECOND.
        # Rejoin and split matches its handling of tokens emptied by normalization.
        tokens = self.segment(text)
        if not isinstance(tokens, list) or not all(isinstance(w, str) for w in tokens):
            raise TypeError('Khmer segmenter must return list[str]')
        return ' '.join(self.normalize(w) for w in tokens).split(), len(tokens)

    def score(self, ref, hyp):
        r, h = self.clean(ref), self.clean(hyp)
        rw, rlen = self.words(r)
        hw, _ = self.words(h)
        return {
            'cer_upstream': stats(self.normalize(r), self.normalize(h)),
            # Preserve original upstream denominator (segment count before normalization).
            'wer_upstream': rate_from_counts(levenshtein(rw, hw), rlen),
            'cer_nfc_conservative': stats(conservative(ref), conservative(hyp)),
        }


def prediction_map(records, required_ids):
    result = {}
    for row in records:
        identity = row.get('id')
        if not isinstance(identity, str) or identity in result:
            raise ValueError('Missing or duplicate prediction ID')
        if not isinstance(row.get('prediction'), str):
            raise ValueError('prediction must be a string; empty OCR output is allowed')
        result[identity] = row['prediction']
    if set(result) != required_ids:
        raise ValueError(f'Prediction IDs differ: {len(required_ids - set(result))} missing, '
                         f'{len(set(result) - required_ids)} extra')
    return result


def environment():
    packages = {}
    for name in ('datasets', 'Pillow', 'khmer-nlp-toolkits', 'python-Levenshtein',
                 'Levenshtein', 'torch', 'transformers', 'peft'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(),
                unicode_version=unicodedata.unidata_version, packages=packages)


def run(args):
    for warning in WARNINGS:
        print('WARNING: ' + warning, file=sys.stderr)
    rows, meta = load_manifest(args.data, verify_images=bool(args.adapter))
    metric = Metrics(args.upstream)
    # Existing output directories are rejected to prevent mixing model runs.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else {}
    if not isinstance(config, dict):
        raise ValueError('Adapter config must be a JSON object')
    model_info = json.loads(Path(args.model_info).read_text(encoding='utf-8'))
    for key in ('name', 'revision', 'training_data_exposure', 'generation_settings'):
        if key not in model_info:
            raise ValueError(f'Model info missing {key}')
    if not isinstance(model_info['revision'], str) or not model_info['revision'].strip() or 'REPLACE_' in model_info['revision']:
        raise ValueError('Set an actual model/service revision in model info')
    random.seed(args.seed)
    started = time.perf_counter()
    predictions = []
    adapter_source_sha256 = None
    if args.predictions:
        mapping = prediction_map(jsonl(args.predictions), {r['id'] for r in rows})
        predictions = [dict(id=r['id'], prediction=mapping[r['id']]) for r in rows]
    else:
        from PIL import Image
        module, factory = args.adapter.split(':', 1)
        adapter_module = importlib.import_module(module)
        adapter_file = getattr(adapter_module, '__file__', None)
        adapter_source_sha256 = file_hash(adapter_file) if adapter_file else None
        backend = getattr(adapter_module, factory)(config)
        try:
            # Failures abort; no skipped pages and no success-only denominator.
            with (out / 'predictions.partial.jsonl').open('x', encoding='utf-8') as stream:
                for index, row in enumerate(rows):
                    with Image.open(Path(args.data) / row['image_file']) as im:
                        image = im.convert('RGB')
                    # Do not pass reference text or label_source into model inference.
                    prediction = backend.predict(image, {'id': row['id'],
                                                         'book_id': row['book_id'],
                                                         'page': row['page']})
                    if not isinstance(prediction, str):
                        raise TypeError('OCR adapter must return str')
                    record = dict(id=row['id'], prediction=prediction)
                    predictions.append(record)
                    stream.write(canonical(record) + '\n')
                    stream.flush()
                    print(f'OCR {index + 1}/100', file=sys.stderr)
        finally:
            if hasattr(backend, 'close'):
                backend.close()
    prediction_map(predictions, {r['id'] for r in rows})
    write_jsonl(out / 'predictions.jsonl', predictions)
    (out / 'predictions.partial.jsonl').unlink(missing_ok=True)
    page_scores = []
    for row, pred in zip(rows, predictions):
        page_scores.append(dict(id=row['id'], book_id=row['book_id'], page=row['page'],
                                reference=row['text'], prediction=pred['prediction'],
                                metrics=metric.score(row['text'], pred['prediction'])))
    write_jsonl(out / 'per_page.jsonl', page_scores)
    # Hash adapter config but never persist it; it may contain endpoint secrets.
    report = dict(schema=1, status='complete-100-page-test', pages=len(rows),
                  completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  adapter_source_sha256=adapter_source_sha256,
                  dataset=meta, model=model_info,
                  inference_executed=bool(args.adapter),
                  prediction_source='adapter' if args.adapter else 'supplied-predictions',
                  adapter=args.adapter, adapter_config_sha256=digest_bytes(canonical(config).encode()),
                  seed=args.seed, seed_scope='Python random only; adapter must configure its runtime',
                  environment=environment(), upstream_commit=UPSTREAM_COMMIT,
                  upstream_source_sha256=UPSTREAM_HASHES,
                  evaluator_sha256=file_hash(__file__),
                  predictions_sha256=file_hash(out / 'predictions.jsonl'),
                  elapsed_seconds=time.perf_counter() - started,
                  aggregation='sum(page edit distances) / sum(page reference lengths)',
                  cer_unit='Unicode code point; not grapheme cluster',
                  wer_segmenter='khmer-nlp-toolkits: word_type=com, rt_type=list; segment before normalize',
                  content_policy='Full text: retain punctuation, Latin, LaTeX and figure placeholders',
                  metrics=aggregate([r['metrics'] for r in page_scores]), warnings=WARNINGS)
    dump(out / 'report.json', report)
    print(json.dumps(report['metrics'], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Export exactly the human-labelled test split')
    prep.add_argument('--dataset', default='CADT-IDRI/khmer-sala-ocr-17k')
    prep.add_argument('--dataset-revision', required=True)
    prep.add_argument('--out', required=True)
    prep.set_defaults(func=prepare)
    evaluate = commands.add_parser('run', help='Infer or score saved predictions on all 100 pages')
    evaluate.add_argument('--data', required=True)
    evaluate.add_argument('--upstream', required=True, help='Pinned khmer-sala-ocr checkout')
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument('--predictions', help='JSONL of id/prediction pairs')
    source.add_argument('--adapter', help='module:factory; returns object with predict(image, metadata)')
    evaluate.add_argument('--config', help='Adapter-specific JSON configuration')
    evaluate.add_argument('--model-info', required=True)
    evaluate.add_argument('--seed', type=int, default=42)
    evaluate.add_argument('--out', required=True)
    evaluate.set_defaults(func=run)
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as error:
        parser.exit(1, f'ERROR: {type(error).__name__}: {error}\n'
                    'No complete benchmark result is valid unless report.json was produced.\n')


if __name__ == '__main__':
    main()
