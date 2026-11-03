"""Offline synthetic tests; never measures OCR accuracy or real Khmer segmentation."""
import argparse
import contextlib
import io
import json
import os
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate as e


def fake_metrics():
    metric = e.Metrics.__new__(e.Metrics)
    metric.clean = lambda t: t.strip()
    metric.normalize = lambda t: ''.join(t.split())
    metric.segment = str.split
    return metric


def fixture(folder):
    from PIL import Image
    (folder / 'images').mkdir()
    Image.new('RGB', (1, 1), 'white').save(folder / 'images/one.png')
    rows = [dict(book_id=f'synthetic-{i}', page=1, split='test', label_source='human',
                 text='abc def', image_file='images/one.png',
                 image_sha256=e.file_hash(folder / 'images/one.png')) for i in range(100)]
    for row in rows:
        row['id'] = e.page_id(row)
    e.write_jsonl(folder / 'manifest.jsonl', rows)
    e.dump(folder / 'dataset.json', dict(origin='huggingface-test-export', split='test',
           dataset='SYNTHETIC_TEST_FIXTURE_NOT_SALA',
           manifest_sha256=e.file_hash(folder / 'manifest.jsonl')))
    e.dump(folder / 'model.json', dict(name='synthetic', revision='test-only',
           training_data_exposure='synthetic', generation_settings={}))
    return rows


class Core(unittest.TestCase):
    def test_known_edit_distances(self):
        self.assertEqual(e.levenshtein('kitten', 'sitting'), 3)
        self.assertEqual(e.levenshtein(['ក', 'ខ'], ['ក', 'គ']), 1)
        self.assertEqual(e.levenshtein('', 'abc'), 3)
        self.assertEqual(e.levenshtein('abc', ''), 3)
        self.assertEqual(e.levenshtein('ក', 'ក'), 0)

    def test_distance_properties(self):
        rng = random.Random(42)
        for _ in range(100):
            a = ''.join(rng.choices('abកខ', k=rng.randrange(9)))
            b = ''.join(rng.choices('abកខ', k=rng.randrange(9)))
            d = e.levenshtein(a, b)
            self.assertEqual(d, e.levenshtein(b, a))
            self.assertGreaterEqual(d, abs(len(a) - len(b)))
            self.assertLessEqual(d, max(len(a), len(b)))

    def test_no_clamp_and_empty(self):
        self.assertEqual(e.stats('a', 'bbbb')['ratio'], 4)
        self.assertIsNone(e.stats('', 'a')['ratio'])
        self.assertEqual(e.stats('', '')['ratio'], 0)

    def test_micro_is_not_macro(self):
        scores = [{'cer': e.stats('a', 'b')}, {'cer': e.stats('abcdefghi', 'abcdefghi')}]
        self.assertAlmostEqual(e.aggregate(scores)['cer']['ratio'], 0.1)

    def test_wer_uses_word_tokens(self):
        score = fake_metrics().score('cat dog', 'cat fog')
        self.assertEqual(score['wer_upstream']['edits'], 1)
        self.assertEqual(score['wer_upstream']['reference_units'], 2)
        self.assertEqual(score['wer_upstream']['ratio'], 0.5)

    def test_nfc_preserves_khmer_distinctions(self):
        self.assertNotEqual(e.conservative('្ត'), e.conservative('្ដ'))
        self.assertEqual(e.conservative('e\u0301\n a'), 'é a')
        self.assertIn('\u200b', e.conservative('ក\u200bខ'))

    def test_gold_guards(self):
        row = dict(book_id='b', page=1, split='test', label_source='human', text='ក')
        self.assertEqual(len(e.validate_rows([row], expected=1)), 1)
        for bad in [dict(row, label_source='gemini'), dict(row, split='train'),
                    dict(row, text=''), dict(row, page=True)]:
            with self.assertRaises(ValueError):
                e.validate_rows([bad], expected=1)
        with self.assertRaises(ValueError):
            e.validate_rows([row])
        with self.assertRaises(ValueError):
            e.validate_rows([row, row], expected=2)

    def test_prediction_alignment(self):
        records = [dict(id='b', prediction=''), dict(id='a', prediction='ក')]
        self.assertEqual(e.prediction_map(records, {'a', 'b'})['b'], '')
        for bad in [records[:1], records + [records[0]], [dict(id='a', prediction=None)]]:
            with self.assertRaises(ValueError):
                e.prediction_map(bad, {'a', 'b'})

    def test_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder)
            e.load_manifest(folder, verify_images=True)
            with (folder / 'manifest.jsonl').open('a') as f:
                f.write('\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                e.load_manifest(folder)

    def test_synthetic_score_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = fixture(folder)
            e.write_jsonl(folder / 'pred.jsonl', [dict(id=r['id'], prediction='abc xyz')
                                                  for r in reversed(rows)])
            args = argparse.Namespace(data=str(folder), upstream='unused', adapter=None,
                    predictions=str(folder / 'pred.jsonl'), config=None,
                    model_info=str(folder / 'model.json'), seed=42, out=str(folder / 'out'))
            with patch.object(e, 'Metrics', return_value=fake_metrics()), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                e.run(args)
            report = json.loads((folder / 'out/report.json').read_text())
            self.assertEqual(report['pages'], 100)
            self.assertFalse(report['inference_executed'])
            self.assertEqual(report['metrics']['wer_upstream']['edits'], 100)
            self.assertEqual(report['metrics']['wer_upstream']['reference_units'], 200)
            self.assertEqual(report['metrics']['cer_upstream']['ratio'], .5)

    def test_failed_inference_has_no_report_and_no_reference_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fixture(folder)
            class Broken:
                def predict(self, image, metadata):
                    if 'text' in metadata or 'reference' in metadata:
                        raise AssertionError('Reference leaked')
                    raise RuntimeError('synthetic failure')
                def close(self):
                    pass
            module = types.ModuleType('synthetic_adapter')
            module.factory = lambda config: Broken()
            args = argparse.Namespace(data=str(folder), upstream='unused',
                    adapter='synthetic_adapter:factory', predictions=None, config=None,
                    model_info=str(folder / 'model.json'), seed=42, out=str(folder / 'out'))
            with patch.dict('sys.modules', {'synthetic_adapter': module}), \
                 patch.object(e, 'Metrics', return_value=fake_metrics()), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                    e.run(args)
            self.assertFalse((folder / 'out/report.json').exists())


@unittest.skipUnless(os.environ.get('SALA_UPSTREAM'), 'Set SALA_UPSTREAM for source-normalization checks')
class Upstream(unittest.TestCase):
    def metric(self):
        # Dependency stub tests the inspected normalizer/precleaner, NOT Khmer segmentation.
        names = ['khmer_nlp_toolkits', 'khmer_nlp_toolkits.text',
                 'khmer_nlp_toolkits.text.segmentation']
        modules = {name: types.ModuleType(name) for name in names}
        modules[names[-1]].word_segment = lambda text, **kw: text.split()
        with patch.dict('sys.modules', modules):
            return e.Metrics(os.environ['SALA_UPSTREAM'])

    def test_source_hashes_and_khmer_rules(self):
        metric = self.metric()
        self.assertEqual(metric.normalize('ក \u200b ខ'), 'កខ')
        self.assertEqual(metric.normalize('្ត'), metric.normalize('្ដ'))
        self.assertEqual(metric.normalize('កេី'), metric.normalize('កើ'))
        self.assertEqual(metric.clean(' a...\\n b\n'), 'a  b')
        self.assertEqual(metric.score('កេី', 'កើ')['cer_upstream']['ratio'], 0)
        self.assertGreater(metric.score('កេី', 'កើ')['cer_nfc_conservative']['ratio'], 0)

    def test_segment_before_normalize(self):
        metric = self.metric()
        seen = []
        def segment(text):
            seen.append(text)
            return text.split()
        metric.segment = segment
        metric.score('ក ខ', 'ក ខ')
        self.assertEqual(seen, ['ក ខ', 'ក ខ'])


if __name__ == '__main__':
    unittest.main()
