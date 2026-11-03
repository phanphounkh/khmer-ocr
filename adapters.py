"""Optional adapters. No models or network connections are opened on import."""
import base64
import io
import os
import urllib.request
from pathlib import Path


class HttpOCR:
    """Generic SORYA-AI integration contract; adapt payload to your actual API."""
    def __init__(self, config):
        self.config = config
        if not config['url'].startswith(('http://', 'https://')):
            raise ValueError('HTTP(S) endpoint required')

    def predict(self, image, metadata):
        import json
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        body = json.dumps({
            'image_base64': base64.b64encode(buffer.getvalue()).decode('ascii'),
            'mime_type': 'image/png',
            'prompt': self.config.get('prompt', 'Transcribe the page in reading order. Return plain text only.'),
        }).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        if self.config.get('token_env'):
            headers['Authorization'] = 'Bearer ' + os.environ[self.config['token_env']]
        request = urllib.request.Request(self.config['url'], data=body, headers=headers,
                                         method='POST')
        with urllib.request.urlopen(request, timeout=self.config.get('timeout_seconds', 180)) as response:
            payload = json.load(response)
        text = payload['text']
        if not isinstance(text, str):
            raise TypeError('Endpoint must return a JSON string in text')
        return text


def http(config):
    return HttpOCR(config)


def deepseek(config):
    """Wrap inspected upstream engine; user supplies prepared local model snapshots.

    The upstream engine requires CUDA, torch/transformers/peft and executes model
    remote code. See README. This wrapper has NOT been GPU-tested here.
    """
    from evaluate import load_source, UPSTREAM_COMMIT
    import subprocess
    root = Path(config['upstream']).resolve()
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                   text=True).strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError('DeepSeek wrapper requires the documented upstream commit')
    engine_path = root / 'src/utils/inference_engine.py'
    expected = '2d480bc08b5dac6c6043c3ce02c4f64664ac311dda86e9bdcaf061106ae5fef2'
    from evaluate import file_hash
    if file_hash(engine_path) != expected:
        raise ValueError('DeepSeek engine source differs from reviewed source')
    if not Path(config['model_path']).is_dir():
        raise ValueError('model_path must be an existing local pinned model snapshot')
    module = load_source(engine_path, '_sala_inference')
    engine = module.DeepSeekOCREngine(config['model_path'], config.get('adapter_path'))

    class Wrapper:
        def predict(self, image, metadata):
            return engine.predict(image, **config.get('inference', {}))

        def close(self):
            engine.close()

    return Wrapper()
