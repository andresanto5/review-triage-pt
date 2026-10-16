"""Real HTTP contract tests against the frozen local model, with synthetic text."""
import hashlib
import asyncio
import json
from pathlib import Path
import socket
import sys
import threading
import time
import tempfile
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import joblib
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from api import create_app
from train import normalize


class StartupTests(unittest.TestCase):
    def test_missing_model_prevents_startup(self):
        async def start():
            app = create_app(Path('/nonexistent/review-triage-model.joblib'))
            async with app.router.lifespan_context(app):
                self.fail('Missing model should prevent readiness')
        with self.assertRaises(FileNotFoundError):
            asyncio.run(start())

    def test_invalid_threshold_prevents_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'invalid.joblib'
            joblib.dump({'threshold': float('nan')}, path)
            async def start():
                app = create_app(path)
                async with app.router.lifespan_context(app):
                    self.fail('Invalid model should prevent readiness')
            with self.assertRaisesRegex(ValueError, 'Invalid model threshold'):
                asyncio.run(start())


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = joblib.load(ROOT/'artifacts/baseline.joblib')
        cls.sock = socket.socket()
        cls.sock.bind(('127.0.0.1', 0))
        cls.base = f'http://127.0.0.1:{cls.sock.getsockname()[1]}'
        cls.server = uvicorn.Server(uvicorn.Config(create_app(), log_level='error', access_log=False))
        cls.thread = threading.Thread(target=cls.server.run, kwargs={'sockets': [cls.sock]}, daemon=True)
        cls.thread.start()
        for _ in range(200):
            if cls.server.started:
                return
            if not cls.thread.is_alive():
                raise RuntimeError('API failed to start')
            time.sleep(.05)
        raise RuntimeError('API startup timed out')

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=5)
        cls.sock.close()

    def send(self, payload, content_type='application/json', raw=False):
        body = payload if raw else json.dumps(payload).encode()
        req = Request(self.base+'/predict', data=body, headers={'Content-Type': content_type})
        try:
            response = urlopen(req, timeout=10)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read())

    def test_health_model_fingerprint(self):
        with urlopen(self.base+'/health') as response:
            result = json.loads(response.read())
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['model_sha256'], hashlib.sha256((ROOT/'artifacts/baseline.joblib').read_bytes()).hexdigest())

    def test_prediction_matches_frozen_pipeline(self):
        for title, text in [('Péssimo', 'O produto chegou quebrado e não funciona.'), ('Ótimo', 'Gostei muito do produto e da entrega.'), ('', '  AVALIAÇÃO com https://example.com e contato teste@example.com  ')]:
            status, result = self.send({'title': title, 'text': text})
            self.assertEqual(status, 200)
            expected = float(self.bundle['pipeline'].predict_proba([normalize(title+' '+text)])[0, 1])
            self.assertAlmostEqual(result['negative_score'], expected, places=12)
            self.assertEqual(result['review_required'], expected >= self.bundle['threshold'])
            self.assertEqual(result['threshold'], self.bundle['threshold'])
            self.assertFalse(result['score_calibrated'])
            self.assertNotIn('text', result)

    def test_invalid_payloads_do_not_echo_inputs(self):
        marker = 'PRIVATE_REVIEW_MARKER'
        for payload in [{}, {'text': None}, {'text': 123}, {'text': 'curto'}, {'text': ' '*20}, {'text': marker, 'rating': 1}, {'text': marker, 'title': 'x'*501}, {'text': marker*1000}]:
            status, result = self.send(payload)
            self.assertEqual(status, 422)
            self.assertNotIn(marker, json.dumps(result))
        status, result = self.send(b'{"text": "PRIVATE_REVIEW_MARKER",', raw=True)
        self.assertEqual(status, 422)
        self.assertNotIn(marker, json.dumps(result))

    def test_content_type_and_actual_body_limit(self):
        self.assertEqual(self.send(b'anything', 'text/plain', raw=True)[0], 415)
        self.assertEqual(self.send(b' '*65537, raw=True)[0], 413)


if __name__ == '__main__':
    unittest.main()
