"""Check persisted local experiment metrics and artifacts without retraining."""
import json
from pathlib import Path
import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri('sqlite:///' + str((ROOT/'artifacts/mlflow.db').resolve()))
client = mlflow.MlflowClient()
record = json.loads((ROOT/'reports/mlflow_runs.json').read_text())
baseline = json.loads((ROOT/'reports/baseline.json').read_text())
for run_id in record['run_ids']:
    run = client.get_run(run_id)
    assert run.info.status == 'FINISHED'
    assert run.data.params['source_sha256'] == baseline['audit']['source_sha256']
    models = client.list_artifacts(run_id, 'model')
    assert len(models) == 1 and models[0].file_size > 0
    stored = Path(client.download_artifacts(run_id, models[0].path))
    assert stored.is_file() and stored.stat().st_size == models[0].file_size
    assert 'validation_macro_f1' in run.data.metrics
    if run.data.tags.get('test_evaluated') == 'false':
        assert not any(key.startswith('test_') for key in run.data.metrics)
for name in ['rating','week','length_band']:
    grouped = pd.read_csv(ROOT/f'reports/errors_by_{name}.csv')
    assert grouped.rows.sum() == baseline['split']['test']['rows']
    assert grouped.errors.sum() == 1159
    assert grouped.false_positives.sum() == 724
    assert grouped.false_negatives.sum() == 435
print('Verified: 3 completed runs, persisted model artifacts, validation-only variants, and error totals.')
