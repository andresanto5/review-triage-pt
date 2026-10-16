"""Retrospective test diagnostics; title ablation uses validation ONLY."""
import hashlib
import json
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from train import ROOT,prepare,split_temporal,metrics,SEED

def aggregate_errors(frame, group):
    return frame.groupby(group,observed=True).agg(rows=('error','size'),errors=('error','sum'),error_rate=('error','mean'),false_positives=('fp','sum'),false_negatives=('fn','sum')).reset_index()

def flatten(m,prefix):
    return {f'{prefix}_macro_f1':m['classification']['macro avg']['f1-score'],f'{prefix}_negative_precision':m['classification']['1']['precision'],f'{prefix}_negative_recall':m['classification']['1']['recall'],f'{prefix}_average_precision':m['average_precision'],f'{prefix}_review_fraction':m['review_fraction']}

def main():
    reports=ROOT/'reports'; artifacts=ROOT/'artifacts'
    df,audit=prepare(ROOT/'data/raw/B2W-Reviews01.csv')
    train,valid,test=split_temporal(df)
    baseline=joblib.load(artifacts/'baseline.joblib')
    original=json.loads((reports/'baseline.json').read_text(encoding='utf-8'))
    assert baseline['source_sha256']==audit['source_sha256']==original['audit']['source_sha256']
    assert len(test)==original['split']['test']['rows']
    pred=baseline['pipeline'].predict_proba(test.text)[:,1]>=baseline['threshold']
    check=test.assign(predicted=pred,error=pred!=test.target,fp=pred & test.target.eq(0),fn=(~pred)&test.target.eq(1))
    check['week']=check.date.dt.strftime('%Y-W%W')
    check['length_band']=pd.cut(check.text.str.len(),[0,50,150,400,float('inf')],labels=['10-50','51-150','151-400','401+'])
    files=[]
    for column in ['rating','week','length_band']:
        path=reports/f'errors_by_{column}.csv';aggregate_errors(check,column).to_csv(path,index=False);files.append(path)
    # Identical bodies cannot cross train/validation for the ablation, even
    # when titles differ. Compare both models on this SAME validation subset.
    tr=train.loc[train.body.str.len().ge(10)].copy()
    conflicts=tr.groupby('body').target.nunique()
    tr=tr.loc[~tr.body.isin(conflicts[conflicts.gt(1)].index)].drop_duplicates('body')
    va=valid.loc[valid.body.str.len().ge(10)&~valid.body.isin(set(tr.body))].copy()
    va=va.drop_duplicates('body')
    assert not set(tr.body)&set(va.body)
    assert tr.target.nunique()==va.target.nunique()==2
    # Same training rows, hyperparameters and validation rows for both variants.
    comparisons={}; models={}
    for variant,column in [('title_body','text'),('body_only','body')]:
        model=clone(baseline['pipeline'])
        model.fit(tr[column],tr.target)
        scores=model.predict_proba(va[column])[:,1]
        threshold=max(np.arange(.1,.901,.025),key=lambda t:f1_score(va.target,scores>=t,average='macro'))
        comparisons[variant]=metrics(va.target,scores,threshold)
        models[variant]={'pipeline':model,'threshold':float(threshold)}
    result={'protocol':'Validation-only ablation; thresholds selected and measured on validation, optimistic for model selection. Test not evaluated for either new model.', 'train_rows':len(tr),'validation_rows':len(va),'body_overlap':0,'source_sha256':audit['source_sha256'],'variants':comparisons}
    result_path=reports/'title_ablation_validation.json';result_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    # SQLite metadata + artifacts remain local, ignored by Git.
    mlflow.set_tracking_uri('sqlite:///'+str((artifacts/'mlflow.db').resolve()))
    client=mlflow.MlflowClient()
    experiment=client.get_experiment_by_name('review-triage-pt')
    exp_id=experiment.experiment_id if experiment else client.create_experiment('review-triage-pt',artifact_location=(artifacts/'mlflow').resolve().as_uri())
    mlflow.set_experiment(experiment_id=exp_id)
    runs=[]
    common={'source_sha256':audit['source_sha256'],'seed':SEED,'features_max':50000,'ngram_range':'1,2','class_weight':'balanced','train_script_sha256':hashlib.sha256((ROOT/'src/train.py').read_bytes()).hexdigest()}
    with mlflow.start_run(run_name='baseline_retrospective') as run:
        mlflow.set_tags({'recording':'retrospective import of completed baseline','scope':'historical data, not production','mlflow_version':mlflow.__version__})
        mlflow.log_params(common)
        mlflow.log_metrics(flatten(original['validation'],'validation')|flatten(original['test'],'test'))
        mlflow.log_artifact(str(reports/'baseline.json'))
        for path in files: mlflow.log_artifact(str(path),'error_diagnostics')
        mlflow.log_artifact(str(artifacts/'baseline.joblib'),'model')
        runs.append(run.info.run_id)
    for variant in comparisons:
        with mlflow.start_run(run_name=variant+'_validation_ablation') as run:
            mlflow.set_tags({'scope':'validation-only model selection','test_evaluated':'false','mlflow_version':mlflow.__version__})
            mlflow.log_params(common|{'variant':variant,'training_rows':len(tr),'validation_rows':len(va),'threshold':models[variant]['threshold']})
            mlflow.log_metrics(flatten(comparisons[variant],'validation'))
            mlflow.log_artifact(str(result_path))
            path=artifacts/f'{variant}_validation.joblib';joblib.dump(models[variant],path)
            mlflow.log_artifact(str(path),'model');runs.append(run.info.run_id)
    for runid in runs: assert client.get_run(runid).info.status=='FINISHED'
    (reports/'mlflow_runs.json').write_text(json.dumps({'run_ids':runs,'tracking':'local SQLite','verified_status':'FINISHED','mlflow_version':mlflow.__version__},indent=2),encoding='utf-8')
    print(json.dumps({'runs':runs,'ablation':result},indent=2))

if __name__=='__main__':main()
