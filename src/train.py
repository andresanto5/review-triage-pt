"""Reproducible temporal baseline. No demographic fields or ratings as features."""
import hashlib
import json
import platform
import re
import unicodedata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score, f1_score

ROOT=Path(__file__).resolve().parents[1]
SEED=42

def normalize(text):
    text=unicodedata.normalize('NFKC',str(text)).lower()
    text=re.sub(r'https?://\S+',' URL ',text)
    text=re.sub(r'\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b',' EMAIL ',text)
    return re.sub(r'\s+',' ',text).strip()

def prepare(path):
    # Actual CSV uses commas, although upstream README mentions semicolons.
    df=pd.read_csv(path,usecols=['submission_date','review_title','review_text','overall_rating'])
    audit={'raw_rows':len(df),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    df['date']=pd.to_datetime(df.submission_date,errors='coerce',utc=True).dt.floor('D')
    df['rating']=pd.to_numeric(df.overall_rating,errors='coerce')
    df['text']=(df.review_title.fillna('')+' '+df.review_text.fillna('')).map(normalize)
    df['body']=df.review_text.fillna('').map(normalize)
    df=df.loc[df.date.notna() & df.rating.isin([1,2,3,4,5]) & df.text.str.len().ge(10)].copy()
    audit['invalid_or_short_removed']=audit['raw_rows']-len(df)
    df['target']=df.rating.le(2).astype(int)
    df['hash']=df.text.map(lambda x:hashlib.sha256(x.encode()).hexdigest())
    conflicts=df.groupby('hash').target.nunique()
    bad=set(conflicts[conflicts.gt(1)].index)
    audit['conflicting_text_groups']=len(bad)
    audit['conflicting_rows_removed']=int(df.hash.isin(bad).sum())
    df=df.loc[~df.hash.isin(bad)].sort_values(['date','hash'])
    before=len(df)
    df=df.drop_duplicates('hash',keep='first')
    audit['duplicate_rows_removed']=before-len(df)
    audit['retained_rows']=len(df)
    audit['negative_rate']=float(df.target.mean())
    return df,audit

def split_temporal(df):
    first=df.date.quantile(.70)
    second=df.date.quantile(.85)
    train=df.loc[df.date.le(first)].copy()
    valid=df.loc[df.date.gt(first)&df.date.le(second)].copy()
    test=df.loc[df.date.gt(second)].copy()
    assert train.date.max()<valid.date.min() and valid.date.max()<test.date.min()
    sets=[set(x.hash) for x in (train,valid,test)]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert all(x.target.nunique()==2 for x in (train,valid,test))
    return train,valid,test

def metrics(y,score,threshold):
    pred=(score>=threshold).astype(int)
    return {'threshold':float(threshold),'classification':classification_report(y,pred,output_dict=True,zero_division=0),
            'confusion_matrix_tn_fp_fn_tp':confusion_matrix(y,pred,labels=[0,1]).ravel().tolist(),
            'average_precision':float(average_precision_score(y,score)),
            'roc_auc':float(roc_auc_score(y,score)), 'review_fraction':float(pred.mean())}

def main():
    source=ROOT/'data/raw/B2W-Reviews01.csv'
    df,audit=prepare(source)
    train,valid,test=split_temporal(df)
    model=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=3,max_df=.98,max_features=50000,sublinear_tf=True)),
                    ('classifier',LogisticRegression(max_iter=500,class_weight='balanced',random_state=SEED))])
    model.fit(train.text,train.target)
    valid_score=model.predict_proba(valid.text)[:,1]
    # Threshold selected exclusively on validation macro-F1. No test tuning.
    thresholds=np.arange(.1,.901,.025)
    chosen=max(thresholds,key=lambda t:f1_score(valid.target,valid_score>=t,average='macro'))
    test_score=model.predict_proba(test.text)[:,1]
    majority=int(train.target.mode().iloc[0])
    naive=np.full(len(test),majority,dtype=float)
    report={'seed':SEED,'python':platform.python_version(),'sklearn':sklearn.__version__,'audit':audit,
        'target':'1 = rating 1 or 2; 0 = rating 3, 4 or 5; proxy from stars, not human sentiment labels',
        'features':'review title and body only; TF-IDF fit on train only',
        'split':{name:{'rows':len(part),'first_day':str(part.date.min().date()),'last_day':str(part.date.max().date()),'negative_rate':float(part.target.mean())} for name,part in zip(['train','validation','test'],[train,valid,test])},
        'validation':metrics(valid.target,valid_score,chosen),
        'test':metrics(test.target,test_score,chosen),
        'majority_baseline_test':metrics(test.target,naive,.5),
        'limitations':['Historical 2018 data; no evidence of current production performance.','Only normalized exact duplicates removed; near duplicates and product overlap may remain.','No demographic attributes used.','Scores are not calibrated probabilities.','No measured business impact or deployment validation.']}
    artifacts=ROOT/'artifacts';artifacts.mkdir(exist_ok=True)
    joblib.dump({'pipeline':model,'threshold':float(chosen),'source_sha256':audit['source_sha256']},artifacts/'baseline.joblib')
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    (reports/'baseline.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    # Error audit exports identifiers and numbers only, never raw review text.
    errors=pd.DataFrame({'hash':test.hash,'actual':test.target,'score':test_score,'predicted':(test_score>=chosen).astype(int)})
    errors.loc[errors.actual.ne(errors.predicted)].to_csv(artifacts/'test_errors_private.csv',index=False)
    summary=f"# Primeiro baseline\n\nLinhas retidas: {len(df):,}.\n\nTreino: {len(train):,}; validação: {len(valid):,}; teste: {len(test):,}.\n\nLimiar escolhido na validação: {chosen:.3f}.\n\nMacro-F1 no teste: {report['test']['classification']['macro avg']['f1-score']:.4f}.\n\nMacro-F1 da referência majoritária: {report['majority_baseline_test']['classification']['macro avg']['f1-score']:.4f}.\n\nAverage precision: {report['test']['average_precision']:.4f}.\n\nFração encaminhada à revisão: {report['test']['review_fraction']:.2%}.\n\nAs notas são um proxy de avaliação negativa. Não é uma medição de sentimento anotada por humanos nem evidência de ganho de negócio.\n"
    (reports/'RESULTADOS.md').write_text(summary,encoding='utf-8')
    print(summary)

if __name__=='__main__': main()
