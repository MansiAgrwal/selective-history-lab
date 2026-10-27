"""Descriptive checks of Bahg, Sloutsky & Turner (2025), OSF n2mxu.

Uses the released processed aggregate, not a reproduction of raw exclusions.
Confidence is ordinal; no probability calibration is inferred from its scale.
"""
import argparse,hashlib,json,urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import rdata

SOURCE='https://osf.io/download/rzm7g/'
LABELS={'control':'Control','active':'AL','curationRS':'PR','curationOS':'PP','curationAS':'PA'}

def run(source,out):
    source=Path(source);out=Path(out);out.mkdir(exist_ok=True,parents=True)
    if not source.exists():
        source.parent.mkdir(exist_ok=True,parents=True)
        source.write_bytes(urllib.request.urlopen(SOURCE,timeout=60).read())
    objects=rdata.read_rda(source);d=objects['dat.complete'].copy()
    d.columns=[str(x) for x in d.columns]
    if len(d)!=9600 or d.subject.nunique()!=300:raise ValueError('Source aggregate changed')
    if not (d.groupby('subject',observed=True).size()==32).all():raise ValueError('Unexpected test counts')
    if not d.CONF.between(1,11).all():raise ValueError('Unexpected confidence coding')
    if not (d.ACCnumeric==(d.ACC.astype(str)=='correct').astype(int)).all():raise ValueError('Accuracy mismatch')
    if not (d.PRSNT.astype(str)==np.where(d.RPRSNT>3,'presented','notpresented')).all():
        raise ValueError('Exposure label no longer matches threshold')
    # Author processing code adds one to the original 0..10 rating.
    d['confidence_rating']=d.CONF-1
    d['condition']=d.COND.astype(str).map(LABELS)
    d['exposure_group']=np.where(d.RPRSNT==0,'zero',np.where(d.RPRSNT<4,'one_to_three','four_or_more'))
    p=d.groupby(['STIM','condition','subject'],observed=True).agg(
        accuracy=('ACCnumeric','mean'),confidence=('confidence_rating','mean'),
        sampled_dimensions=('MEAN_N_ENCODED','first'),exposure_entropy=('ENTROPY','first')).reset_index()
    error=d[d.ACCnumeric==0].groupby('subject',observed=True).confidence_rating.mean()
    p['confidence_on_errors']=p.subject.map(error)
    s=p.groupby(['STIM','condition'],observed=True).agg(participants=('subject','nunique'),
        accuracy=('accuracy','mean'),mean_confidence=('confidence','mean'),
        mean_confidence_on_errors=('confidence_on_errors','mean'),
        participants_with_errors=('confidence_on_errors','count'),
        mean_sampled_dimensions=('sampled_dimensions','mean'),
        mean_exposure_entropy=('exposure_entropy','mean')).reset_index()
    s.to_csv(out/'condition_summary.csv',index=False)
    # Retain participant clustering in descriptive exposure summaries.
    e=d.groupby(['STIM','condition','subject','exposure_group'],observed=True).agg(
        trials=('ACCnumeric','size'),accuracy=('ACCnumeric','mean'),
        confidence=('confidence_rating','mean')).reset_index()
    es=e.groupby(['STIM','condition','exposure_group'],observed=True).agg(
        participants=('subject','nunique'),trials=('trials','sum'),accuracy=('accuracy','mean'),
        confidence=('confidence','mean')).reset_index()
    es.to_csv(out/'exposure_summary.csv',index=False)
    audit={'source':SOURCE,'project':'https://osf.io/n2mxu/','retrieved':'2026-10-25',
        'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_bytes':source.stat().st_size,
        'participants_in_processed_aggregate':300,'test_rows':9600,'cells':15,'participants_per_cell':20,
        'test_rows_per_person':32,'original_confidence_scale':'0 to 10; ordinal, not assumed a probability',
        'released_CONF_coding':'1 to 11; original processing script adds one',
        'underexposure_label':'notpresented in the source means fewer than four training presentations, not necessarily zero',
        'source_exclusion_pipeline_reproduced':False,'new_causal_effect_estimated':False,
        'raw_source_redistributed':False,'license_note':'No project license recorded in inspected OSF metadata; downloader provided instead of raw-file redistribution'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2));return s,es,audit

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/cache_bahg/dat_all.Rdata');ap.add_argument('--out',default='outputs/public_personalization')
    a=ap.parse_args();run(a.data,a.out)
