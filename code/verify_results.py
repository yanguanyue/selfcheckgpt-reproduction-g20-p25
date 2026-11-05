from pathlib import Path
import json, hashlib
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=ROOT/'results'; rs=[json.loads(x) for x in (p/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rs)==19080
    assert len({r['id'] for r in rs})==1908
    assert all(0<=r['avg_nll']<=r['max_nll'] for r in rs)
    assert len({(r['id'],r['seed'],r['k']) for r in rs})==len(rs)
    sels=json.loads((p/'sample_selections.json').read_text())
    ss={(r['doc_id'],r['seed'],r['k']):r['sample_indices'] for r in sels}
    for (doc,seed,k),idx in ss.items():
        assert len(set(idx))==k
        if k==1: assert idx==ss[(doc,seed,3)][:1]
        if k==3: assert idx==ss[(doc,seed,5)][:3]
    for r in rs:
        assert r['y_nonfactual']==int(r['annotation']!='accurate')
        assert r['sample_indices']==ss[(r['doc_id'],r['seed'],r['k'])]
    cases=json.loads((p/'cases.json').read_text(encoding='utf-8'))
    assert len(cases)>=10 and len({c['id'] for c in cases})==len(cases)
    assert all(len(c['selected_sample_texts'])==5 and c['wiki_reference'] for c in cases)
    assert json.loads((p/'upstream_verification.json').read_text())['status']=='passed'
    checks={'status':'passed','records':len(rs),'unique_sentences':1908,'nested_selections':'passed',
        'label_direction':'passed','case_evidence':'passed','official_implementation_parity':'passed'}
    (p/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks))
if __name__=='__main__':main()
