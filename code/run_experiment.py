"""Faithful CPU-only SelfCheck-Unigram avg/max reproduction; no LLM API calls.

The original passage is included exactly as in upstream SelfCheckNgram.predict.
spaCy sentence segmentation followed by re-tokenization preserves upstream add().
All 1/3/5 subsets are nested within each per-document seeded permutation.
"""
from pathlib import Path
from collections import Counter
import argparse, json, hashlib, time, sys, platform, csv, importlib.metadata as md
import numpy as np
import spacy

ROOT = Path(__file__).resolve().parents[1]
def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def tokenize_counts(nlp, doc):
    # Upstream strips sentence text and calls nlp(sentence) again before counting.
    return Counter(tok.text.lower() for sent in doc.sents for tok in nlp.make_doc(sent.text.strip()))

def score(counts, tokens):
    total = sum(counts.values())
    if not tokens or total == 0: raise ValueError('Empty sentence/counts')
    missing = set(tokens) - counts.keys()
    if missing: raise ValueError(f'Unseen tokens: {missing}; verify original-passage inclusion')
    values = -np.log(np.array([counts[t] / total for t in tokens]))
    return float(values.mean()), float(values.max())

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config', default=str(ROOT/'configs/experiment.json'))
    parser.add_argument('--limit',type=int); parser.add_argument('--output',default='results')
    parser.add_argument('--verify-upstream',action='store_true'); args=parser.parse_args()
    start=time.perf_counter(); cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    out=ROOT/args.output; out.mkdir(parents=True,exist_ok=True)
    raw=(ROOT/'data/dataset.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==cfg['dataset_sha256'], 'Dataset checksum mismatch'
    data=json.loads(raw); data=data[:args.limit] if args.limit else data
    assert len({d['wiki_bio_test_idx'] for d in data})==len(data)
    for d in data:
        assert len(d['gpt3_sentences'])==len(d['annotation'])
        assert len(d['gpt3_text_samples'])==20
        assert set(d['annotation']) <= {'accurate','minor_inaccurate','major_inaccurate'}
    log=[]
    def emit(event,**kw):
        record={'event':event,'elapsed_s':round(time.perf_counter()-start,4),**kw}
        log.append(record); print(json.dumps(record,ensure_ascii=False),flush=True)
    emit('start',passages=len(data),config=cfg)
    nlp=spacy.load(cfg['tokenizer_model']); spacy.require_cpu()
    counts=[]; flat=[]
    for d in data: flat.extend([d['gpt3_text']]+d['gpt3_text_samples'])
    for i,doc in enumerate(nlp.pipe(flat,batch_size=32)):
        counts.append(tokenize_counts(nlp,doc))
        if (i+1)%420==0: emit('tokenization',processed=i+1,total=len(flat))
    tokenization_s=time.perf_counter()-start
    sent_tokens=[[ [t.text.lower() for t in nlp.make_doc(s)] for s in d['gpt3_sentences']] for d in data]
    rows=[]; selections=[]; timing=[]
    for seed in cfg['seeds']:
        for k in cfg['reference_budgets']+([20] if seed==cfg['anchor_seed'] else []):
            tick=time.perf_counter()
            for di,d in enumerate(data):
                ident=d['wiki_bio_test_idx']
                rng=np.random.default_rng(np.random.SeedSequence([seed,ident]))
                ids=rng.permutation(20)[:k].tolist()
                selections.append({'doc_id':ident,'seed':seed,'k':k,'sample_indices':ids})
                cnt=counts[21*di].copy()
                for ix in ids: cnt.update(counts[21*di+1+ix])
                for si,(text,label,toks) in enumerate(zip(d['gpt3_sentences'],d['annotation'],sent_tokens[di])):
                    avg,mx=score(cnt,toks)
                    rows.append({'doc_id':ident,'sentence_id':si,'id':f'{ident}:{si}', 'seed':seed,'k':k,
                        'sentence':text,'annotation':label,'y_nonfactual':int(label!='accurate'),
                        'avg_nll':avg,'max_nll':mx,'token_count':len(toks),'sample_indices':ids,
                        'config':'configs/experiment.json'})
            elapsed=time.perf_counter()-tick
            timing.append({'seed':seed,'k':k,'cached_scoring_s':elapsed}); emit('scoring',seed=seed,k=k,seconds=elapsed)
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    save(out/'sample_selections.json',selections)
    save(out/'dataset_manifest.json',{'document_ids':[d['wiki_bio_test_idx'] for d in data],
        'passages':len(data),'sentences':sum(len(d['annotation']) for d in data),
        'labels':dict(Counter(x for d in data for x in d['annotation'])),
        'selection':'all public evaluation passages, original dataset order; no train/dev tuning'})
    if args.verify_upstream:
        import importlib.util
        spec=importlib.util.spec_from_file_location('official',ROOT/'code/upstream/modeling_ngram.py')
        official=importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
        checks=[]
        for d in data[:3]:
            model=official.UnigramModel(lowercase=True)
            sel=next(x for x in selections if x['doc_id']==d['wiki_bio_test_idx'] and x['seed']==42 and x['k']==5)
            model.add(d['gpt3_text'])
            for ix in sel['sample_indices']:model.add(d['gpt3_text_samples'][ix])
            model.train(k=0); ref=model.evaluate(d['gpt3_sentences'])['sent_level']
            ours=[r for r in rows if r['doc_id']==d['wiki_bio_test_idx'] and r['seed']==42 and r['k']==5]
            for key,upkey in [('avg_nll','avg_neg_logprob'),('max_nll','max_neg_logprob')]:
                delta=float(np.max(np.abs(np.array([r[key] for r in ours])-ref[upkey])))
                assert delta<1e-12, (key,delta)
                checks.append({'doc_id':d['wiki_bio_test_idx'],'score':key,'max_absolute_difference':delta})
        save(out/'upstream_verification.json',{'status':'passed','checks':checks,'tolerance':1e-12})
        emit('upstream_verification',status='passed',checks=len(checks))
    env={'python':sys.version,'platform':platform.platform(),'cpu':platform.processor(),
        'versions':{p:md.version(p) for p in ['spacy','numpy','en-core-web-sm']},
        'tokenization_and_startup_s':tokenization_s,'scoring':timing,'wall_s':time.perf_counter()-start,
        'gpu_required':False,'paid_api_calls':0}
    save(out/'environment.json',env); emit('complete',records=len(rows),wall_s=env['wall_s'])
    (out/'run_log.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in log),encoding='utf-8')
if __name__=='__main__':main()
