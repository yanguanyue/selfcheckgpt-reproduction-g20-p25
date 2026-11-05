"""Recompute all tables, cluster-bootstrap intervals, cases and budget figures."""
from pathlib import Path
from collections import defaultdict, Counter
import json,csv,sys,time,importlib.metadata as md
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,auc
from scipy.stats import pearsonr,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def metrics(rs,key):
    y=np.array([r['y_nonfactual'] for r in rs]); s=np.array([r[key] for r in rs])
    p,rec,_=precision_recall_curve(y,s)
    docs=defaultdict(list)
    for r in rs: docs[r['doc_id']].append(r)
    human=[np.mean([{'accurate':0,'minor_inaccurate':0.5,'major_inaccurate':1}[r['annotation']] for r in v]) for v in docs.values()]
    # Upstream uses mean sentence maxima for max, token-weighted mean for avg.
    predicted=[np.average([r[key] for r in v],weights=([r['token_count'] for r in v] if key=='avg_nll' else None)) for v in docs.values()]
    return {'ap':float(average_precision_score(y,s)),'auroc':float(roc_auc_score(y,s)),
        'pr_auc_trapezoid':float(auc(rec,p)),'factual_ap':float(average_precision_score(1-y,-s)),
        'pearson':float(pearsonr(human,predicted).statistic),'spearman':float(spearmanr(human,predicted).statistic)}
def main():
    started=time.perf_counter()
    rows=[json.loads(x) for x in (OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    cfg=json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
    groups=defaultdict(list)
    for r in rows: groups[(r['seed'],r['k'])].append(r)
    runs=[]
    for (seed,k),rs in groups.items():
        for key in ['avg_nll','max_nll']:runs.append({'seed':seed,'k':k,'method':key,**metrics(rs,key)})
    summary=[]
    for k in [1,3,5,20]:
        for key in ['avg_nll','max_nll']:
            found=[r for r in runs if r['k']==k and r['method']==key]
            item={'k':k,'method':key,'repeats':len(found)}
            for m in ['ap','auroc','pr_auc_trapezoid','factual_ap','pearson','spearman']:
                vals=[r[m] for r in found]; item[m+'_mean']=float(np.mean(vals)); item[m+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
            summary.append(item)
    save('metrics_by_run.json',runs);save('summary.json',summary)
    for name,values in [('metrics_by_run.csv',runs),('summary.csv',summary)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    rs=groups[(42,5)]; y=np.array([r['y_nonfactual'] for r in rs]); avg=np.array([r['avg_nll'] for r in rs]); mx=np.array([r['max_nll'] for r in rs])
    doc_ids=sorted({r['doc_id'] for r in rs}); lookup={v:i for i,v in enumerate(doc_ids)}; doc_index=np.array([lookup[r['doc_id']] for r in rs])
    rng=np.random.default_rng(cfg['bootstrap_seed']); boot=defaultdict(list)
    for _ in range(cfg['bootstrap_repeats']):
        weights=np.bincount(rng.integers(0,len(doc_ids),len(doc_ids)),minlength=len(doc_ids))[doc_index]
        if len(np.unique(y[weights>0]))<2:continue
        for name,func in [('ap',average_precision_score),('auroc',roc_auc_score)]:
            a=float(func(y,avg,sample_weight=weights)); b=float(func(y,mx,sample_weight=weights))
            boot['avg_'+name].append(a);boot['max_'+name].append(b);boot['delta_'+name].append(b-a)
    ci={k:{'low':float(np.quantile(v,.025)),'high':float(np.quantile(v,.975))} for k,v in boot.items()}
    save('bootstrap_ci.json',{'unit':'passage cluster','k':5,'seed':42,'repeats':cfg['bootstrap_repeats'],'confidence':.95,'intervals':ci})
    save('random_baseline.json',{'ap':float(y.mean()),'auroc':.5,'meaning':'constant-score no-skill reference; AP equals positive prevalence'})
    data={d['wiki_bio_test_idx']:d for d in json.loads((ROOT/'data/dataset.json').read_text(encoding='utf-8'))}
    # Descriptive ranking errors, not thresholded classification or a new truth annotation.
    selected=[];used=set()
    for pool,kind,reason in [
        (sorted((r for r in rs if r['y_nonfactual']),key=lambda x:x['max_nll']),'低分非事实句','在非事实句中最大负对数词频较低，词汇重复不能排除事实错误。'),
        (sorted((r for r in rs if not r['y_nonfactual']),key=lambda x:-x['max_nll']),'高分事实句','在人工作为事实标注的句子中得分较高，低频专名或细节可能造成误报。'),
        (sorted(rs,key=lambda x:-(x['max_nll']-x['avg_nll'])),'平均与最大评分分歧','最大评分突出最稀有词，平均评分会被常见词稀释。')]:
        count=0
        for r in pool:
            if r['doc_id'] in used:continue
            d=data[r['doc_id']]
            selected.append({**r,'error_type':kind,'explanation':reason,
                'wiki_reference':d['wiki_bio_text'],'selected_sample_texts':[d['gpt3_text_samples'][i] for i in r['sample_indices']],
                'interpretation_limit':'真实标签沿用作者人工标注；错误类型指排序边界，未按测试集选择分类阈值。'})
            used.add(r['doc_id']);count+=1
            if count==4:break
    save('cases.json',selected)
    text=['# 12 个完整错误与边界案例','配置：K=5，seed=42。高分表示非事实风险较大。错误为排序边界，不是阈值分类错误。','']
    for i,c in enumerate(selected,1):
        text.extend([f"## {i}. {c['id']} {c['error_type']}",f"- 输入：{c['sentence']}",f"- 作者标签：{c['annotation']}；二值 y={c['y_nonfactual']}",f"- 基线 avg={c['avg_nll']:.6f}；目标 max={c['max_nll']:.6f}（nats）",f"- 抽样索引：{c['sample_indices']}；seed=42；K=5",f"- 解释：{c['explanation']}",f"- WikiBio 人工参照材料（不作为算法输入）：{c['wiki_reference']}"])
        text.extend([f"- 参考输出 {j+1}：{v}" for j,v in enumerate(c['selected_sample_texts'])]);text.append('')
    (OUT/'cases.md').write_text('\n'.join(text),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for key,label,color in [('avg_nll','Unigram average','#386a9b'),('max_nll','Unigram maximum','#ad5a28')]:
        ss=[next(x for x in summary if x['method']==key and x['k']==k) for k in [1,3,5]]
        for ax,m,title in [(axes[0],'ap','Non-factual AP'),(axes[1],'auroc','Non-factual AUROC')]:
            ax.errorbar([1,3,5],[s[m+'_mean']*100 for s in ss],yerr=[s[m+'_std']*100 for s in ss],label=label,color=color,marker='o',capsize=4)
            ax.set(xlabel='Reference outputs K',ylabel=title+' (%)',xticks=[1,3,5]);ax.grid(alpha=.2)
    axes[0].axhline(y.mean()*100,color='gray',ls='--',lw=1,label='No-skill AP');axes[0].legend(fontsize=8)
    fig.savefig(OUT/'budget_curve.png',dpi=200);fig.savefig(OUT/'budget_curve.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,3.5),layout='constrained')
    for score_,label in [(avg,'Unigram average'),(mx,'Unigram maximum')]:
        p,rec,_=precision_recall_curve(y,score_);ax.plot(rec,p,label=label)
    ax.axhline(y.mean(),color='gray',ls='--',label='No-skill');ax.set(xlabel='Recall',ylabel='Precision',title='K=5, seed=42, 1,908 sentences');ax.legend();fig.savefig(OUT/'pr_curve.png',dpi=200);plt.close(fig)
    save('evaluation_environment.json',{'python':sys.version,'versions':{p:md.version(p) for p in ['numpy','scikit-learn','scipy','matplotlib']},'wall_s':time.perf_counter()-started})
    print(json.dumps({'summary':summary,'ci':ci,'cases':[{k:c[k] for k in ['id','sentence','annotation','avg_nll','max_nll','error_type']} for c in selected]},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
