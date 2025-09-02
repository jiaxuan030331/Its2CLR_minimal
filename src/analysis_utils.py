import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

def load_pseudo_dir(base: Path, phase: str='epoch4'):
    p = base / phase / 'ins_pseudo_label_train.p'
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}')
    return pickle.load(open(p,'rb'))

def summarize_pseudo(d: dict):
    tumor=[]; normal=[]; y=[]; s=[]
    for bag, m in d.items():
        for k,v in m.items():
            if 'tumor' in bag:
                tumor.append(v); y.append(1); s.append(v)
            else:
                normal.append(v); y.append(0); s.append(v)
    tumor=np.array(tumor); normal=np.array(normal)
    stats={
        'n_total': int(len(s)),
        'n_tumor': int(len(tumor)),
        'n_normal': int(len(normal)),
        'mean_tumor': float(np.mean(tumor)) if len(tumor)>0 else np.nan,
        'mean_normal': float(np.mean(normal)) if len(normal)>0 else np.nan,
        'p>0.5_tumor': float(np.mean(tumor>0.5)) if len(tumor)>0 else np.nan,
        'p>0.5_normal': float(np.mean(normal>0.5)) if len(normal)>0 else np.nan,
    }
    try:
        stats['instance_auc']=roc_auc_score(y,s)
    except Exception:
        stats['instance_auc']=np.nan
    return stats

__all__=['load_pseudo_dir','summarize_pseudo']
