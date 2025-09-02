import json
from pathlib import Path
from src.paths import A_DIR, B_DIR
from src.analysis_utils import load_pseudo_dir, summarize_pseudo

def summarize_one(base: Path):
    out={}
    for phase in ['epoch_init','epoch4']:
        try:
            d=load_pseudo_dir(base, phase)
            out[phase]=summarize_pseudo(d)
        except FileNotFoundError:
            pass
    return out

def main():
    res={'A-256': summarize_one(A_DIR), 'B-256': summarize_one(B_DIR)}
    print(json.dumps(res, indent=2))

if __name__=='__main__':
    main()
