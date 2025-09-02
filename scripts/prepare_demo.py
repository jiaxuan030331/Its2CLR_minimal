import os
from pathlib import Path
from PIL import Image

def ensure_dirs(root: Path):
    for p in [root/'single/training/tumor_real_016', root/'single/training/normal_real_016', root/'single/testing/test_real_016', root.parent/'labels/annotation']:
        p.mkdir(parents=True, exist_ok=True)


def gen_from_wsi(root: Path, wsi_path: Path, count=256, patch=224, stride=1024):
    Image.MAX_IMAGE_PIXELS=None
    img=Image.open(wsi_path.as_posix())
    tr_pos=root/'single/training/tumor_real_016'
    tr_neg=root/'single/training/normal_real_016'
    te_dir=root/'single/testing/test_real_016'
    for d in [tr_pos,tr_neg,te_dir]:
        for f in d.glob('*.jpeg'):
            f.unlink()
    written=0
    for y in range(0, 1024*16, stride):
        for x in range(0, 1024*16, stride):
            if written>=count:
                return written
            tile=img.crop((x,y,x+patch,y+patch)).convert('RGB')
            name=f"{y//stride}_{x//stride}.jpeg"
            for out in (tr_pos, tr_neg, te_dir):
                tile.save((out/name).as_posix(), 'JPEG', quality=85)
            written+=1
    return written


def gen_synthetic(root: Path, count=256, patch=224):
    import random
    tr_pos=root/'single/training/tumor_real_016'
    tr_neg=root/'single/training/normal_real_016'
    te_dir=root/'single/testing/test_real_016'
    for d in [tr_pos,tr_neg,te_dir]:
        for f in d.glob('*.jpeg'):
            f.unlink()
    for i in range(count):
        name=f"0_{i}.jpeg"
        # tumor-ish: reddish
        img = Image.new('RGB', (patch, patch), (240, random.randint(0,40), random.randint(0,40)))
        img.save((tr_pos/name).as_posix(), 'JPEG', quality=85)
        # normal-ish: bluish
        img2 = Image.new('RGB', (patch, patch), (random.randint(0,40), random.randint(0,40), 240))
        img2.save((tr_neg/name).as_posix(), 'JPEG', quality=85)
        # test: mixed
        img3 = Image.new('RGB', (patch, patch), (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        img3.save((te_dir/name).as_posix(), 'JPEG', quality=85)
    return count


def write_bag_lists(src_dir: Path):
    ds_dir = src_dir/'dataset'
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir/'train_bags.txt').write_text('/training/tumor_real_016/\n/training/normal_real_016/\n')
    (ds_dir/'val_bags.txt').write_text('/training/tumor_real_016/\n/training/normal_real_016/\n')
    (ds_dir/'test_bags.txt').write_text('/testing/test_real_016/\n')


def write_labels(label_root: Path, data_root: Path):
    import pickle
    trp = data_root/'single/training/tumor_real_016'
    trn = data_root/'single/training/normal_real_016'
    te  = data_root/'single/testing/test_real_016'
    train_gt = {}
    pseudo = {}
    for d in [trp, trn]:
        key=f"/training/{d.name}"
        train_gt.setdefault(key, {})
        pseudo.setdefault(key, {})
        is_tumor = 'tumor' in d.name
        for fn in os.listdir(d):
            if fn.endswith('.jpeg'):
                base=fn[:-5]
                train_gt[key][base] = 1 if is_tumor else 0
                pseudo[key][base]   = 0.9 if is_tumor else 0.1
    (label_root/'annotation').mkdir(parents=True, exist_ok=True)
    with open(label_root/'annotation/gt_ins_labels_train.p','wb') as f:
        pickle.dump(train_gt, f)
    with open(label_root/'ins_pseudo_label_train.p','wb') as f:
        pickle.dump(pseudo, f)
    te_gt = {f"/testing/{te.name}": {fn[:-5]:0 for fn in os.listdir(te) if fn.endswith('.jpeg')}}
    # ensure at least one positive in test
    k = next(iter(te_gt.keys()))
    if te_gt[k]:
        first = next(iter(te_gt[k].keys()))
        te_gt[k][first]=1
    with open(label_root/'annotation/gt_ins_labels_test.p','wb') as f:
        pickle.dump(te_gt, f)


def main():
    repo = Path(__file__).resolve().parents[1]
    data_root = repo/'data'
    ensure_dirs(data_root)
    wsi_path = Path('/workspace/dsmil-wsi/test-c16/input/test_016.tif')
    if wsi_path.exists():
        n = gen_from_wsi(data_root, wsi_path)
        print('generated from WSI:', n)
    else:
        n = gen_synthetic(data_root)
        print('generated synthetic:', n)
    write_bag_lists(repo/'src/train')
    write_labels(repo/'data/labels', data_root)
    print('done')

if __name__ == '__main__':
    main()
