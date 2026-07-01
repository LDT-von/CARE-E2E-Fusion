import glob, os, torch
results_root = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real'
for subdir in sorted(os.listdir(results_root)):
    d = os.path.join(results_root, subdir)
    if not os.path.isdir(d):
        continue
    best = sorted(glob.glob(os.path.join(d, 'fold_*_best.pt')))
    last = sorted(glob.glob(os.path.join(d, 'fold_*_last.pt')))
    print(f'{subdir}: {len(best)} best, {len(last)} last')
    if best:
        for b in best:
            ckpt = torch.load(b, map_location='cpu')
            ep = ckpt.get('epoch', '?')
            score = ckpt.get('best_score', '?')
            print(f'  {os.path.basename(b)}: epoch={ep} score={score}')
    if last and len(best) < 5:
        for l in last[len(best):]:
            ckpt = torch.load(l, map_location='cpu')
            ep = ckpt.get('epoch', '?')
            score = ckpt.get('best_score', '?')
            print(f'  {os.path.basename(l)}: epoch={ep} score={score}')
