"""CARE-style SEM denoiser training from paired fast/slow scans.

Dataset layout (local only, never in git):

  D:\\SEM training\\datasets\\<set name>\\NN a.tif / NN b.tif  (+ JEOL .txt sidecars)
      a = fast scan (noisy)  -> training input
      b = slow scan (clean)  -> training target

Training recipe (validated on the plastic pair set):
  1. register  - rigid-align b to a (smoothed phase correlation + integer
     brute-force refine), then normalize b into a's intensity scale with a
     per-pair TLS gain/offset fit on smoothed central crops. The normalized
     residual is pure noise, so the target adds no structure the input lacks.
  2. train     - compact U-Net on random 256x256 patches (flips for
     augmentation), MSE + 0.1 * (1 - SSIM), Adam 2e-4, GPU.
  3. eval      - every 4th pair held out: PSNR/SSIM of raw vs denoised vs
     BM3D against the target, comparison sheet PNG.

Subcommands:
  register --data PATH [--work DIR]
  train    --work DIR --out MODEL.pt [--epochs 60] [--batches 40] [--batch-size 16]
  eval     --work DIR --model MODEL.pt
  run      --data PATH --out MODEL.pt [--work DIR]   (all three)

Default work dir: D:\\SEM training\\train_work. Default model dir: models\\
next to this script (paired with denoise_tool.py's manifest).
"""
import argparse
import json
import os
import random
import re
import sys

DEFAULT_WORK = r'D:\SEM training\train_work'
PS = 256          # training patch size
BAR_H = 128       # JEOL data-bar rows appended below the image content
JEOL_CONTENT_STEP = 960  # JEOL content heights are multiples of 960 (960/1920/3840)
PAIR_A_RE = re.compile(r'^(.*?)[\s_-]?a\.tif$', re.IGNORECASE)


# ---------------------------------------------------------------- data utils

def sidecar_image_size(path):
    """(width, height) content size from the matching JEOL .txt sidecar, or None."""
    p = os.path.splitext(path)[0] + '.txt'
    if not os.path.isfile(p):
        return None
    try:
        with open(p, errors='ignore') as fh:
            m = re.search(r'\$CM_IMAGE_SIZE\s+(\d+)\s+(\d+)', fh.read())
    except OSError:
        return None
    return (int(m.group(1)), int(m.group(2))) if m else None


def bar_rows(img, path):
    """JEOL data-bar rows at the bottom of the image (0 if none).

    Prefers the sidecar's $CM_IMAGE_SIZE content size; without a matching
    sidecar, falls back to the JEOL export heights (content = multiple of
    960 plus a 128-row bar).
    """
    h, w = img.shape
    size = sidecar_image_size(path)
    if size and size[0] == w and 0 < size[1] < h:
        return h - size[1]
    if h > BAR_H and (h - BAR_H) % JEOL_CONTENT_STEP == 0:
        return BAR_H
    return 0


def load_sem_tif(path, crop_bar=True):
    """JEOL SEM export -> float32 grayscale; data bar cropped unless told not to."""
    import numpy as np
    import tifffile
    arr = tifffile.imread(path)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        img = arr[:, :, :3].mean(2)
    elif arr.ndim == 3:
        img = arr[:, :, 0]
    else:
        img = arr
    if crop_bar:
        nbar = bar_rows(img, path)
        if nbar:
            img = img[:img.shape[0] - nbar]
    return img.astype('float32')


def list_datasets(data_path):
    """A single dataset folder, or a parent whose subfolders are datasets."""
    if any(os.path.isfile(os.path.join(data_path, f)) for f in os.listdir(data_path)
           if PAIR_A_RE.match(f)):
        return [data_path]
    return sorted(os.path.join(data_path, d) for d in os.listdir(data_path)
                  if os.path.isdir(os.path.join(data_path, d)))


def find_pairs(folder):
    """{key: (a_path, b_path)} for 'NN a.tif'/'NN b.tif' pairs, plus tile-name
    pairs across montager subfolders (e.g. standard aquisition / max aquisition)."""
    pairs = {}
    for name in os.listdir(folder):
        m = PAIR_A_RE.match(name)
        if not m:
            continue
        stem = m.group(1)
        for cand in (stem + 'b.tif', stem + ' b.tif', stem + '-b.tif', stem + '_b.tif'):
            bpath = os.path.join(folder, cand)
            if os.path.isfile(bpath):
                pairs[stem.strip() or name] = (os.path.join(folder, name), bpath)
                break
    subs = []
    for d in sorted(os.listdir(folder)):
        sub = os.path.join(folder, d)
        if os.path.isdir(sub) and any(f.lower().endswith(('.tif', '.tiff'))
                                      for f in os.listdir(sub)):
            subs.append(sub)
    if len(subs) >= 2:
        def scan_time(sub):
            for f in os.listdir(sub):
                if f.lower().endswith('.txt'):
                    try:
                        with open(os.path.join(sub, f), encoding='utf-8',
                                  errors='replace') as fh:
                            m = re.search(r'\$SM_SCAN_TIME\s+(\d+)', fh.read())
                    except OSError:
                        continue
                    if m:
                        return int(m.group(1))
            return 0
        subs.sort(key=scan_time)
        a_dir, b_dir = subs[0], subs[-1]
        for f in os.listdir(a_dir):
            if f.lower().endswith(('.tif', '.tiff')):
                bpath = os.path.join(b_dir, f)
                if os.path.isfile(bpath):
                    pairs[os.path.splitext(f)[0]] = (os.path.join(a_dir, f), bpath)
    return pairs


def sniff_condition(folder):
    """Best-effort detector/settings sniff from JEOL .txt sidecars."""
    import re as _re
    text = ''
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith('.txt'):
                try:
                    with open(os.path.join(root, name), encoding='utf-8',
                              errors='replace') as fh:
                        text = fh.read()
                    break
                except OSError:
                    continue
        if text:
            break
    cond = {}
    for m in _re.finditer(r'^\$?([A-Za-z0-9_.\-]+)(?:[=:][ \t]*|[ \t]+)([^\r\n]+)$',
                          text, _re.M):
        k, v = m.group(1).upper(), m.group(2).strip()
        if not v:
            continue
        if any(t in k for t in ('INSTRUMENT', 'DETECTOR', 'MAG', 'ACCEL_VOLT',
                                'PIXEL_SIZE', 'WD', 'VACUUM_MODE')):
            cond[k] = v
    return cond


def sigma_mad(img):
    """Noise sigma via MAD of the Laplacian (robust, structure-free)."""
    import numpy as np
    from scipy.ndimage import laplace
    return float(np.median(np.abs(laplace(img))) * 1.4826 / (20 ** 0.5))


def register_pair(a, b):
    """Rigid-align a onto b's frame: smoothed phase corr + integer refine."""
    import numpy as np
    from scipy.ndimage import gaussian_filter, shift as ndi_shift
    from skimage.registration import phase_cross_correlation
    as_ = gaussian_filter(a, 2)
    bs_ = gaussian_filter(b, 2)
    scale = 4
    (shift, _, _) = phase_cross_correlation(as_[::scale, ::scale],
                                            bs_[::scale, ::scale],
                                            upsample_factor=10)
    dy0 = int(round(shift[0] * scale))
    dx0 = int(round(shift[1] * scale))
    h, w = a.shape
    R = 256
    cy, cx = h // 2, w // 2
    ys, xs = slice(cy - R, cy + R), slice(cx - R, cx + R)
    bc = bs_[ys, xs]
    best = (-2.0, dy0, dx0)
    for dy in range(dy0 - 3, dy0 + 4):
        for dx in range(dx0 - 3, dx0 + 4):
            ac = as_[ys.start - dy:ys.stop - dy, xs.start - dx:xs.stop - dx]
            if ac.shape != bc.shape:
                continue
            c = _ncc(ac, bc)
            if c > best[0]:
                best = (c, dy, dx)
    dy, dx = int(best[1]), int(best[2])
    aligned = ndi_shift(a, (dy, dx), order=1)
    return aligned, (dy, dx, float(_ncc(gaussian_filter(aligned, 2), bs_)))


def _ncc(x, y):
    import numpy as np
    x = x.ravel() - x.mean()
    y = y.ravel() - y.mean()
    d = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / d) if d > 0 else 0.0


def tls_gain_offset(a, b):
    """TLS line b ~ g*a + o on smoothed central crops (OLS is biased here)."""
    import numpy as np
    from scipy.ndimage import gaussian_filter
    h, w = a.shape
    ys = slice(h // 4, 3 * h // 4)
    xs = slice(w // 4, 3 * w // 4)
    as_ = gaussian_filter(a[ys, xs], 2).ravel()
    bs_ = gaussian_filter(b[ys, xs], 2).ravel()
    if as_.size > 400000:
        idx = np.random.default_rng(0).choice(as_.size, 400000, replace=False)
        as_, bs_ = as_[idx], bs_[idx]
    ax = as_ - as_.mean()
    by = bs_ - bs_.mean()
    m = np.array([[float(np.mean(ax * ax)), float(np.mean(ax * by))],
                  [float(np.mean(ax * by)), float(np.mean(by * by))]])
    w, vecs = np.linalg.eigh(m)
    vec = vecs[:, int(np.argmax(w))]  # principal direction -> TLS slope
    if abs(vec[0]) > 1e-12:
        g = float(vec[1] / vec[0])
    else:
        g = float(np.sum(ax * by) / np.sum(ax * ax))
    if g <= 0 or abs(g) > 4:  # degenerate fit -> fall back to OLS
        g = float(np.sum(ax * by) / np.sum(ax * ax))
    o = float(bs_.mean() - g * as_.mean())
    return g, o


# ---------------------------------------------------------------- model

def build_unet(base=24):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(min(8, o), o), nn.SiLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(min(8, o), o), nn.SiLU())

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            b, b2, b4, b8 = base, base * 2, base * 4, base * 8
            self.e1 = Block(1, b)
            self.e2 = Block(b, b2)
            self.e3 = Block(b2, b4)
            self.e4 = Block(b4, b8)
            self.pool = nn.MaxPool2d(2)
            self.d3 = Block(b8 + b4, b4)
            self.d2 = Block(b4 + b2, b2)
            self.d1 = Block(b2 + b, b)
            self.out = nn.Conv2d(b, 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            e4 = self.e4(self.pool(e3))
            u3 = F.interpolate(e4, size=e3.shape[-2:], mode='bilinear', align_corners=False)
            d3 = self.d3(torch.cat([u3, e3], 1))
            u2 = F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False)
            d2 = self.d2(torch.cat([u2, e2], 1))
            u1 = F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
            d1 = self.d1(torch.cat([u1, e1], 1))
            return self.out(d1)

    return UNet()


def ssim_fn(pred, target, window, data_range=1.0):
    import torch.nn.functional as F
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    pad = window.shape[-1] // 2
    mu1 = F.conv2d(pred, window, padding=pad)
    mu2 = F.conv2d(target, window, padding=pad)
    m1sq, m2sq, m12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(pred * pred, window, padding=pad) - m1sq
    s2 = F.conv2d(target * target, window, padding=pad) - m2sq
    s12 = F.conv2d(pred * target, window, padding=pad) - m12
    v = ((2 * m12 + c1) * (2 * s12 + c2)) / ((m1sq + m2sq + c1) * (s1 + s2 + c2))
    return v.mean()


def tiled_predict(model, img, device, tile=512, overlap=64):
    import numpy as np
    import torch
    h, w = img.shape
    py, px = (8 - h % 8) % 8, (8 - w % 8) % 8
    padded = np.pad(img, ((0, py), (0, px)), mode='edge')
    hp, wp = padded.shape
    stride = tile - overlap
    out = np.zeros((hp, wp), 'float32')
    cnt = np.zeros((hp, wp), 'float32')
    ys = list(range(0, max(hp - tile, 0) + 1, stride))
    xs = list(range(0, max(wp - tile, 0) + 1, stride))
    if ys[-1] != max(hp - tile, 0):
        ys.append(max(hp - tile, 0))
    if xs[-1] != max(wp - tile, 0):
        xs.append(max(wp - tile, 0))
    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = padded[y:y + tile, x:x + tile]
                t = torch.from_numpy(patch[None, None]).to(device)
                p = model(t)[0, 0].cpu().numpy()
                out[y:y + tile, x:x + tile] += p
                cnt[y:y + tile, x:x + tile] += 1
    return (out / np.maximum(cnt, 1))[:h, :w]


# ---------------------------------------------------------------- subcommands

def cmd_register(args):
    import numpy as np
    datasets = list_datasets(args.data)
    all_pairs = []
    conditions = {}
    prev = {}
    mpath = os.path.join(args.work, 'manifest.json')
    if os.path.isfile(mpath) and not getattr(args, 'force', False):
        try:
            with open(mpath, encoding='utf-8') as fh:
                prev = {(p['dataset'], p['key']): p
                        for p in json.load(fh).get('pairs', [])}
        except (OSError, ValueError):
            prev = {}
    for ds in datasets:
        pairs = find_pairs(ds)
        if not pairs:
            print('  (no a/b pairs in %s - skipping)' % ds)
            continue
        name = os.path.basename(os.path.normpath(ds))
        conditions[name] = sniff_condition(ds)
        print('%s: %d pairs' % (name, len(pairs)))
        for key in sorted(pairs):
            a_path, b_path = pairs[key]
            slug = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
            pslug = re.sub(r'[^A-Za-z0-9]+', '_', key).strip('_')
            npz = os.path.join(args.work, '%s__%s.npz' % (slug, pslug))
            cached = prev.get((name, key))
            if cached and cached.get('npz') == npz and os.path.isfile(npz):
                all_pairs.append(cached)
                print('  %-6s cached' % key)
                continue
            a = load_sem_tif(a_path) / 255.0
            b = load_sem_tif(b_path) / 255.0
            aligned, (dy, dx, ncc) = register_pair(a, b)
            g, o = tls_gain_offset(a, b)
            target = (b - o) / g
            sa, sb = sigma_mad(a), sigma_mad(b)
            np.savez_compressed(npz, aligned=aligned.astype('float32'),
                                target=target.astype('float32'))
            all_pairs.append({'dataset': name, 'key': key, 'npz': npz,
                              'shift': [dy, dx], 'ncc': round(ncc, 4),
                              'gain': round(g, 4), 'offset': round(o, 2),
                              'sigma_a': round(sa, 2), 'sigma_b': round(sb, 2),
                              'ratio': round(sa / sb, 3) if sb else None})
            print('  %-6s shift (%+d,%+d) ncc %.3f  gain %.3f off %.1f  '
                  'sigma %.1f/%.1f (x%.2f)' % (key, dy, dx, ncc, g, o, sa, sb,
                                               sa / sb if sb else 0))
    if not all_pairs:
        print('No pairs found under %s' % args.data)
        return 1
    meta = {'pairs': all_pairs, 'datasets': conditions,
            'registered': __import__('datetime').date.today().isoformat()}
    with open(os.path.join(args.work, 'manifest.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    print('%d pairs registered -> %s' % (len(all_pairs), args.work))
    return 0


def _load_work(workdir):
    with open(os.path.join(workdir, 'manifest.json')) as fh:
        meta = json.load(fh)
    return meta


def cmd_train(args):
    import numpy as np
    import torch
    meta = _load_work(args.work)
    pairs = meta['pairs']
    eval_idx = set(range(3, len(pairs), 4)) if len(pairs) >= 4 else {len(pairs) - 1}
    train_pairs = [p for i, p in enumerate(pairs) if i not in eval_idx]
    eval_pairs = [p for i, p in enumerate(pairs) if i in eval_idx]
    print('%d train / %d eval pairs' % (len(train_pairs), len(eval_pairs)))
    data = {p['npz']: np.load(p['npz']) for p in pairs}
    train = [(data[p['npz']]['aligned'], data[p['npz']]['target']) for p in train_pairs]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: %s' % device)
    model = build_unet().to(device)
    window = _ssim_window()
    window = window.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # fixed validation crop from the first eval pair
    ev = data[eval_pairs[0]['npz']]
    eh, ew = ev['aligned'].shape
    vy, vx = eh // 2 - 256, ew // 2 - 256
    val_in = ev['aligned'][vy:vy + 512, vx:vx + 512]
    val_tg = ev['target'][vy:vy + 512, vx:vx + 512]
    val_t = (torch.from_numpy(val_in[None, None]).to(device),
             torch.from_numpy(val_tg[None, None]).to(device))

    def psnr(a, b):
        mse = float(((a - b) ** 2).mean())
        return 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)

    best = (-1.0, None)
    rng = random.Random(0)
    step = 0
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for _ in range(args.batches):
            x, y = _sample_batch(train, args.batch_size)
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = ((pred - y) ** 2).mean() + 0.1 * (1 - ssim_fn(pred, y, window))
            loss.backward()
            opt.step()
            tot += float(loss)
            step += 1
        model.eval()
        with torch.no_grad():
            pv = model(val_t[0])[0, 0].cpu().numpy()
        score = psnr(pv, val_tg)
        if score > best[0]:
            import copy
            best = (score, copy.deepcopy(model.state_dict()))
        print('epoch %2d  loss %.4f  val-PSNR %.2f dB%s'
              % (epoch + 1, tot / args.batches, score,
                 '  *best' if score == best[0] else ''))
    if best[1] is not None:
        model.load_state_dict(best[1])

    ckpt = {'state_dict': model.state_dict(), 'base': 24, 'scale': 255,
            'input': 'fast scan /255', 'target': 'slow scan TLS-normalized',
            'trained': __import__('datetime').date.today().isoformat(),
            'datasets': sorted({p['dataset'] for p in pairs}),
            'pairs': len(pairs)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ckpt, args.out)
    print('saved %s (val-PSNR %.2f dB)' % (args.out, best[0]))
    return 0


def _ssim_window(k=11, sigma=1.5):
    import torch
    x = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(x * x) / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def _sample_batch(pairs, batch_size):
    import numpy as np
    import torch
    a = np.empty((batch_size, 1, PS, PS), 'float32')
    t = np.empty((batch_size, 1, PS, PS), 'float32')
    for i in range(batch_size):
        xa, ta = random.choice(pairs)
        h, w = xa.shape
        py = rng_slice(h)
        px = rng_slice(w)
        ca = xa[py:py + PS, px:px + PS]
        ct = ta[py:py + PS, px:px + PS]
        if random.random() < 0.5:
            ca, ct = ca[:, ::-1], ct[:, ::-1]
        if random.random() < 0.5:
            ca, ct = ca[::-1, :], ct[::-1, :]
        a[i, 0] = np.ascontiguousarray(ca)
        t[i, 0] = np.ascontiguousarray(ct)
    return torch.from_numpy(a), torch.from_numpy(t)


def rng_slice(size):
    return random.randint(8, max(9, size - PS - 8))


def cmd_eval(args):
    import numpy as np
    import torch
    import tifffile
    meta = _load_work(args.work)
    pairs = meta['pairs']
    eval_idx = set(range(3, len(pairs), 4)) if len(pairs) >= 4 else {len(pairs) - 1}
    eval_pairs = [p for i, p in enumerate(pairs) if i in eval_idx]
    if getattr(args, 'max_eval', 0) and len(eval_pairs) > args.max_eval > 1:
        step = (len(eval_pairs) - 1) / (args.max_eval - 1)
        eval_pairs = [eval_pairs[int(round(i * step))]
                      for i in range(args.max_eval)]
    ckpt = torch.load(args.model, map_location='cpu', weights_only=True)
    model = build_unet(ckpt.get('base', 24))
    model.load_state_dict(ckpt['state_dict'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    def psnr(a, b):
        mse = float(((a - b) ** 2).mean())
        return 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)

    try:
        from skimage.metrics import structural_similarity as ssim_sk
    except ImportError:
        ssim_sk = None

    try:
        import bm3d as bm3d_mod
        have_bm3d = True
    except ImportError:
        have_bm3d = False

    rows = []
    crops = []
    for p in eval_pairs:
        d = np.load(p['npz'])
        inp, tgt = d['aligned'], d['target']
        den = tiled_predict(model, inp, device)
        den_c = np.clip(den, 0, 1)
        bm = None
        if have_bm3d:
            try:
                bm = np.clip(bm3d_mod.bm3d(inp, sigma_psd=float(p['sigma_a'])), 0, 1)
            except Exception as e:
                print('bm3d failed: %s' % e)
        row = {'pair': '%s/%s' % (p['dataset'], p['key'])}
        row['raw'] = (round(float(psnr(inp, tgt)), 2),
                      round(float(ssim_sk(inp, tgt, data_range=1.0)), 4) if ssim_sk else None)
        row['unet'] = (round(float(psnr(den_c, tgt)), 2),
                       round(float(ssim_sk(den_c, tgt, data_range=1.0)), 4) if ssim_sk else None)
        if bm is not None:
            row['bm3d'] = (round(float(psnr(bm, tgt)), 2),
                           round(float(ssim_sk(bm, tgt, data_range=1.0)), 4) if ssim_sk else None)
        rows.append(row)
        print('%-14s raw %s  unet %s%s' % (
            row['pair'], row['raw'], row['unet'],
            '  bm3d %s' % (row['bm3d'],) if bm is not None else ''))
        h, w = inp.shape
        crops.append((inp[h // 2 - 128:h // 2 + 128, w // 4:w // 4 + 256],
                      den_c[h // 2 - 128:h // 2 + 128, w // 4:w // 4 + 256],
                      bm[h // 2 - 128:h // 2 + 128, w // 4:w // 4 + 256] if bm is not None else None,
                      tgt[h // 2 - 128:h // 2 + 128, w // 4:w // 4 + 256]))

    # comparison sheet: rows = pairs, cols = raw / unet / bm3d / target
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ncol = 4 if have_bm3d else 3
    labels = ['raw (fast scan)', 'U-Net denoised', 'BM3D', 'target (slow scan)'][:ncol]
    fig, axes = plt.subplots(len(crops), ncol, figsize=(3.0 * ncol, 3.0 * len(crops)),
                             squeeze=False)
    for r, crop in enumerate(crops):
        cells = [crop[0], crop[1], crop[2], crop[3]][:ncol]
        for c, (im, lab) in enumerate(zip(cells, labels)):
            ax = axes[r][c]
            if im is None:
                ax.axis('off')
                continue
            ax.imshow(im, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(lab, fontsize=10)
            if c == 0:
                ax.set_ylabel(rows[r]['pair'], fontsize=8)
    stem = os.path.splitext(os.path.abspath(args.model))[0]
    sheet = stem + '_eval.png'
    fig.tight_layout()
    fig.savefig(sheet, dpi=110)
    print('sheet -> %s' % sheet)
    with open(stem + '_metrics.json', 'w') as fh:
        json.dump(rows, fh, indent=2)

    # update the denoise_tool manifest next to the model
    mdir = os.path.dirname(os.path.abspath(args.model))
    mpath = os.path.join(mdir, 'manifest.json')
    mdata = {'models': []}
    if os.path.isfile(mpath):
        try:
            with open(mpath) as fh:
                mdata = json.load(fh)
        except Exception:
            pass
    conds = meta.get('datasets', {})
    cond = {}
    if conds:
        keys = set.intersection(*(set(c) for c in conds.values()))
        cond = {k: next(iter({c[k] for c in conds.values()})) for k in keys
                if len({c[k] for c in conds.values()}) == 1}
    entry = {'name': os.path.splitext(os.path.basename(args.model))[0],
             'file': os.path.basename(args.model),
             'condition': cond,
             'pairs': len(pairs),
             'trained': meta.get('registered'),
             'eval': rows}
    mdata['models'] = [m for m in mdata.get('models', [])
                       if m.get('file') != entry['file']]
    mdata['models'].append(entry)
    with open(mpath, 'w') as fh:
        json.dump(mdata, fh, indent=2)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, data=False, work=True):
        if data:
            p.add_argument('--data', required=True, help='dataset folder or datasets parent')
        if work:
            p.add_argument('--work', default=DEFAULT_WORK, help='registered-pair cache dir')

    p = sub.add_parser('register')
    common(p, data=True)
    p.add_argument('--force', action='store_true',
                   help='re-register pairs even if cached npz exists')
    p.set_defaults(fn=cmd_register)

    p = sub.add_parser('train')
    common(p)
    p.add_argument('--out', required=True, help='output model path (.pt)')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batches', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=16)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser('eval')
    common(p)
    p.add_argument('--model', required=True, help='trained model (.pt)')
    p.add_argument('--max-eval', type=int, default=8,
                   help='cap on eval pairs (BM3D is slow on full frames)')
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser('run')
    common(p, data=True)
    p.add_argument('--force', action='store_true',
                   help='re-register pairs even if cached npz exists')
    p.add_argument('--out', required=True, help='output model path (.pt)')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batches', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=16)
    p.set_defaults(fn=lambda a: cmd_register(a) or cmd_train(a) or cmd_eval(a))

    args = ap.parse_args()
    if not hasattr(args, 'fn'):
        ap.error('missing subcommand')
    if getattr(args, 'data', None) and args.cmd == 'register':
        os.makedirs(args.work, exist_ok=True)
    sys.exit(args.fn(args))


if __name__ == '__main__':
    main()
