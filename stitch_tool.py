"""Ashlar stitcher - auto-detects tile sets and stitches them, GUI or drop-in.

Tile naming: <set> X### Y###.tif  (e.g. 00001 X003 Y002.tif, 1-indexed, zero-padded)

Ways to use:
  1. Place stitch_tool.py + stitch_here.bat in a folder of tiles and
     double-click stitch_here.bat  -> stitches every set in that folder,
     writing stitched/<set>_ashlar<suffix>.ome.tif
  2. Run install_sendto.bat once, then right-click any folder ->
     Send To -> "Stitch with Ashlar"
  3. python stitch_tool.py                  -> GUI (folder picker, set list, log)
  4. python stitch_tool.py <folder> [--scan-only] [--sets 00001,00003]
        [--suffix _v7] [--overwrite] [--overlap 0.20]

Requires the python environment that has ashlar installed (on the lab PCs:
C:\\miniconda3\\python.exe - stitch_here.bat and install_sendto.bat pick it
automatically). Parameters are the validated set: overlap 0.20,
filter-sigma 15, maximum-shift 150 um, align-channel 0, pixel_size 0.5
placeholder, plus the v7 registration patch (padded FFT + +/-300 px search
limit + multi-scale pair registration), so output quality matches v7.
"""
import argparse
import contextlib
import os
import queue
import re
import sys
import threading
import time
from collections import Counter

OVERLAP = 0.20
PIXEL_SIZE = 0.5
FILTER_SIGMA = 15
MAX_SHIFT_UM = 150
ALIGN_CHANNEL = 0
DEFAULT_SUFFIX = '_v7'
CONDA_PY = r'C:\miniconda3\python.exe'
TILE_RE = re.compile(r'^([A-Za-z0-9]+) X(\d+) Y(\d+)\.tif$', re.IGNORECASE)


class TileSet:
    def __init__(self, scan):
        self.scan = scan
        self.coords = set()
        self.files = []
        self.sizes = {}

    def _range(self):
        xs = [c[0] for c in self.coords]
        ys = [c[1] for c in self.coords]
        return min(xs), max(xs), min(ys), max(ys)

    def grid(self):
        x0, x1, y0, y1 = self._range()
        return x1 - x0 + 1, y1 - y0 + 1

    def missing(self):
        x0, x1, y0, y1 = self._range()
        want = {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
        return sorted(want - self.coords)

    def outliers(self):
        if not self.sizes:
            return []
        common = Counter(self.sizes.values()).most_common(1)[0][0]
        return [f for f, s in self.sizes.items() if s != common]

    def note(self):
        parts = []
        m = self.missing()
        if m:
            parts.append('%d tiles missing (e.g. X%03d Y%03d)' % (len(m), m[0][0], m[0][1]))
        o = self.outliers()
        if o:
            parts.append('%d odd-size tiles (e.g. %s)' % (len(o), o[0]))
        return '; '.join(parts) if parts else 'complete'


def scan_folder(folder):
    try:
        from PIL import Image
    except ImportError:
        Image = None
    sets = {}
    for name in os.listdir(folder):
        m = TILE_RE.match(name)
        if not m:
            continue
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        s = sets.setdefault(m.group(1), TileSet(m.group(1)))
        s.coords.add((int(m.group(2)), int(m.group(3))))
        s.files.append(name)
        if Image is not None:
            try:
                with Image.open(path) as im:
                    s.sizes[name] = im.size
            except Exception:
                s.sizes[name] = None
    return [sets[k] for k in sorted(sets)]


def report(sets, folder):
    print('%d image set(s) in %s' % (len(sets), folder))
    for s in sets:
        gx, gy = s.grid()
        print('  %s: %d tiles, grid %dx%d, %s' % (s.scan, len(s.files), gx, gy, s.note()))


_patched = False


def ensure_patch():
    global _patched
    if _patched:
        return
    import numpy as np
    import scipy.ndimage
    import ashlar.utils

    LIMIT = 300

    def register_padded(img1, img2, sigma, upsample=10):
        img1w = ashlar.utils.window(ashlar.utils.whiten(img1, sigma)).astype(np.float32)
        img2w = ashlar.utils.window(ashlar.utils.whiten(img2, sigma)).astype(np.float32)
        h, w = img1.shape
        big = (2 * h, 2 * w)
        f1 = np.fft.rfft2(img1w, s=big)
        f2 = np.fft.rfft2(img2w, s=big)
        cc = np.fft.irfft2(f1 * np.conj(f2), s=big)
        ys = np.arange(big[0])
        ys = np.where(ys < big[0] / 2, ys, ys - big[0])
        xs = np.arange(big[1])
        xs = np.where(xs < big[1] / 2, xs, xs - big[1])
        mask = (np.abs(ys)[:, None] <= LIMIT) & (np.abs(xs)[None, :] <= LIMIT)
        cc = np.where(mask, cc, -1e30)
        peak = np.unravel_index(np.argmax(cc), cc.shape)
        r = 4
        wy = np.arange(peak[0] - r, peak[0] + r + 1) % big[0]
        wx = np.arange(peak[1] - r, peak[1] + r + 1) % big[1]
        win = cc[np.ix_(wy, wx)]
        if upsample > 1:
            z = scipy.ndimage.zoom(win, upsample, order=3)
            iy, ix = np.unravel_index(np.argmax(z), z.shape)
            ky = (peak[0] + iy / upsample - r) % big[0]
            kx = (peak[1] + ix / upsample - r) % big[1]
        else:
            ky, kx = float(peak[0]), float(peak[1])
        dy = ky - big[0] if ky >= big[0] / 2 else ky
        dx = kx - big[1] if kx >= big[1] / 2 else kx
        shift = np.array([dy, dx], dtype=np.float32)
        corr = float(cc[peak])
        n1 = float(np.linalg.norm(img1w))
        n2 = float(np.linalg.norm(img2w))
        if corr > 0 and n1 > 0 and n2 > 0:
            error = -np.log(corr / (n1 * n2))
        else:
            error = np.inf
        return shift, error

    def patch_register_pair():
        from ashlar.reg import EdgeAligner

        def register_pair_big(self, t1, t2):
            key = tuple(sorted((t1, t2)))
            try:
                shift, error = self._cache[key]
            except KeyError:
                smin = self.intersection(key[0], key[1]).shape
                smax = np.round(self.metadata.size * 0.1)
                sizes = [smin]
                while any(sizes[-1] < smax):
                    sizes.append(sizes[-1] * 2)
                cap = np.round(self.metadata.size * 0.5).astype(int)
                s = np.array(smin)
                while any(s < cap):
                    s = np.minimum(s * 2, cap)
                    sizes.append(s.copy())
                results = [self._register(key[0], key[1], sz) for sz in sizes]
                shift, _ = min(results, key=lambda r: r[1])
                _, o1, o2 = self.overlap(key[0], key[1], shift=shift)
                error = ashlar.utils.nccw(o1, o2, self.filter_sigma)
                self._cache[key] = (shift, error)
            if t1 > t2:
                shift = -shift
            return shift.copy(), error

        EdgeAligner.register_pair = register_pair_big

    ashlar.utils.register = register_padded
    patch_register_pair()
    _patched = True


class _LogWriter:
    def __init__(self, log):
        self.log = log
        self.buf = ''

    def write(self, s):
        self.buf += s
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            if line.strip():
                self.log(line)

    def flush(self):
        if self.buf.strip():
            self.log(self.buf.strip())
            self.buf = ''


@contextlib.contextmanager
def _capture(log):
    w = _LogWriter(log)
    with contextlib.redirect_stdout(w), contextlib.redirect_stderr(w):
        yield


def stitch_one(folder, scan, out_path, overlap=OVERLAP, log=print, capture=True):
    ensure_patch()
    from ashlar.scripts.ashlar import main
    pattern = ('filepattern|' + folder + '|pattern=' + scan +
               ' X{col:03} Y{row:03}.tif|overlap=' + str(overlap) +
               '|pixel_size=' + str(PIXEL_SIZE))
    argv = ['ashlar', pattern, '--output', out_path,
            '--maximum-shift', str(MAX_SHIFT_UM),
            '--align-channel', str(ALIGN_CHANNEL),
            '--filter-sigma', str(FILTER_SIGMA)]
    ctx = _capture(log) if capture else contextlib.nullcontext()
    with ctx:
        try:
            main(argv)
        except SystemExit:
            pass
    return os.path.isfile(out_path)


def run_console(folder, args):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    sets = scan_folder(folder)
    if not sets:
        print('No tile files matching "<set> X### Y###.tif" in ' + folder)
        return 1
    report(sets, folder)
    if args.scan_only:
        return 0
    if args.sets:
        want = [x.strip() for x in args.sets.split(',') if x.strip()]
        known = {s.scan for s in sets}
        bad = [w for w in want if w not in known]
        if bad:
            print('Unknown set number(s): ' + ', '.join(bad))
            return 1
        sets = [s for s in sets if s.scan in want]
    outdir = os.path.join(folder, 'stitched')
    os.makedirs(outdir, exist_ok=True)
    failed = []
    for s in sets:
        out = os.path.join(outdir, '%s_ashlar%s.ome.tif' % (s.scan, args.suffix))
        print('--- %s: %d tiles ---' % (s.scan, len(s.files)))
        if os.path.exists(out) and not args.overwrite:
            print('    %s already exists, skipping (use --overwrite to re-stitch)' % out)
            continue
        t0 = time.time()
        try:
            ok = stitch_one(folder, s.scan, out, overlap=args.overlap,
                            capture=False)
        except Exception as e:
            print('    ERROR: %s' % e)
            ok = False
        print('    %s in %.0fs -> %s' % ('done' if ok else 'FAILED',
                                        time.time() - t0, out))
        if not ok:
            failed.append(s.scan)
    if failed:
        print('Failed: ' + ', '.join(failed))
        return 1
    return 0


def run_gui(root=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    here = os.path.dirname(os.path.abspath(__file__))

    class App:
        def __init__(self, root):
            self.root = root
            root.title('Ashlar Stitcher')
            missing = []
            for mod, label in (('ashlar', 'ashlar'), ('PIL', 'Pillow')):
                try:
                    __import__(mod)
                except ImportError:
                    missing.append(label)
            self.env_warning = (
                'WARNING: this Python lacks %s - close and relaunch via '
                'stitch_here.bat (uses %s)' % (', '.join(missing), CONDA_PY)
            ) if missing else ''
            self.folder = tk.StringVar(value=here)
            self.suffix = tk.StringVar(value=DEFAULT_SUFFIX)
            self.overwrite = tk.BooleanVar(value=False)
            self.checked = {}
            self.q = queue.Queue()
            self.running = False

            top = ttk.Frame(root, padding=8)
            top.pack(fill='x')
            ttk.Entry(top, textvariable=self.folder).pack(side='left', fill='x', expand=True)
            ttk.Button(top, text='Browse...', command=self.browse).pack(side='left', padx=4)
            ttk.Button(top, text='Scan', command=self.rescan).pack(side='left')

            mid = ttk.Frame(root, padding=(8, 0, 8, 4))
            mid.pack(fill='x')
            ttk.Label(mid, text='Output suffix').pack(side='left')
            ttk.Entry(mid, textvariable=self.suffix, width=10).pack(side='left', padx=4)
            ttk.Checkbutton(mid, text='Overwrite existing',
                            variable=self.overwrite).pack(side='left', padx=8)
            self.run_btn = ttk.Button(mid, text='Stitch checked sets', command=self.start)
            self.run_btn.pack(side='right')

            cols = ('sel', 'scan', 'tiles', 'grid', 'note')
            self.tree = ttk.Treeview(root, columns=cols, show='headings', height=9)
            for c, w in zip(cols, (40, 70, 60, 70, 360)):
                self.tree.heading(c, text=c.capitalize())
                self.tree.column(c, width=w, anchor='w')
            self.tree.pack(fill='both', expand=True, padx=8, pady=4)
            self.tree.bind('<Button-1>', self.on_click)

            self.logbox = tk.Text(root, height=12, state='disabled')
            self.logbox.pack(fill='both', expand=True, padx=8, pady=(0, 2))
            self.status = ttk.Label(root, text='', padding=(8, 2))
            self.status.pack(fill='x')
            root.after(200, self.rescan)
            root.after(150, self.poll)

        def browse(self):
            d = filedialog.askdirectory(initialdir=self.folder.get() or here)
            if d:
                self.folder.set(d)
                self.rescan()

        def rescan(self):
            if self.running:
                return
            self.tree.delete(*self.tree.get_children())
            self.checked.clear()
            folder = self.folder.get()
            try:
                sets = scan_folder(folder)
            except Exception as e:
                self.set_status('Cannot scan %s: %s' % (folder, e))
                return
            for s in sets:
                gx, gy = s.grid()
                iid = self.tree.insert('', 'end', values=(
                    '[x]', s.scan, len(s.files), '%dx%d' % (gx, gy), s.note()))
                self.checked[iid] = True
            self.set_status('%d set(s) found. Uncheck rows to skip, then Stitch.'
                            % len(sets))

        def on_click(self, e):
            if self.running or self.tree.identify_column(e.x) != '#1':
                return
            iid = self.tree.identify_row(e.y)
            if iid in self.checked:
                self.checked[iid] = not self.checked[iid]
                self.tree.set(iid, 'sel', '[x]' if self.checked[iid] else '[ ]')

        def start(self):
            folder = self.folder.get()
            picks = [self.tree.set(i, 'scan') for i in self.tree.get_children()
                     if self.checked.get(i)]
            if not picks:
                messagebox.showinfo('Ashlar Stitcher', 'Nothing checked to stitch.')
                return
            if not os.path.isdir(folder):
                messagebox.showerror('Ashlar Stitcher', 'Folder not found: ' + folder)
                return
            try:
                import ashlar  # noqa: F401
            except ImportError:
                messagebox.showerror(
                    'Ashlar Stitcher',
                    'ashlar is not available in this Python.\n'
                    'Launch via stitch_here.bat or %s.' % CONDA_PY)
                return
            outdir = os.path.join(folder, 'stitched')
            self.running = True
            self.run_btn.state(['disabled'])
            self.log_write('')
            self.set_status('Stitching: ' + ', '.join(picks))
            threading.Thread(
                target=self.worker,
                args=(folder, outdir, picks, self.suffix.get(), self.overwrite.get()),
                daemon=True).start()

        def worker(self, folder, outdir, picks, suffix, overwrite):
            os.makedirs(outdir, exist_ok=True)
            for scan in picks:
                out = os.path.join(outdir, '%s_ashlar%s.ome.tif' % (scan, suffix))
                self.q.put(('log', '--- %s -> %s ---' % (scan, out)))
                if os.path.exists(out) and not overwrite:
                    self.q.put(('log',
                                '    already exists, skipping (enable Overwrite to re-stitch)'))
                    continue
                t0 = time.time()
                try:
                    ok = stitch_one(folder, scan, out,
                                    log=lambda m: self.q.put(('log', m)))
                except Exception as e:
                    self.q.put(('log', '    ERROR: %s' % e))
                    ok = False
                self.q.put(('log', '    %s in %.0fs'
                            % ('done' if ok else 'FAILED', time.time() - t0)))
            self.q.put(('done', None))

        def poll(self):
            try:
                while True:
                    kind, msg = self.q.get_nowait()
                    if kind == 'log':
                        self.log_write(msg)
                    else:
                        self.running = False
                        self.run_btn.state(['!disabled'])
                        self.set_status('Finished.')
                        self.rescan()
            except queue.Empty:
                pass
            self.root.after(150, self.poll)

        def log_write(self, msg):
            self.logbox['state'] = 'normal'
            self.logbox.insert('end', msg + '\n')
            self.logbox.see('end')
            self.logbox['state'] = 'disabled'

        def set_status(self, msg):
            if self.env_warning:
                msg = self.env_warning + ' | ' + msg
            self.status['text'] = msg

    if root is None:
        root = tk.Tk()
        owner = True
    else:
        owner = False
    root.geometry('740x660')
    app = App(root)
    if owner:
        root.mainloop()
    return app


def main():
    ap = argparse.ArgumentParser(
        description='Auto-detect tile sets and stitch with Ashlar (GUI if no folder given).')
    ap.add_argument('folder', nargs='?', help='folder of tiles (omit to open the GUI)')
    ap.add_argument('--suffix', default=DEFAULT_SUFFIX,
                    help='output name suffix (default: %s)' % DEFAULT_SUFFIX)
    ap.add_argument('--sets', default='', help='comma-separated set numbers to stitch')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--scan-only', action='store_true')
    ap.add_argument('--overlap', type=float, default=OVERLAP)
    args = ap.parse_args()
    if args.folder is None:
        run_gui()
        return
    folder = args.folder
    if os.path.isfile(folder):
        folder = os.path.dirname(folder)
    sys.exit(run_console(folder, args))


if __name__ == '__main__':
    main()
