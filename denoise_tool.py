"""CBI SEM denoiser - applies trained models to folders of SEM images.

Tile naming: <set> X### Y###.tif  (same detector as the stitcher). Folders of
plain SEM images without tile coordinates are denoised as a single batch;
stitching needs tile coordinates, so only tiled folders can be stitched.
The JEOL data bar below the content (detected from the JEOL sidecar at any
resolution) is copied through untouched - only image content is denoised.

Ways to use:
  1. Place denoise_tool.py + denoise_here.bat in a folder of tiles and
     double-click denoise_here.bat  -> denoises every set into denoised/
  2. Run install_sendto_denoise.bat once, then right-click any folder ->
     Send To -> "Denoise with CBI model"
  3. python denoise_tool.py                  -> GUI (folder, model, set list)
  4. python denoise_tool.py <folder> [--scan-only] [--sets 00001,00003]
        [--model NAME-or-path] [--invert] [--overwrite]

Model selection: models\\ next to this script holds trained .pt files plus a
manifest.json written by train_denoiser.py. The tool sniffs the JEOL .txt
sidecar in the tile folder and picks the model whose recorded settings
match; otherwise the first model with a warning. --invert applies 255-x
after denoising (inverted display polarity, white background).
"""
import argparse
import os
import queue
import re
import sys
import threading
import time

from stitch_tool import CONDA_PY, TILE_RE, scan_folder

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DEFAULT_SUFFIX = ''
TILE = 512
OVERLAP = 64


class FlatSet:
    """TIFFs in a folder that do not follow the '<set> X### Y###.tif' pattern."""

    def __init__(self, folder):
        self.scan = 'all'
        self.folder = folder
        self.files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(('.tif', '.tiff'))
            and not TILE_RE.match(f)
            and os.path.isfile(os.path.join(folder, f)))

    def grid(self):
        return len(self.files), 1


def scan_any(folder):
    """Tile sets if present, plus one FlatSet for any non-pattern TIFFs."""
    sets = scan_folder(folder)
    loose = FlatSet(folder)
    if loose.files:
        sets.append(loose)
    return sets


def sidecar_text(folder):
    for name in os.listdir(folder):
        if name.lower().endswith('.txt'):
            try:
                with open(os.path.join(folder, name), encoding='utf-8',
                          errors='replace') as fh:
                    return fh.read()
            except OSError:
                continue
    return ''


def load_manifest(models_dir=MODELS_DIR):
    import json
    path = os.path.join(models_dir, 'manifest.json')
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh).get('models', [])
    except (OSError, ValueError):
        return []


def pick_model(entries, folder, log=print):
    """Match recorded model settings against the folder's JEOL sidecar."""
    text = sidecar_text(folder).lower()
    best, score = None, 0
    for e in entries:
        s = 0
        cond = e.get('condition') or {}
        for v in (cond.values() if isinstance(cond, dict) else []):
            if v and str(v).lower() in text:
                s += 1
        if s > score:
            best, score = e, s
    if best is None and entries:
        best = entries[0]
        if log:
            log('WARNING: no model settings match this folder\'s sidecar - '
                'using "%s"' % best.get('name'))
    return best


def load_model(path):
    import torch
    from train_denoiser import build_unet
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = build_unet(ckpt.get('base', 24))
    model.load_state_dict(ckpt['state_dict'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    return model, device


def denoise_one(folder, scan, model, device, out_dir, invert=False,
                log=print, overwrite=False):
    from train_denoiser import load_sem_tif, tiled_predict, bar_rows
    os.makedirs(out_dir, exist_ok=True)
    done = failed = skipped = 0
    for name in scan.files:
        out = os.path.join(out_dir, name)
        if os.path.exists(out) and not overwrite:
            skipped += 1
            continue
        try:
            img = load_sem_tif(os.path.join(folder, name), crop_bar=False) / 255.0
            h = img.shape[0]
            nbar = bar_rows(img, os.path.join(folder, name))
            den = tiled_predict(model, img[:h - nbar] if nbar else img,
                                device, tile=TILE, overlap=OVERLAP)
            import numpy as np
            import tifffile
            arr = np.clip(den, 0, 1)
            if nbar:
                arr = np.vstack([arr, img[h - nbar:]])
            if invert:
                arr = 1.0 - arr
            tifffile.imwrite(out, (arr * 255.0 + 0.5).astype('uint8'))
            done += 1
        except Exception as e:
            log('    ERROR %s: %s' % (name, e))
            failed += 1
    return done, skipped, failed


def report(sets, folder, entries, chosen):
    print('%d image set(s) in %s' % (len(sets), folder))
    for s in sets:
        if isinstance(s, FlatSet):
            print('  %s: %d images (no tile pattern)'
                  % (s.scan, len(s.files)))
        else:
            gx, gy = s.grid()
            print('  %s: %d tiles, grid %dx%d' % (s.scan, len(s.files), gx, gy))
    if entries:
        print('model: %s' % chosen.get('name'))
    else:
        print('WARNING: no models found in %s - train one with train_denoiser.py' % MODELS_DIR)


def run_console(folder, args):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    sets = scan_any(folder)
    if not sets:
        print('No .tif/.tiff images in ' + folder)
        return 1
    entries = load_manifest(args.models_dir)
    if args.model:
        match = [e for e in entries if e.get('name') == args.model
                 or e.get('file') == args.model]
        chosen = match[0] if match else {'name': args.model,
                                         'file': os.path.basename(args.model)}
        chosen['_path'] = args.model if os.path.isfile(args.model) else os.path.join(
            args.models_dir, chosen['file'])
    else:
        chosen = pick_model(entries, folder) or {}
        chosen = dict(chosen)
        chosen['_path'] = os.path.join(args.models_dir, chosen.get('file', ''))
    report(sets, folder, entries, chosen)
    if args.scan_only:
        return 0
    if not os.path.isfile(chosen.get('_path', '')):
        print('Model file not found: %s' % chosen.get('_path'))
        return 1
    try:
        model, device = load_model(chosen['_path'])
    except Exception as e:
        print('Cannot load model (%s): %s' % (chosen['_path'], e))
        return 1
    print('device: %s' % device)
    if args.sets:
        want = [x.strip() for x in args.sets.split(',') if x.strip()]
        known = {s.scan for s in sets}
        bad = [w for w in want if w not in known]
        if bad:
            print('Unknown set number(s): ' + ', '.join(bad))
            return 1
        sets = [s for s in sets if s.scan in want]
    out_dir = os.path.join(folder, 'denoised')
    failures = []
    for s in sets:
        print('--- %s: %d tiles ---' % (s.scan, len(s.files)))
        t0 = time.time()
        done, skipped, failed = denoise_one(
            folder, s, model, device, out_dir, invert=args.invert,
            overwrite=args.overwrite)
        print('    %d denoised, %d skipped, %d failed in %.0fs -> %s'
              % (done, skipped, failed, time.time() - t0, out_dir))
        if failed:
            failures.append(s.scan)
    if failures:
        print('Failed sets: ' + ', '.join(failures))
        return 1
    return 0


def run_gui(root=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    here = os.path.dirname(os.path.abspath(__file__))

    class App:
        def __init__(self, root):
            self.root = root
            root.title('CBI SEM Denoiser')
            self.folder = tk.StringVar(value=here)
            self.invert = tk.BooleanVar(value=False)
            self.overwrite = tk.BooleanVar(value=False)
            self.checked = {}
            self.q = queue.Queue()
            self.running = False
            self.models = load_manifest()

            top = ttk.Frame(root, padding=8)
            top.pack(fill='x')
            ttk.Entry(top, textvariable=self.folder).pack(side='left', fill='x', expand=True)
            ttk.Button(top, text='Browse...', command=self.browse).pack(side='left', padx=4)
            ttk.Button(top, text='Scan', command=self.rescan).pack(side='left')

            mid = ttk.Frame(root, padding=(8, 0, 8, 4))
            mid.pack(fill='x')
            ttk.Label(mid, text='Model').pack(side='left')
            names = [m.get('name', m.get('file', '?')) for m in self.models]
            self.model_box = ttk.Combobox(mid, values=names, width=28,
                                          state='readonly' if names else 'disabled')
            if names:
                self.model_box.current(0)
            self.model_box.pack(side='left', padx=4)
            ttk.Checkbutton(mid, text='Invert', variable=self.invert).pack(side='left', padx=4)
            ttk.Checkbutton(mid, text='Overwrite existing',
                            variable=self.overwrite).pack(side='left', padx=4)
            self.run_btn = ttk.Button(mid, text='Denoise checked sets', command=self.start)
            self.run_btn.pack(side='right')

            cols = ('sel', 'scan', 'tiles', 'grid')
            self.tree = ttk.Treeview(root, columns=cols, show='headings', height=9)
            for c, w in zip(cols, (40, 70, 60, 70)):
                self.tree.heading(c, text=c.capitalize())
                self.tree.column(c, width=w, anchor='w')
            self.tree.pack(fill='both', expand=True, padx=8, pady=4)
            self.tree.bind('<Button-1>', self.on_click)

            self.logbox = tk.Text(root, height=14, state='disabled')
            self.logbox.pack(fill='both', expand=True, padx=8, pady=(0, 2))
            self.status = ttk.Label(root, text='', padding=(8, 2))
            self.status.pack(fill='x')
            if not self.models:
                self.set_status('No models in %s - train one with train_denoiser.py'
                                % MODELS_DIR)
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
            try:
                sets = scan_any(self.folder.get())
            except Exception as e:
                self.set_status('Cannot scan: %s' % e)
                return
            for s in sets:
                note = ('%d images' % len(s.files) if isinstance(s, FlatSet)
                        else '%dx%d' % s.grid())
                iid = self.tree.insert('', 'end', values=(
                    '[x]', s.scan, len(s.files), note))
                self.checked[iid] = True
            self.set_status('%d set(s) found. Uncheck rows to skip, then Denoise.'
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
                messagebox.showinfo('CBI SEM Denoiser', 'Nothing checked to denoise.')
                return
            if not os.path.isdir(folder):
                messagebox.showerror('CBI SEM Denoiser', 'Folder not found: ' + folder)
                return
            name = self.model_box.get()
            entry = next((m for m in self.models
                          if m.get('name', m.get('file')) == name), None)
            path = os.path.join(MODELS_DIR, entry['file']) if entry else ''
            if not entry or not os.path.isfile(path):
                messagebox.showerror('CBI SEM Denoiser', 'Model file missing: ' + path)
                return
            self.running = True
            self.run_btn.state(['disabled'])
            self.log_write('')
            self.set_status('Denoising: ' + ', '.join(picks))
            threading.Thread(
                target=self.worker,
                args=(folder, path, picks, self.invert.get(), self.overwrite.get()),
                daemon=True).start()

        def worker(self, folder, model_path, picks, invert, overwrite):
            try:
                model, device = load_model(model_path)
                self.q.put(('log', 'model: %s (%s)' % (os.path.basename(model_path), device)))
                out_dir = os.path.join(folder, 'denoised')
                for scan in picks:
                    sets = {s.scan: s for s in scan_any(folder)}
                    if scan not in sets:
                        continue
                    self.q.put(('log', '--- %s -> %s ---' % (scan, out_dir)))
                    t0 = time.time()
                    done, skipped, failed = denoise_one(
                        folder, sets[scan], model, device, out_dir, invert=invert,
                        log=lambda m: self.q.put(('log', m)), overwrite=overwrite)
                    self.q.put(('log', '    %d denoised, %d skipped, %d failed in %.0fs'
                                % (done, skipped, failed, time.time() - t0)))
            except Exception as e:
                self.q.put(('log', 'ERROR: %s' % e))
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
            self.status['text'] = msg

    if root is None:
        root = tk.Tk()
        owner = True
    else:
        owner = False
    root.geometry('640x640')
    app = App(root)
    if owner:
        root.mainloop()
    return app


def main():
    ap = argparse.ArgumentParser(
        description='Apply trained CBI denoiser models to tile folders '
                    '(GUI if no folder given).')
    ap.add_argument('folder', nargs='?', help='folder of tiles (omit to open the GUI)')
    ap.add_argument('--scan-only', action='store_true')
    ap.add_argument('--sets', default='', help='comma-separated set numbers to denoise')
    ap.add_argument('--model', default='', help='model name (from manifest) or .pt path')
    ap.add_argument('--models-dir', default=MODELS_DIR)
    ap.add_argument('--invert', action='store_true', help='apply 255-x after denoising')
    ap.add_argument('--overwrite', action='store_true')
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
