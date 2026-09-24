import os
import sys
import numpy as np
import scipy.ndimage
import ashlar.utils

_orig_register = ashlar.utils.register

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


def run_test():
    rng = np.random.default_rng(0)
    base = scipy.ndimage.gaussian_filter(rng.random((256, 384)).astype(np.float32), 5)
    for s in ([17.0, -23.0], [-41.0, 11.0], [5.4, 7.7], [0.0, 0.0]):
        img1 = base
        img2 = scipy.ndimage.shift(base, s, order=1)
        sh_a, e_a = _orig_register(img1, img2, 15)
        sh_p, e_p = register_padded(img1, img2, 15)
        ok = (np.allclose(sh_p, -np.array(s), atol=1.5)
              and np.allclose(sh_a, sh_p, atol=1.0))
        print(f's={s} ashlar={np.round(sh_a, 2)} mine={np.round(sh_p, 2)} match={ok}')
        assert ok, 'convention mismatch'
    print('synthetic test PASSED')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'test':
        run_test()
    else:
        scan, out = sys.argv[1], sys.argv[2]
        ov = sys.argv[3] if len(sys.argv) > 3 else '0.20'
        ashlar.utils.register = register_padded
        patch_register_pair()
        from ashlar.scripts.ashlar import main
        base = sys.argv[4] if len(sys.argv) > 4 else os.getcwd()
        pattern = f"filepattern|{base}|pattern={scan} X{{col:03}} Y{{row:03}}.tif|overlap={ov}|pixel_size=0.5"
        sys.argv = ['ashlar', pattern, '--output', out,
                    '--maximum-shift', '150', '--align-channel', '0', '--filter-sigma', '15']
        main(sys.argv)
