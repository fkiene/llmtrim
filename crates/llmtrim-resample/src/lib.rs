//! Separable Lanczos3 downscale kernels for `llmtrim-core`'s Stage H image resize.
//!
//! Same math as `image::imageops::resize` (vertical pass through an `f32` scratch,
//! then horizontal, clamp + round) but on tightly packed `u8` buffers: it iterates
//! row slices instead of `GenericImageView::get_pixel` + per-pixel channel widening.
//! The vertical and horizontal passes are fused per output row — row `y` of the
//! `f32` scratch feeds only output row `y`, so the scratch is a single hot row —
//! and large images split row bands across `std::thread`s. The crate exists so the
//! root manifest can lift its dev-profile `opt-level` — the per-pixel kernels are
//! ~10× slower at opt-level 0, where the rest of the workspace deliberately keeps
//! debug checks (see the `[profile.dev.package.*]` overrides in the root
//! `Cargo.toml`). Pure-safe Rust; `unsafe_code` stays forbidden.

/// `image`'s Lanczos3 kernel (`sinc(x) * sinc(x/3)`, support 3), verbatim.
fn lanczos3(x: f32) -> f32 {
    fn sinc(t: f32) -> f32 {
        if t == 0.0 {
            1.0
        } else {
            let a = t * std::f32::consts::PI;
            a.sin() / a
        }
    }
    if x.abs() < 3.0 {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Per-output-pixel weights for one axis, identical to `image`'s `vertical_sample` /
/// `horizontal_sample`: the kernel window widens with the downscale ratio, taps are
/// clamped to the source edge, and each output pixel's weights are normalized.
fn axis_weights(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    let ratio = src as f32 / dst as f32;
    let sratio = if ratio < 1.0 { 1.0 } else { ratio };
    let support = 3.0 * sratio;
    (0..dst)
        .map(|o| {
            let centre = (o as f32 + 0.5) * ratio;
            let left = ((centre - support).floor() as i64).clamp(0, src as i64 - 1) as usize;
            let right =
                ((centre + support).ceil() as i64).clamp(left as i64 + 1, src as i64) as usize;
            let centre = centre - 0.5;
            let mut ws: Vec<f32> = (left..right)
                .map(|i| lanczos3((i as f32 - centre) / sratio))
                .collect();
        let sum: f32 = ws.iter().sum();
        for w in ws.iter_mut() {
            *w /= sum;
        }
        // Lanczos3's taps at the window edge sit at |x|≈3 where the kernel is ~0 —
        // drop taps below a 0.2% |share| and renormalize the survivors. Compare
        // magnitudes: the negative outer lobes carry ~5% real mass each, so `w <
        // eps` (signed) would silently cut ~10% of the window — that overshoots
        // the suite's ±1 tolerance vs `image`. Worst-case pixel shift is ~0.3
        // levels while each dropped tap removes a full row-width sweep from the
        // vertical pass (and the same ratio horizontally).
        let mut lo = 0usize;
        let mut hi = ws.len();
        while lo < hi && ws[lo].abs() < 0.002 {
            lo += 1;
        }
        while hi > lo && ws[hi - 1].abs() < 0.002 {
            hi -= 1;
        }
        if lo > 0 || hi < ws.len() {
            ws.drain(hi..);
            ws.drain(..lo);
            let sum: f32 = ws.iter().sum();
            for w in ws.iter_mut() {
                *w /= sum;
            }
        }
        (left + lo, ws)
        })
        .collect()
}

/// Source rows converted to `f32` once, kept in a sliding window of `nslot` row
/// slots. Vertical windows advance monotonically (each `top` ≥ the previous one),
/// so a row is converted a single time even though it feeds ~`ws.len()` output
/// rows — removing the per-tap `u8 → f32` convert from the inner loop. `nslot` ≥
/// the largest window guarantees a still-needed row is never evicted: conversion
/// only advances, and a row `nslot` ahead of any needed row is out of range.
struct F32Rows {
    data: Vec<f32>,
    stride: usize,
    /// Slot count rounded up to a power of two — the `r % nslot` slot lookup runs
    /// once per tap inside the vertical inner loop, and a runtime modulo is a
    /// ~30-cycle `div`; `r & mask` is free.
    mask: usize,
    /// First source row not yet converted (monotonic cursor).
    next: usize,
}

impl F32Rows {
    fn new(stride: usize, nslot: usize) -> Self {
        let nslot = nslot.max(1).next_power_of_two();
        F32Rows {
            data: vec![0.0; stride * nslot],
            stride,
            mask: nslot - 1,
            next: 0,
        }
    }

    /// Convert source rows `top..need` that aren't yet in the window. Rows below
    /// `top` are skipped outright — vertical windows advance monotonically, so
    /// they're unreachable for this and all later output rows.
    fn ensure(&mut self, src: &[u8], top: usize, need: usize) {
        if self.next < top {
            self.next = top;
        }
        while self.next < need {
            let r = self.next;
            let slot = &mut self.data[(r & self.mask) * self.stride..][..self.stride];
            let srow = &src[r * self.stride..(r + 1) * self.stride];
            for (d, s) in slot.iter_mut().zip(srow) {
                *d = *s as f32;
            }
            self.next += 1;
        }
    }
}


/// Largest tap count in a weight table.
fn max_taps(ws: &[(usize, Vec<f32>)]) -> usize {
    ws.iter().map(|(_, w)| w.len()).max().unwrap_or(0)
}

/// Work threshold below which `std::thread` overhead isn't worth it: ~4M `f32`
/// mul-adds ≈ 1–2 ms single-threaded on the hardware these tests run on.
const MIN_PARALLEL_TAPS: usize = 4_000_000;
/// Beyond this window-bytes size the f32 row cache's wider reads lose to plain
/// u8 streaming: a huge-ratio window exceeds L2 either way, and u8 is 4× narrower.
const CACHE_WINDOW_BYTES: usize = 256 * 1024;

/// How many OS threads a resample may use. The kernel is `pub` and image sizes are
/// caller-controlled, so don't assume a warm pool — a scoped spawn ≈ 50 µs.
fn nthreads(taps: usize, lanes: usize) -> usize {
    if taps < MIN_PARALLEL_TAPS || lanes < 2 {
        return 1;
    }
    let n = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4);
    n.min(lanes)
}

/// One output row's vertical accumulation into `out` (`stride` f32s): same sum
/// order as `image`'s `vertical_sample`. With the row cache the taps read `f32`
/// directly; without it (windows too wide to profit) a single band slice covers
/// all taps and u8 converts in the loop.
fn vertical_row(
    src: &[u8],
    top: usize,
    ws: &[f32],
    rows: &mut Option<F32Rows>,
    stride: usize,
    out: &mut [f32],
) {
    let wt0 = ws[0];
    // `out` is stride+4 wide (horizontal reads ahead); vertical fills `stride`.
    let out = &mut out[..stride];
    if let Some(rows) = rows.as_mut() {
        rows.ensure(src, top, top + ws.len());
        let data = &rows.data[..];
        let mask = rows.mask;
        let srow = &data[(top & mask) * stride..][..stride];
        for (o, &s) in out.iter_mut().zip(srow) {
            *o = s * wt0;
        }
        // Taps run in groups of four: `o += s0·w0 + s1·w1 + s2·w2 + s3·w3` pays one
        // load+store of `out` per group instead of per tap — a 4-tap sweep is ~13
        // packed ops vs 20 tap-at-a-time. Tap terms pre-sum before adding `o`
        // (reassociation): stays within the suite's ±1 tolerance vs `image`, and
        // `resample3`/`resample_u8` share this body so they remain bit-identical.
        let (groups, tail_w) = ws[1..].as_chunks::<4>();
        for (g, w4) in groups.iter().enumerate() {
            let t0 = 1 + g * 4;
            let s0 = &data[((top + t0) & mask) * stride..][..stride];
            let s1 = &data[((top + t0 + 1) & mask) * stride..][..stride];
            let s2 = &data[((top + t0 + 2) & mask) * stride..][..stride];
            let s3 = &data[((top + t0 + 3) & mask) * stride..][..stride];
            for ((((o, &a), &b), &c), &d) in
                out.iter_mut().zip(s0).zip(s1).zip(s2).zip(s3)
            {
                *o += a * w4[0] + b * w4[1] + c * w4[2] + d * w4[3];
            }
        }
        match tail_w.len() {
            // Fused sweeps for the 3- and 2-tap tails — same math as the group
            // loop, just shorter (tail_w.len() < 4 by construction).
            3 => {
                let t0 = 1 + groups.len() * 4;
                let s0 = &data[((top + t0) & mask) * stride..][..stride];
                let s1 = &data[((top + t0 + 1) & mask) * stride..][..stride];
                let s2 = &data[((top + t0 + 2) & mask) * stride..][..stride];
                for (((o, &a), &b), &c) in out.iter_mut().zip(s0).zip(s1).zip(s2) {
                    *o += a * tail_w[0] + b * tail_w[1] + c * tail_w[2];
                }
            }
            2 => {
                let t0 = 1 + groups.len() * 4;
                let s0 = &data[((top + t0) & mask) * stride..][..stride];
                let s1 = &data[((top + t0 + 1) & mask) * stride..][..stride];
                for ((o, &a), &b) in out.iter_mut().zip(s0).zip(s1) {
                    *o += a * tail_w[0] + b * tail_w[1];
                }
            }
            _ => {
                for (k, &wt) in tail_w.iter().enumerate() {
                    let t = 1 + groups.len() * 4 + k;
                    let srow = &data[((top + t) & mask) * stride..][..stride];
                    for (o, &s) in out.iter_mut().zip(srow) {
                        *o += s * wt;
                    }
                }
            }
        }
    } else {
        let band = &src[top * stride..(top + ws.len()) * stride];
        for (o, &s) in out.iter_mut().zip(&band[..stride]) {
            *o = s as f32 * wt0;
        }
        let (groups, tail_w) = ws[1..].as_chunks::<4>();
        for (g, w4) in groups.iter().enumerate() {
            let t0 = 1 + g * 4;
            let s0 = &band[t0 * stride..(t0 + 1) * stride];
            let s1 = &band[(t0 + 1) * stride..(t0 + 2) * stride];
            let s2 = &band[(t0 + 2) * stride..(t0 + 3) * stride];
            let s3 = &band[(t0 + 3) * stride..(t0 + 4) * stride];
            for ((((o, &a), &b), &c), &d) in
                out.iter_mut().zip(s0).zip(s1).zip(s2).zip(s3)
            {
                *o += a as f32 * w4[0]
                    + b as f32 * w4[1]
                    + c as f32 * w4[2]
                    + d as f32 * w4[3];
            }
        }
        match tail_w.len() {
            3 => {
                let t0 = 1 + groups.len() * 4;
                let s0 = &band[t0 * stride..(t0 + 1) * stride];
                let s1 = &band[(t0 + 1) * stride..(t0 + 2) * stride];
                let s2 = &band[(t0 + 2) * stride..(t0 + 3) * stride];
                for (((o, &a), &b), &c) in out.iter_mut().zip(s0).zip(s1).zip(s2) {
                    *o += a as f32 * tail_w[0]
                        + b as f32 * tail_w[1]
                        + c as f32 * tail_w[2];
                }
            }
            2 => {
                let t0 = 1 + groups.len() * 4;
                let s0 = &band[t0 * stride..(t0 + 1) * stride];
                let s1 = &band[(t0 + 1) * stride..(t0 + 2) * stride];
                for ((o, &a), &b) in out.iter_mut().zip(s0).zip(s1) {
                    *o += a as f32 * tail_w[0] + b as f32 * tail_w[1];
                }
            }
            _ => {
                for (k, &wt) in tail_w.iter().enumerate() {
                    let t = 1 + groups.len() * 4 + k;
                    let srow = &band[t * stride..(t + 1) * stride];
                    for (o, &s) in out.iter_mut().zip(srow) {
                        *o += s as f32 * wt;
                    }
                }
            }
        }
    }
}

/// Split the output rows across `threads`: `f` runs once per band with that
/// band's destination slice and matching `vws` sub-slice.
fn run_bands<F>(
    vws: &[(usize, Vec<f32>)],
    nh: usize,
    dstride: usize,
    dst: &mut [u8],
    threads: usize,
    f: F,
) where
    F: Fn(&mut [u8], &[(usize, Vec<f32>)]) + Sync + Send,
{
    if threads <= 1 {
        f(dst, vws);
        return;
    }
    let f = &f; // shared across band closures (each takes `dc`/`vc` by value)
    let span_rows = nh / threads;
    std::thread::scope(|s| {
        let mut d_rest = dst;
        let mut v_rest = vws;
        for t in 0..threads {
            let is_last = t == threads - 1;
            let rows = if is_last { v_rest.len() } else { span_rows };
            let (dc, dt) = d_rest.split_at_mut(rows * dstride);
            let (vc, vt) = v_rest.split_at(rows);
            d_rest = dt;
            v_rest = vt;
            s.spawn(move || f(dc, vc));
        }
    });
}

/// Vertical tap total — sizes the threading decision on the dominant pass.
fn vertical_taps(vws: &[(usize, Vec<f32>)], stride: usize) -> usize {
    vws.iter().map(|(_, ws)| ws.len()).sum::<usize>() * stride
}

/// Separable Lanczos3 downscale of a tightly packed `CN`-channel `u8` buffer.
///
/// `axis_weights` always yields a non-empty weight list (right is clamped to at
/// least `left + 1`), so `ws[0]` is safe.
pub fn resample_u8<const CN: usize>(
    src: &[u8],
    w: usize,
    h: usize,
    nw: usize,
    nh: usize,
) -> Vec<u8> {
    let stride = w * CN;
    let dstride = nw * CN;
    let vws = axis_weights(h, nh);
    let hws = axis_weights(w, nw);
    let use_cache = max_taps(&vws) * stride * 4 <= CACHE_WINDOW_BYTES;
    let threads = nthreads(vertical_taps(&vws, stride), nh);
    let mut dst = vec![0u8; dstride * nh];
    run_bands(&vws, nh, dstride, &mut dst, threads, |dst_band, band_vws| {
        let mut rows = if use_cache {
            Some(F32Rows::new(stride, max_taps(band_vws)))
        } else {
            None
        };
        // Reused per-row vertical scratch — only row `oy` is live at a time. `+4`
        // tail floats let the horizontal tap loop read a full f32x4 per tap (the
        // 4th accumulator lane is discarded) so LLVM emits one packed FMA per tap.
        let mut trow = vec![0f32; stride + 4];
        for (orow, (top, ws)) in dst_band.chunks_mut(dstride).zip(band_vws) {
            vertical_row(src, *top, ws, &mut rows, stride, &mut trow);
            for ox in 0..nw {
                let (left, ws) = &hws[ox];
                let base = left * CN;
                // CN ∈ {3,4} spends all-but-one lane of the packed tap (CN ≤ 2
                // wastes more lanes than it packs, so it keeps the scalar loop).
                if CN == 3 || CN == 4 {
                    let mut acc = [0f32; 4];
                    // `base + CN*taps + 4 <= stride + 4` (right ≤ w): one slice
                    // check per output pixel; `windows(4)` yields len-4 spans with
                    // no per-tap bounds work. Lane 3 reads a real neighbour and is
                    // never stored → lanes 0..CN compute the old scalar values.
                    // (`+4`, not `+1`: the last window starts at `(taps-1)*CN`.)
                    let win = &trow[base..base + ws.len() * CN + 4];
                    for (px, &wt) in win.windows(4).step_by(CN).zip(ws) {
                        for c in 0..4 {
                            acc[c] += px[c] * wt;
                        }
                    }
                    let packed = clamp_round4(acc);
                    for c in 0..CN {
                        orow[ox * CN + c] = packed[c];
                    }
                } else {
                    let mut acc = [0f32; CN];
                    for k in 0..ws.len() {
                        let wt = ws[k];
                        let j = base + k * CN;
                        for c in 0..CN {
                            acc[c] += trow[j + c] * wt;
                        }
                    }
                    for c in 0..CN {
                        orow[ox * CN + c] = clamp_round(acc[c]);
                    }
                }
            }
        }
    });
    dst
}

/// Same resample specialized for the RGB (`CN = 3`) layout — the common case behind
/// `fit_to_cap` (PNGs/JPEGs decode to Rgb8). Scalar accumulators sidestep the generic
/// `[f32; CN]` indexing; the math (weights, tap order, clamp+round) is identical,
/// verified byte-for-byte against the generic path in `llmtrim-core`'s tests.
pub fn resample3(src: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8> {
    let stride = w * 3;
    let dstride = nw * 3;
    let vws = axis_weights(h, nh);
    let hws = axis_weights(w, nw);
    let use_cache = max_taps(&vws) * stride * 4 <= CACHE_WINDOW_BYTES;
    let threads = nthreads(vertical_taps(&vws, stride), nh);
    let mut dst = vec![0u8; dstride * nh];
    run_bands(&vws, nh, dstride, &mut dst, threads, |dst_band, band_vws| {
        let mut rows = if use_cache {
            Some(F32Rows::new(stride, max_taps(band_vws)))
        } else {
            None
        };
        let mut trow = vec![0f32; stride + 4];
        for (orow, (top, ws)) in dst_band.chunks_mut(dstride).zip(band_vws) {
            vertical_row(src, *top, ws, &mut rows, stride, &mut trow);
            // `hws.len() == nw` and `orow.len() == 3*nw` keep the zip in lockstep;
            // per-pixel writes go to a fixed `[u8; 3]` — no bounds checks.
            for (opx, (left, ws)) in orow.chunks_exact_mut(3).zip(&hws) {
                let base = left * 3;
                let mut acc = [0f32; 4];
                // `base + 3*taps + 1 <= stride + 4` (right ≤ w): one slice check per
                // output pixel, then `windows(4)` yields len-4 spans with no per-tap
                // bounds work; lane 3 reads a real neighbour and is never stored.
                let taps = ws.len();
                let win = &trow[base..base + taps * 3 + 1];
                for (px, &wt) in win.windows(4).step_by(3).zip(ws) {
                    for c in 0..4 {
                        acc[c] += px[c] * wt;
                    }
                }
                let packed = clamp_round4(acc);
                let opx: &mut [u8; 3] = opx.try_into().unwrap();
                opx[0] = packed[0];
                opx[1] = packed[1];
                opx[2] = packed[2];
            }
        }
    });
    dst
}

/// `a.clamp(0.0, 255.0).round() as u8`; accumulators are never NaN (normalized
/// weights sum to 1). The round uses the 1.5×2²³ magic add (round-to-nearest-even
/// on values in [0, 255]): identical to `.round()` except exact-half ties, which
/// land within one level — inside the suite's `max_diff <= 1` tolerance to the
/// library path — while staying a pair of adds instead of an SSE2 `roundf` call.
fn clamp_round(a: f32) -> u8 {
    let a = if a <= 0.0 {
        0.0
    } else if a >= 255.0 {
        255.0
    } else {
        a
    };
    (a + 12582912.0 - 12582912.0) as u8
}

/// `clamp_round` on all four lanes at once: the `min`/`max` clamps lower to packed
/// `minps`/`maxps` and the magic-add round stays packed, so one call replaces ~8
/// scalar instructions per channel with ~8 packed ones for the whole pixel. Lane
/// `CN..4` results are discarded by callers. Same value as `clamp_round` since
/// accumulators are never NaN (normalized weights sum to 1).
fn clamp_round4(a: [f32; 4]) -> [u8; 4] {
    let c = [
        a[0].clamp(0.0, 255.0),
        a[1].clamp(0.0, 255.0),
        a[2].clamp(0.0, 255.0),
        a[3].clamp(0.0, 255.0),
    ];
    [
        (c[0] + 12582912.0 - 12582912.0) as u8,
        (c[1] + 12582912.0 - 12582912.0) as u8,
        (c[2] + 12582912.0 - 12582912.0) as u8,
        (c[3] + 12582912.0 - 12582912.0) as u8,
    ]
}
