//! Image downscaling for Stage H (multimodal).
//!
//! Resize images down to each provider's **effective resolution cap** — the size the
//! model actually uses — so this is *quality-neutral*: the provider would downscale
//! to the same dimensions anyway, so the model sees no difference, while we cut
//! upload bytes and (for pixel-priced providers) tokens. We never upscale or shrink
//! below the cap (that *would* lose quality). Format is preserved (PNG stays
//! lossless); only png/jpeg are built in. Lanczos3 for high-quality downscaling.
//!
//! Caps (from provider docs, verified 2026):
//! - OpenAI: fit within 2048×2048, shortest side ≤768 (its high-detail resize).
//!   <https://platform.openai.com/docs/guides/images-vision>
//! - Anthropic: long edge ≤1568 and ≤~1.15 MP.
//!   <https://docs.claude.com/en/docs/build-with-claude/vision>

use base64::Engine;

/// A provider's effective image resolution cap. Resizing to it is quality-neutral.
#[derive(Clone, Copy)]
pub struct ImageCap {
    pub max_long: u32,
    pub max_short: u32,
    pub max_pixels: u64,
    /// Tile edge for tile-priced providers (OpenAI 512px tiles); 0 = not tile-priced.
    pub tile: u32,
}

pub const CAP_OPENAI: ImageCap = ImageCap {
    max_long: 2048,
    max_short: 768,
    max_pixels: u64::MAX,
    tile: 512,
};

pub const CAP_ANTHROPIC: ImageCap = ImageCap {
    max_long: 1568,
    max_short: u32::MAX,
    max_pixels: 1_150_000,
    tile: 0,
};

// Gemini downsamples large images to a ~3072px long edge before processing, so capping
// there is quality-neutral. (Conservative: if the real cap is larger we just downscale
// less; never below it.)
pub const CAP_GOOGLE: ImageCap = ImageCap {
    max_long: 3072,
    max_short: u32::MAX,
    max_pixels: u64::MAX,
    tile: 0,
};

fn b64() -> base64::engine::general_purpose::GeneralPurpose {
    base64::engine::general_purpose::STANDARD
}

/// Target dimensions to satisfy every constraint in `cap`, preserving aspect ratio.
/// `None` if the image is already within the cap (no resize needed).
fn target_dims(w: u32, h: u32, cap: ImageCap) -> Option<(u32, u32)> {
    let (wf, hf) = (f64::from(w), f64::from(h));
    let long = wf.max(hf);
    let short = wf.min(hf);
    let mut scale = 1.0_f64;
    if long > f64::from(cap.max_long) {
        scale = scale.min(f64::from(cap.max_long) / long);
    }
    if short > f64::from(cap.max_short) {
        scale = scale.min(f64::from(cap.max_short) / short);
    }
    let pixels = wf * hf;
    if pixels > cap.max_pixels as f64 {
        scale = scale.min((cap.max_pixels as f64 / pixels).sqrt());
    }
    if scale >= 1.0 {
        return None;
    }
    let nw = (wf * scale).floor().max(1.0) as u32;
    let nh = (hf * scale).floor().max(1.0) as u32;
    Some((nw, nh))
}

/// Snap a dimension DOWN to a tile multiple, but only to shave a *barely-filled*
/// partial tile (remainder < 10% of a tile, ≤~51px) — saving a whole tile's tokens
/// (OpenAI: 170/tile) for a negligible (<10%, one-axis) downscale. Never below one tile.
fn snap_tile(dim: u32, tile: u32) -> u32 {
    if tile == 0 || dim <= tile {
        return dim;
    }
    let rem = dim % tile;
    if rem != 0 && rem < tile / 10 {
        dim - rem
    } else {
        dim
    }
}

/// Resize a base64 image down to `cap` (preserving format + aspect), then tile-snap
/// for tile-priced providers. `None` if it can't decode, the format isn't supported,
/// or it's already optimal.
pub fn fit_to_cap(data: &str, cap: ImageCap) -> Option<String> {
    let bytes = b64().decode(data.trim()).ok()?;
    let format = image::guess_format(&bytes).ok()?;
    let img = image::load_from_memory(&bytes).ok()?;
    let (w, h) = (img.width(), img.height());
    // Cap resize (or original dims), then trim a wasteful partial tile.
    let (cw, ch) = target_dims(w, h, cap).unwrap_or((w, h));
    let (nw, nh) = (snap_tile(cw, cap.tile), snap_tile(ch, cap.tile));
    if nw == w && nh == h {
        return None; // already optimal
    }
    let resized = img.resize(nw, nh, image::imageops::FilterType::Lanczos3);
    let mut buf = std::io::Cursor::new(Vec::new());
    resized.write_to(&mut buf, format).ok()?;
    Some(b64().encode(buf.get_ref()))
}

/// Resize the payload of a `data:<media>;base64,<data>` URI down to `cap`.
pub fn fit_data_uri(uri: &str, cap: ImageCap) -> Option<String> {
    let (header, data) = uri.split_once(',')?;
    if !header.contains("base64") {
        return None;
    }
    Some(format!("{header},{}", fit_to_cap(data, cap)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_b64(w: u32, h: u32) -> String {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(w, h));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        b64().encode(buf.get_ref())
    }

    fn dims(b64data: &str) -> (u32, u32) {
        let bytes = b64().decode(b64data).unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        (img.width(), img.height())
    }

    #[test]
    fn openai_caps_short_side_to_768() {
        let big = png_b64(1000, 900); // short side 900 > 768
        let out = fit_to_cap(&big, CAP_OPENAI).expect("resized");
        let (w, h) = dims(&out);
        assert!(
            w.min(h) <= 768,
            "short side capped at 768 (got {}x{})",
            w,
            h
        );
        assert!(w.max(h) <= 2048);
    }

    #[test]
    fn anthropic_caps_megapixels() {
        let big = png_b64(1200, 1100); // 1.32 MP > 1.15 MP
        let out = fit_to_cap(&big, CAP_ANTHROPIC).expect("resized");
        let (w, h) = dims(&out);
        assert!(u64::from(w) * u64::from(h) <= 1_150_000, "within 1.15 MP");
        assert!(w.max(h) <= 1568);
    }

    #[test]
    fn within_cap_is_untouched() {
        let small = png_b64(640, 480); // within both caps
        assert!(fit_to_cap(&small, CAP_OPENAI).is_none());
        assert!(fit_to_cap(&small, CAP_ANTHROPIC).is_none());
    }

    #[test]
    fn data_uri_preserves_header() {
        let uri = format!("data:image/png;base64,{}", png_b64(1000, 1000));
        let out = fit_data_uri(&uri, CAP_OPENAI).expect("resized");
        assert!(out.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn non_image_is_skipped() {
        let txt = b64().encode(b"not an image at all");
        assert!(fit_to_cap(&txt, CAP_OPENAI).is_none());
    }

    #[test]
    fn openai_tile_snap_trims_sliver_tile() {
        // 1025 wide = 1px over a 512 tile boundary → snap to 1024, saving a tile
        // column (within caps otherwise, so only the snap fires).
        let img = png_b64(1025, 700);
        let out = fit_to_cap(&img, CAP_OPENAI).expect("tile-snapped");
        let (w, _h) = dims(&out);
        assert!(w <= 1024, "snapped under the 512 tile boundary (got {w})");
    }

    #[test]
    fn tile_snap_leaves_well_filled_tiles_alone() {
        // 640 wide: remainder 128 > 51 (10% of 512) → not a sliver → untouched.
        assert!(fit_to_cap(&png_b64(640, 480), CAP_OPENAI).is_none());
    }
}
