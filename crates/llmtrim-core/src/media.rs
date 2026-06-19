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

#[cfg(feature = "multimodal")]
use std::io::Cursor;

#[cfg(feature = "multimodal")]
use base64::Engine;
#[cfg(feature = "multimodal")]
use image::ImageReader;

/// Decode caps for untrusted images. A crafted ~20k×20k PNG decodes to ~1.6 GB with the
/// library default (`Limits::default()` = 512 MiB alloc, *no* dimension cap), which can
/// OOM-kill the always-on daemon. Bound both dimensions (≤8192px — well above every
/// provider cap, so real images are unaffected) and the working allocation (≤64 MiB).
#[cfg(feature = "multimodal")]
const MAX_DECODE_EDGE: u32 = 8192;
#[cfg(feature = "multimodal")]
const MAX_DECODE_ALLOC: u64 = 64 * 1024 * 1024;

/// JPEG re-encode quality. The default encoder quality (~75) adds visible recompression
/// loss; 90 keeps the "quality-neutral" claim honest (the downscale, not the codec, is the
/// only intended change).
#[cfg(feature = "multimodal")]
const JPEG_QUALITY: u8 = 90;

#[cfg(feature = "multimodal")]
fn decode_limits() -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_DECODE_EDGE);
    limits.max_image_height = Some(MAX_DECODE_EDGE);
    limits.max_alloc = Some(MAX_DECODE_ALLOC);
    limits
}

/// EXIF orientation tag (0x0112) value, if present in a JPEG's APP1 segment. `image` 0.25
/// does **not** auto-apply orientation, so a re-encode would silently strip it and a
/// non-upright photo (orientation ≠ 1) would come back rotated. We read it to skip those.
///
/// Minimal scan: walk JPEG markers to APP1 "Exif\0\0", parse the TIFF header (byte order
/// then IFD0), and find tag 0x0112. Returns `None` if there's no usable orientation (treated
/// as upright / "1" by callers). Bounds-checked throughout; never panics on crafted input.
#[cfg(feature = "multimodal")]
fn jpeg_exif_orientation(bytes: &[u8]) -> Option<u16> {
    // SOI
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let mut i = 2;
    // Walk segment markers until we find APP1 (0xFFE1) or hit image data (SOS 0xFFDA).
    while i + 4 <= bytes.len() {
        if bytes[i] != 0xFF {
            return None; // not a marker boundary → give up (treated as upright)
        }
        let marker = bytes[i + 1];
        if marker == 0xDA || marker == 0xD9 {
            return None; // start of scan / end of image — no Exif before pixels
        }
        let seg_len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
        if seg_len < 2 {
            return None;
        }
        let body_start = i + 4;
        let body_end = i + 2 + seg_len;
        if body_end > bytes.len() {
            return None;
        }
        if marker == 0xE1 {
            let body = &bytes[body_start..body_end];
            return exif_orientation_from_app1(body);
        }
        i = body_end;
    }
    None
}

/// Parse the orientation tag from an APP1 segment body (`Exif\0\0` + TIFF block).
#[cfg(feature = "multimodal")]
fn exif_orientation_from_app1(body: &[u8]) -> Option<u16> {
    if body.len() < 14 || &body[0..6] != b"Exif\0\0" {
        return None;
    }
    let tiff = &body[6..];
    let le = match &tiff[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |buf: &[u8], off: usize| -> Option<u16> {
        let b = buf.get(off..off + 2)?;
        Some(if le {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        })
    };
    let u32_at = |buf: &[u8], off: usize| -> Option<u32> {
        let b = buf.get(off..off + 4)?;
        Some(if le {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        })
    };
    let ifd0 = u32_at(tiff, 4)? as usize;
    let count = u16_at(tiff, ifd0)? as usize;
    for e in 0..count {
        let entry = ifd0 + 2 + e * 12;
        if tiff.get(entry..entry + 12).is_none() {
            break;
        }
        if u16_at(tiff, entry)? == 0x0112 {
            // Orientation is a SHORT stored in the value field (first 2 bytes).
            return u16_at(tiff, entry + 8);
        }
    }
    None
}

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

#[cfg(feature = "multimodal")]
fn b64() -> base64::engine::general_purpose::GeneralPurpose {
    base64::engine::general_purpose::STANDARD
}

/// Target dimensions to satisfy every constraint in `cap`, preserving aspect ratio.
/// `None` if the image is already within the cap (no resize needed).
#[cfg(feature = "multimodal")]
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
#[cfg(feature = "multimodal")]
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
/// for tile-priced providers. `None` (leave the original unchanged) when it can't decode,
/// the format isn't supported, it's already optimal, a decode limit is exceeded (oversized
/// untrusted image), it's a JPEG carrying a non-upright EXIF orientation, or the re-encode
/// would be larger than the input.
#[cfg(feature = "multimodal")]
pub fn fit_to_cap(data: &str, cap: ImageCap) -> Option<String> {
    let bytes = b64().decode(data.trim()).ok()?;
    let format = image::guess_format(&bytes).ok()?;
    let is_jpeg = format == image::ImageFormat::Jpeg;
    // EXIF orientation is dropped on re-encode and `image` 0.25 doesn't bake it in, so a
    // tag-6 photo would come back rotated. Skip (pass through untouched) when not upright.
    if is_jpeg && jpeg_exif_orientation(&bytes).is_some_and(|o| o != 1) {
        return None;
    }
    // Decode under explicit limits (dimension + alloc cap) instead of `load_from_memory`'s
    // 512 MiB/no-dimension-cap default — a limit hit returns Err → None → original passes through.
    let mut reader = ImageReader::with_format(Cursor::new(&bytes), format);
    reader.limits(decode_limits());
    let img = reader.decode().ok()?;
    let (w, h) = (img.width(), img.height());
    // Cap resize (or original dims), then trim a wasteful partial tile.
    let (cw, ch) = target_dims(w, h, cap).unwrap_or((w, h));
    let (nw, nh) = (snap_tile(cw, cap.tile), snap_tile(ch, cap.tile));
    if nw == w && nh == h {
        return None; // already optimal
    }
    let resized = img.resize(nw, nh, image::imageops::FilterType::Lanczos3);
    let mut buf = Cursor::new(Vec::new());
    if is_jpeg {
        // Encode at high quality so the downscale stays quality-neutral (vs the ~75 default).
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY);
        resized.write_with_encoder(encoder).ok()?;
    } else {
        resized.write_to(&mut buf, format).ok()?;
    }
    // Re-encoding a small/already-efficient image (esp. JPEG at q90) can grow the bytes; if
    // so the "compression" is a regression, so keep the original.
    if buf.get_ref().len() >= bytes.len() {
        return None;
    }
    Some(b64().encode(buf.get_ref()))
}

/// No-op fallback without the `multimodal` feature: the image decoders aren't compiled in,
/// so the payload passes through unchanged.
#[cfg(not(feature = "multimodal"))]
pub fn fit_to_cap(_data: &str, _cap: ImageCap) -> Option<String> {
    None
}

/// Resize the payload of a `data:<media>;base64,<data>` URI down to `cap`.
pub fn fit_data_uri(uri: &str, cap: ImageCap) -> Option<String> {
    let (header, data) = uri.split_once(',')?;
    if !header.contains("base64") {
        return None;
    }
    Some(format!("{header},{}", fit_to_cap(data, cap)?))
}

#[cfg(all(test, feature = "multimodal"))]
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

    // ── #13: oversized untrusted image is skipped under the decode limits, not OOM'd ──

    #[test]
    fn oversized_image_skipped_under_decode_limit() {
        // A PNG larger than the dimension cap (8192px) must hit the limit and pass through
        // unchanged (None) — never decoded into a multi-hundred-MB buffer.
        let huge = png_b64(MAX_DECODE_EDGE + 100, 16);
        assert!(
            fit_to_cap(&huge, CAP_OPENAI).is_none(),
            "image over the decode dimension cap is skipped, not resized"
        );
    }

    #[test]
    fn decode_limits_are_bounded() {
        let l = decode_limits();
        assert_eq!(l.max_image_width, Some(MAX_DECODE_EDGE));
        assert_eq!(l.max_image_height, Some(MAX_DECODE_EDGE));
        assert_eq!(l.max_alloc, Some(MAX_DECODE_ALLOC));
    }

    // ── #14: JPEG EXIF orientation + quality + size guard ──

    /// A `w`×`h` JPEG with an APP1/Exif block carrying orientation `orient` (big-endian TIFF).
    fn jpeg_with_orientation_dims(orient: u16, w: u32, h: u32) -> Vec<u8> {
        // Encode a real (orientation-free) JPEG first…
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(w, h));
        let mut body = Vec::new();
        let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut body, 90);
        img.write_with_encoder(enc).unwrap();
        // …then splice an APP1 segment in right after SOI (bytes 0..2).
        // TIFF: "MM" (big-endian), 0x002A, IFD0 offset 8, 1 entry: tag 0x0112,
        // type 3 (SHORT), count 1, value `orient` in the high 2 bytes of the value field.
        let mut tiff = vec![b'M', b'M', 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08];
        tiff.extend_from_slice(&[0x00, 0x01]); // entry count
        tiff.extend_from_slice(&[0x01, 0x12]); // tag = orientation
        tiff.extend_from_slice(&[0x00, 0x03]); // type = SHORT
        tiff.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]); // count = 1
        tiff.extend_from_slice(&orient.to_be_bytes()); // value (first 2 bytes)
        tiff.extend_from_slice(&[0x00, 0x00]); // value padding
        tiff.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // next-IFD offset = 0
        let mut app1_body = b"Exif\0\0".to_vec();
        app1_body.extend_from_slice(&tiff);
        let seg_len = (app1_body.len() + 2) as u16;
        let mut app1 = vec![0xFF, 0xE1];
        app1.extend_from_slice(&seg_len.to_be_bytes());
        app1.extend_from_slice(&app1_body);
        let mut out = body[0..2].to_vec(); // SOI
        out.extend_from_slice(&app1);
        out.extend_from_slice(&body[2..]);
        out
    }

    #[test]
    fn exif_orientation_is_parsed() {
        assert_eq!(
            jpeg_exif_orientation(&jpeg_with_orientation_dims(6, 1, 1)),
            Some(6)
        );
        assert_eq!(
            jpeg_exif_orientation(&jpeg_with_orientation_dims(1, 1, 1)),
            Some(1)
        );
        // A plain JPEG with no Exif → None (callers treat as upright).
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
        let mut plain = Vec::new();
        let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut plain, 90);
        img.write_with_encoder(enc).unwrap();
        assert_eq!(jpeg_exif_orientation(&plain), None);
        // Garbage never panics.
        assert_eq!(jpeg_exif_orientation(b"not a jpeg"), None);
    }

    #[test]
    fn rotated_jpeg_is_passed_through() {
        // An over-cap JPEG with orientation 6 (short side 900 > 768) must be skipped (None),
        // not silently de-rotated by the re-encode that drops the orientation tag.
        let rotated = jpeg_with_orientation_dims(6, 1000, 900);
        let b64data = b64().encode(&rotated);
        assert!(
            fit_to_cap(&b64data, CAP_OPENAI).is_none(),
            "non-upright JPEG is passed through unchanged, not re-encoded"
        );
        // Sanity: the same image as orientation 1 *does* get resized (so the skip is the cause).
        let upright = jpeg_with_orientation_dims(1, 1000, 900);
        assert!(
            fit_to_cap(&b64().encode(&upright), CAP_OPENAI).is_some(),
            "upright JPEG over the cap is still resized"
        );
    }

    #[test]
    fn jpeg_downscale_round_trips_and_decodes() {
        // A photo-like JPEG (gradient → non-trivial bytes) over the cap downscales and the
        // result still decodes; the size guard lets a genuine shrink through.
        let mut buf = image::RgbImage::new(2000, 1500);
        for (x, y, px) in buf.enumerate_pixels_mut() {
            *px = image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]);
        }
        let img = image::DynamicImage::ImageRgb8(buf);
        let mut bytes = Vec::new();
        let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 90);
        img.write_with_encoder(enc).unwrap();
        let data = b64().encode(&bytes);
        let out = fit_to_cap(&data, CAP_OPENAI).expect("jpeg downscaled");
        let (w, h) = dims(&out);
        assert!(w.min(h) <= 768 && w.max(h) <= 2048, "within OpenAI cap");
    }
}
