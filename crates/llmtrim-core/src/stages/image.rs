//! Stage H — multimodal: lower image detail tier + downscale embedded images. Opt-in.
//!
//! Downscaling resizes embedded base64 images to the provider's effective resolution
//! cap (see [`crate::media`]) — quality-neutral, since the provider downscales to the
//! same size anyway, while cutting upload bytes and (for pixel-priced providers)
//! tokens. The detail tier is *optional*: `None` leaves the caller's choice;
//! `Some("low")` opts into OpenAI's flat-85-token tier (lossy — its own trade-off).
//!
//! Image savings are provider-image-side, not in the text-token measure, so this uses
//! the `Structural` gate: the input-token gate sees no text change and would always
//! revert image work, so it can't guard this stage. Downscaling is quality-neutral; the
//! `Some("low")` detail tier is genuinely lossy, but it is an explicit opt-in, so it
//! ships unconditionally — the gate intentionally does not second-guess that choice.

use anyhow::Result;

use crate::gate::{GateKind, PlanEntry, Transform};
use crate::ir::Request;
use crate::provider::Provider;

pub struct ImageStage {
    /// Detail tier to request (OpenAI only). `None` leaves the caller's choice;
    /// `Some("low")` forces the flat-85-token tier (lossy).
    pub detail: Option<String>,
}

impl Transform for ImageStage {
    fn name(&self) -> &str {
        "image"
    }

    fn gate_kind(&self) -> GateKind {
        GateKind::Structural
    }

    fn apply(
        &self,
        req: &mut Request,
        provider: &dyn Provider,
        _plan: &mut Vec<PlanEntry>,
    ) -> Result<()> {
        if let Some(tier) = &self.detail {
            provider.set_image_detail(req, tier);
        }
        provider.downscale_images(req);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ProviderKind;
    use crate::pipeline;
    use crate::provider::OpenAiProvider;
    use crate::tokenizer::counter_for;
    use base64::Engine;
    use serde_json::{Value, json};

    fn png_data_uri(size: u32) -> String {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(size, size));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let b = base64::engine::general_purpose::STANDARD.encode(buf.get_ref());
        format!("data:image/png;base64,{b}")
    }

    #[test]
    fn sets_detail_and_downscales_to_cap() {
        // 1000×1000 exceeds OpenAI's 768 short-side cap, so it gets resized.
        let body = json!({
            "model":"gpt-4o",
            "messages":[{"role":"user","content":[
                {"type":"text","text":"describe this"},
                {"type":"image_url","image_url":{"url": png_data_uri(1000)}}
            ]}]
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(ImageStage {
            detail: Some("low".to_string()),
        })];

        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied, "Structural stage always applies");

        assert_eq!(
            req.raw()
                .pointer("/messages/0/content/1/image_url/detail")
                .and_then(Value::as_str),
            Some("low")
        );
        let uri = req
            .raw()
            .pointer("/messages/0/content/1/image_url/url")
            .and_then(Value::as_str)
            .unwrap();
        let data = uri.split_once(',').unwrap().1;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        assert!(
            img.width().min(img.height()) <= 768,
            "resized to OpenAI cap"
        );
    }
}
