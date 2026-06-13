//! Thin entry point for `uniffi-bindgen` so bindings can be generated with
//! `cargo run --bin uniffi-bindgen -- generate --library <cdylib> --language python`.
fn main() {
    uniffi::uniffi_bindgen_main()
}
