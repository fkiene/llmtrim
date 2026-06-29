#!/usr/bin/env python3
"""Generate the tray + app-bundle icons from the master SVGs.

Sources (hand-edited, committed):
  icons/tray.svg      -> the menu-bar glyph (currentColor line art)
  icons/app-icon.svg  -> the green brand square (app / installer / dock icon)

Outputs (committed, so a build never needs Python):
  icons/tray-mono.png        black glyph on transparent  -> macOS template image
  icons/tray-color.png       green glyph on transparent   -> Windows taskbar
  icons/32x32.png            app icon, small PNG          | referenced by
  icons/128x128.png          app icon                     | tauri.conf.json
  icons/128x128@2x.png       app icon @2x (256px)         | bundle.icon
  icons/icon.ico             Windows app icon (multi-size)|
  icons/icon.icns            macOS app icon (multi-size)  |

Re-run after editing a master SVG:  python3 tools/gen_icons.py
Deps: cairosvg, Pillow.
"""

from io import BytesIO
from pathlib import Path

import cairosvg
from PIL import Image

ICONS = Path(__file__).resolve().parent.parent / "icons"

# The menu-bar glyph, padded inside a 28-unit box so it doesn't touch the edges,
# and re-coloured (the master uses currentColor, which rasterises to black).
TRAY_GLYPH = """\
<line x1="12" y1="4" x2="12" y2="20"/>
<path d="M3 12 H9"/><path d="M6 9 L9 12 L6 15"/>
<path d="M21 12 H15"/><path d="M18 9 L15 12 L18 15"/>"""


def tray_svg(stroke: str) -> bytes:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" fill="none" '
        f'stroke="{stroke}" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round"><g transform="translate(2,2)">'
        f"{TRAY_GLYPH}</g></svg>"
    ).encode()


def render(svg: bytes, px: int) -> Image.Image:
    png = cairosvg.svg2png(bytestring=svg, output_width=px, output_height=px)
    return Image.open(BytesIO(png)).convert("RGBA")


def assert_template(img: Image.Image) -> Image.Image:
    """A macOS template image must be black + alpha only (the system tints it);
    any colour would render wrong in the menu bar."""
    colours = {px[:3] for _, px in img.getcolors(maxcolors=1 << 24) if px[3] > 0}
    if colours - {(0, 0, 0)}:
        raise SystemExit(f"tray-mono.png is not a black template image: {colours}")
    return img


def main() -> None:
    # Tray icons: render at a high resolution so retina menu bars stay crisp.
    assert_template(render(tray_svg("#000000"), 64)).save(ICONS / "tray-mono.png")
    render(tray_svg("#34e0a1"), 64).save(ICONS / "tray-color.png")

    # App icon master, rendered once at 1024 and downscaled for every target.
    app = render((ICONS / "app-icon.svg").read_bytes(), 1024)

    def scaled(px: int) -> Image.Image:
        return app.resize((px, px), Image.LANCZOS)

    scaled(32).save(ICONS / "32x32.png")
    scaled(128).save(ICONS / "128x128.png")
    scaled(256).save(ICONS / "128x128@2x.png")

    ico_sizes = [16, 24, 32, 48, 64, 128, 256]
    scaled(256).save(ICONS / "icon.ico", sizes=[(s, s) for s in ico_sizes])

    # Pillow's ICNS encoder only knows slots up to 512 (it silently drops 1024),
    # so cap the list there. The 1024 source still feeds crisp downscales.
    icns_sizes = [16, 32, 64, 128, 256, 512]
    app.save(ICONS / "icon.icns", sizes=[(s, s) for s in icns_sizes])

    print(f"wrote icons to {ICONS}")


if __name__ == "__main__":
    main()
