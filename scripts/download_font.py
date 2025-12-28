"""Download Noto Sans JP font for CJK support."""

import urllib.request
import ssl
from pathlib import Path

# Direct TTF link from GitHub notofonts releases
FONT_URL = "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
FONT_DIR = Path(__file__).parent.parent / "fonts"
FONT_FILE = "NotoSansCJKjp-Regular.otf"
FONT_PATH = FONT_DIR / FONT_FILE


def download_font():
    """Download Noto Sans CJK JP font."""
    FONT_DIR.mkdir(exist_ok=True)

    if FONT_PATH.exists():
        print(f"Font already exists: {FONT_PATH}")
        return True

    print("Downloading Noto Sans CJK JP (~16MB)...")
    try:
        # Create SSL context that doesn't verify (for corporate proxies)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            FONT_URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(req, context=ctx) as response:
            data = response.read()
            FONT_PATH.write_bytes(data)

        print(f"Downloaded: {FONT_PATH} ({FONT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    except Exception as e:
        print(f"Failed to download: {e}")
        print("\nManual download:")
        print("1. Go to https://github.com/notofonts/noto-cjk/tree/main/Sans/OTF/Japanese")
        print("2. Download NotoSansCJKjp-Regular.otf")
        print(f"3. Place it in: {FONT_DIR}")
        return False


if __name__ == "__main__":
    download_font()
