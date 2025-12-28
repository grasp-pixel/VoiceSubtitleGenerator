"""Font management for subtitle generation."""

import logging
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


@dataclass
class FontInfo:
    """Information about a recommended font."""

    name: str  # Font family name for ASS
    display_name: str  # Display name in UI
    filename: str  # Expected filename pattern
    download_url: str  # Download URL (zip or individual files)
    description: str
    is_zip: bool = True  # True if download_url is a zip file
    extra_files: list[str] | None = None  # Additional TTF file URLs if not zip


# GitHub에서 직접 다운로드 가능한 CJK 폰트
# GitHub releases/raw 사용 (직접 다운로드 지원)
_GITHUB_RAW = "https://raw.githubusercontent.com/google/fonts/main/ofl"

RECOMMENDED_FONTS: list[FontInfo] = [
    FontInfo(
        name="Noto Sans KR",
        display_name="Noto Sans KR (권장)",
        filename="NotoSansKR",
        download_url="https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/17_NotoSansKR.zip",
        description="Google 무료 한글 폰트, 깔끔한 고딕체",
    ),
    FontInfo(
        name="Noto Serif KR",
        display_name="Noto Serif KR",
        filename="NotoSerifKR",
        download_url="https://github.com/notofonts/noto-cjk/releases/download/Serif2.003/13_NotoSerifKR.zip",
        description="Google 무료 한글 명조체",
    ),
    FontInfo(
        name="Noto Sans JP",
        display_name="Noto Sans JP (일본어)",
        filename="NotoSansJP",
        download_url="https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/16_NotoSansJP.zip",
        description="Google 무료 일본어 고딕체",
    ),
    FontInfo(
        name="Nanum Gothic",
        display_name="나눔고딕",
        filename="NanumGothic",
        download_url=f"{_GITHUB_RAW}/nanumgothic/NanumGothic-Regular.ttf",
        description="네이버 무료 한글 고딕체",
        is_zip=False,
        extra_files=[
            f"{_GITHUB_RAW}/nanumgothic/NanumGothic-Bold.ttf",
            f"{_GITHUB_RAW}/nanumgothic/NanumGothic-ExtraBold.ttf",
        ],
    ),
    FontInfo(
        name="Nanum Myeongjo",
        display_name="나눔명조",
        filename="NanumMyeongjo",
        download_url=f"{_GITHUB_RAW}/nanummyeongjo/NanumMyeongjo-Regular.ttf",
        description="네이버 무료 한글 명조체",
        is_zip=False,
        extra_files=[
            f"{_GITHUB_RAW}/nanummyeongjo/NanumMyeongjo-Bold.ttf",
            f"{_GITHUB_RAW}/nanummyeongjo/NanumMyeongjo-ExtraBold.ttf",
        ],
    ),
    FontInfo(
        name="Malgun Gothic",
        display_name="맑은 고딕 (시스템)",
        filename="malgun",
        download_url="",  # Windows built-in
        description="Windows 기본 한글 폰트",
    ),
    FontInfo(
        name="D2Coding",
        display_name="D2Coding (개발용)",
        filename="D2Coding",
        download_url="https://github.com/naver/d2codingfont/releases/download/VER1.3.2/D2Coding-Ver1.3.2-20180524.zip",
        description="네이버 개발용 고정폭 폰트",
    ),
    FontInfo(
        name="Pretendard",
        display_name="Pretendard",
        filename="Pretendard",
        download_url="https://github.com/orioncactus/pretendard/releases/download/v1.3.9/Pretendard-1.3.9.zip",
        description="Apple SD Gothic Neo 대체, 깔끔한 고딕체",
    ),
]


class FontManager:
    """Manages font installation and detection."""

    # Common font directories
    WINDOWS_FONTS_DIR = Path("C:/Windows/Fonts")
    USER_FONTS_DIR = Path.home() / "AppData/Local/Microsoft/Windows/Fonts"
    PROJECT_FONTS_DIR = Path(__file__).parent.parent / "fonts"

    def __init__(self):
        """Initialize font manager."""
        self.fonts_dir = self.PROJECT_FONTS_DIR
        self.fonts_dir.mkdir(parents=True, exist_ok=True)

    def get_recommended_fonts(self) -> list[FontInfo]:
        """Get list of recommended fonts."""
        return RECOMMENDED_FONTS.copy()

    def get_font_names(self) -> list[str]:
        """Get list of font display names for UI."""
        return [f.display_name for f in RECOMMENDED_FONTS]

    def get_font_info(self, display_name: str) -> FontInfo | None:
        """Get FontInfo by display name."""
        for font in RECOMMENDED_FONTS:
            if font.display_name == display_name:
                return font
        return None

    def get_font_family_name(self, display_name: str) -> str:
        """Get font family name from display name."""
        for font in RECOMMENDED_FONTS:
            if font.display_name == display_name:
                return font.name
        return display_name  # Fallback to display name

    def get_display_name(self, family_name: str) -> str:
        """Get display name from font family name."""
        for font in RECOMMENDED_FONTS:
            if font.name == family_name:
                return font.display_name
        return family_name

    def is_font_installed(self, font: FontInfo) -> bool:
        """Check if a font is installed on the system."""
        # Check various locations for font files
        search_dirs = [
            self.WINDOWS_FONTS_DIR,
            self.USER_FONTS_DIR,
            self.fonts_dir,
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            # Look for files matching the font filename pattern
            for ext in ["*.ttf", "*.otf", "*.ttc"]:
                for file in search_dir.glob(ext):
                    if font.filename.lower() in file.name.lower():
                        return True

        return False

    def get_font_status(self, display_name: str) -> tuple[bool, str]:
        """
        Get installation status for a font.

        Returns:
            tuple: (is_installed, status_message)
        """
        font = self.get_font_info(display_name)
        if font is None:
            return False, "알 수 없음"

        installed = self.is_font_installed(font)
        if installed:
            return True, "설치됨"
        else:
            return False, "미설치"

    def download_font(
        self,
        display_name: str,
        progress_callback: callable = None,
    ) -> Path | None:
        """
        Download font from Google Fonts.

        Args:
            display_name: Font display name.
            progress_callback: Optional callback for progress updates.

        Returns:
            Path to downloaded zip file, or None if failed.
        """
        font = self.get_font_info(display_name)
        if font is None:
            logger.error(f"Font not found: {display_name}")
            return None

        try:
            logger.info(f"Downloading font: {font.name}")
            if progress_callback:
                progress_callback("다운로드 중...")

            # Add headers to mimic browser request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/zip, application/octet-stream, */*",
            }

            response = requests.get(
                font.download_url,
                headers=headers,
                stream=True,
                timeout=60,
                allow_redirects=True,
            )
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "")
            logger.debug(f"Content-Type: {content_type}")

            # Save to temp file
            temp_dir = Path(tempfile.gettempdir()) / "voice_subtitle_fonts"
            temp_dir.mkdir(parents=True, exist_ok=True)
            zip_path = temp_dir / f"{font.filename}.zip"

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify it's a valid zip file
            if not zipfile.is_zipfile(zip_path):
                # Log first bytes for debugging
                with open(zip_path, "rb") as f:
                    first_bytes = f.read(100)
                logger.error(f"Downloaded file is not a zip. First bytes: {first_bytes[:50]}")
                zip_path.unlink(missing_ok=True)
                return None

            logger.info(f"Downloaded to: {zip_path}")
            return zip_path

        except Exception as e:
            logger.error(f"Failed to download font: {e}")
            return None

    def _download_file(self, url: str, dest_path: Path) -> bool:
        """Download a single file."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(response.content)
            return True
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False

    def install_font(
        self,
        display_name: str,
        progress_callback: callable = None,
    ) -> bool:
        """
        Download and install font.

        Args:
            display_name: Font display name.
            progress_callback: Optional callback for progress updates.

        Returns:
            bool: True if installed successfully.
        """
        font = self.get_font_info(display_name)
        if font is None:
            return False

        # Check if already installed
        if self.is_font_installed(font):
            logger.info(f"Font already installed: {font.name}")
            if progress_callback:
                progress_callback("이미 설치됨")
            return True

        # System fonts have no download URL
        if not font.download_url:
            logger.info(f"System font, no download needed: {font.name}")
            if progress_callback:
                progress_callback("시스템 폰트")
            return True

        self.USER_FONTS_DIR.mkdir(parents=True, exist_ok=True)

        # Handle individual TTF files (not zip)
        if not font.is_zip:
            if progress_callback:
                progress_callback("다운로드 중...")

            # Collect all URLs to download
            urls = [font.download_url]
            if font.extra_files:
                urls.extend(font.extra_files)

            installed_count = 0
            for url in urls:
                filename = url.split("/")[-1]
                dest = self.USER_FONTS_DIR / filename
                if self._download_file(url, dest):
                    installed_count += 1
                    logger.info(f"Installed: {filename}")

            if installed_count > 0:
                logger.info(f"Installed {installed_count} font files")
                if progress_callback:
                    progress_callback("설치 완료!")
                return True
            else:
                if progress_callback:
                    progress_callback("다운로드 실패")
                return False

        # Handle zip files
        if progress_callback:
            progress_callback("다운로드 중...")

        zip_path = self.download_font(display_name, progress_callback)
        if zip_path is None:
            if progress_callback:
                progress_callback("다운로드 실패")
            return False

        try:
            if progress_callback:
                progress_callback("설치 중...")

            # Extract zip
            extract_dir = zip_path.parent / font.filename
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            # Find font files and copy to user fonts directory
            installed_count = 0

            for font_file in extract_dir.rglob("*"):
                if font_file.suffix.lower() in [".ttf", ".otf"]:
                    dest = self.USER_FONTS_DIR / font_file.name
                    shutil.copy2(font_file, dest)
                    installed_count += 1
                    logger.info(f"Installed: {font_file.name}")

            # Cleanup
            shutil.rmtree(extract_dir, ignore_errors=True)
            zip_path.unlink(missing_ok=True)

            if installed_count > 0:
                logger.info(f"Installed {installed_count} font files")
                if progress_callback:
                    progress_callback("설치 완료!")
                return True
            else:
                logger.warning("No font files found in archive")
                if progress_callback:
                    progress_callback("폰트 파일 없음")
                return False

        except Exception as e:
            logger.error(f"Failed to install font: {e}")
            if progress_callback:
                progress_callback("설치 실패")
            return False

    def open_fonts_folder(self) -> None:
        """Open Windows Fonts folder."""
        subprocess.run(["explorer", str(self.USER_FONTS_DIR)], check=False)


# Global instance
_font_manager: FontManager | None = None


def get_font_manager() -> FontManager:
    """Get global font manager instance."""
    global _font_manager
    if _font_manager is None:
        _font_manager = FontManager()
    return _font_manager
