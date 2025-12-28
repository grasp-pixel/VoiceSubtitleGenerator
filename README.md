# Voice Subtitle Generator

일본어 음성 파일에서 한국어 자막을 자동 생성하는 로컬 기반 도구입니다.

## 주요 기능

- **음성 인식**: faster-whisper 기반 일본어 STT
- **번역**: 로컬 LLM(Qwen3) 기반 일한 번역
- **다양한 포맷**: SRT, ASS, VTT 자막 지원
- **GUI**: Dear PyGui 기반 데스크톱 앱

## 요구사항

- Python 3.11+
- NVIDIA GPU (CUDA, 권장)
- macOS / Windows / Linux

## 설치

```bash
# uv 사용 시 (권장)
uv sync

# pip 사용 시
pip install -e .
```

## 실행

```bash
# GUI 모드 (기본)
python main.py

# CLI 모드
python main.py process input.mp3 -o output/
```

## 라이선스

MIT License
