# Voice Subtitle Generator

일본어 음성 파일에서 한국어 자막을 자동 생성하는 로컬 기반 도구입니다.

## 주요 기능

- **음성 인식**: faster-whisper 기반 일본어 STT
- **화자 구분**: PyAnnote를 통한 자동 화자 감지
- **번역**: 로컬 LLM 기반 일한 번역
- **화자 매핑**: 화자에 캐릭터 이름 및 색상 지정
- **다양한 포맷**: SRT, ASS, VTT 자막 지원

## 요구사항

- Windows 10/11
- NVIDIA GPU (CUDA, 권장)
- HuggingFace 토큰 (화자 구분 사용 시)

## 설치

1. [Releases](../../releases)에서 최신 버전 다운로드
2. 압축 해제 후 `VoiceSubtitleGenerator.exe` 실행

### HuggingFace 토큰 설정 (화자 구분용)

1. https://huggingface.co/settings/tokens 에서 토큰 생성
2. https://huggingface.co/pyannote/speaker-diarization-3.1 라이선스 동의
3. 앱 설정에서 토큰 입력

## 라이선스

MIT License
