# Voice Subtitle Generator

일본어 음성 파일에서 한국어 자막을 자동 생성하는 로컬 기반 도구입니다.

## 주요 기능

- **음성 인식**: faster-whisper 기반 일본어 STT
- **번역**: 로컬 LLM(Qwen) 기반 일한 번역
- **다양한 포맷**: SRT, ASS, VTT 자막 지원
- **GUI**: Dear PyGui 기반 데스크톱 앱

## 요구사항

### 기본
- Python 3.12 ~ 3.13
- 16GB+ RAM

### GPU 가속 (권장)
- NVIDIA GPU (RTX 20 시리즈 이상 권장)
- CUDA Toolkit 12.x
- 8GB+ VRAM (번역 모델 크기에 따라 다름)

## 사전 설치 (Windows)

### 1. Visual Studio Build Tools 2022

GPU 가속을 위한 llama-cpp-python 빌드에 필요합니다. (CUDA 12.x는 VS 2022까지만 지원)

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --passive"
```

또는 [Visual Studio Installer](https://visualstudio.microsoft.com/downloads/)에서:
- Visual Studio Build Tools 2022 설치
- **"C++를 사용한 데스크톱 개발"** 워크로드 선택

### 2. Ninja (빌드 도구)

```powershell
winget install Ninja-build.Ninja
```

### 3. CUDA Toolkit 12.x

[NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)에서 다운로드.

설치 시 **"Visual Studio Integration"** 체크 필수.

## 설치 및 실행

### 배포 버전 (권장)

1. `VoiceSubtitleGenerator_vX.X.X.zip` 압축 해제
2. `models/` 폴더에 GGUF 모델 파일 배치
3. `run.bat` 더블클릭

첫 실행 시:
- Python 환경 자동 구성 (uv 사용)
- 종속성 설치 (10-20분)
- GPU 감지 시 CUDA 버전 llama-cpp-python 자동 빌드

### 개발 환경

```bash
# uv 사용 시 (권장)
uv sync

# pip 사용 시
pip install -e .

# 실행
python main.py
```

## 권장 모델

| 용도 | 모델 | VRAM |
|------|------|------|
| STT | faster-whisper large-v3-turbo | ~4GB |
| 번역 | Qwen3-8B-Q4_K_M | ~6GB |

모델은 `models/` 폴더에 배치하세요.

## CLI 사용

```bash
# 단일 파일 처리
python main.py process input.mp3 -o output/

# 폴더 일괄 처리
python main.py process ./audio_folder/ -f ass -o output/
```

## 문제 해결

### "번역: CPU 모드" 표시
llama-cpp-python이 GPU 지원 없이 설치됨. `build_cuda.bat` 실행하여 재빌드.

### CUDA 빌드 실패
- Visual Studio Build Tools 2022 + C++ 워크로드 설치 확인
- CUDA Toolkit 12.x 설치 확인 (Visual Studio Integration 포함)
- `build_cuda.bat` 재실행

### 모델 로딩 실패
- 모델 파일이 `models/` 폴더에 있는지 확인
- 모델 아키텍처와 llama-cpp-python 버전 호환성 확인

## 라이선스

MIT License
