# Visual Information Extraction Task 수행 보고서

## 작성자: 강동욱

[🤗 Dongwooks HF-repo](https://huggingface.co/Dongwookss) 

*본 테스트를 노출시키지 않기 위해 Private로 저장하였습니다. 따라서 ipynb상에 repo 저장부분이 있으나 실제 private 로 저장되었습니다*



![HuggingFace](https://img.shields.io/badge/huggingface-yellow?style=for-the-badge&logo=HuggingFace)
![Colab](https://img.shields.io/badge/Colab-black?style=for-the-badge&logo=GoogleColab)


## 목차
1. [프로젝트 개요](#1-Summary)
2. [실험 결과](#2-Experimental-results)
3. [개발 환경](#3-개발-환경)
4. [기술적 접근 방식](#4-기술적-접근-방식)
5. [결론 및 향후 과제](#5-결론-및-향후-과제)

## 1. Summary

### 1.1 과제 목표

본 프로젝트는 문서 이해 분야의 핵심 과제인 정보 추출(Information Extraction) 태스크를 다룹니다. 구체적으로는 영수증 문서에서 회사명, 날짜, 주소, 총액과 같은 핵심 정보를 자동으로 추출하는 딥러닝 모델을 개발하여, 문서 처리 자동화의 정확성과 효율성을 갖춘 모델을 목표로 합니다.

이를 위해 최신 문서 이해 모델인 LayoutLMv3를 기반으로 하여, 텍스트 정보와 공간 정보를 효과적으로 활용하는 방법을 연구했습니다. 특히 긴 문서 처리를 위한 Sliding Window 기법의 도입과 데이터 품질 개선을 통해, 실제 업무 환경에서 활용 가능한 수준의 성능을 달성하는 것을 목표로 진행하였습니다.

[Dongwooks HF-repo](https://huggingface.co/Dongwookss)

### 1.2 데이터셋 구성
- **입력 데이터**
  - 단어에 대한 정보
  - normalised bounding box 좌표 정보
  - 이미지 정보

- **추출 대상(label) 정보**
  - Company 
  - Date 
  - Address 
  - Total
  - Others

### 1.3 접근 방법
- [LayoutLMv3 모델](https://huggingface.co/microsoft/layoutlmv3-base) Fine-tuning
- Sliding Window 기법 적용
- 데이터 라벨링 품질 개선

## 2. Experimental results

### 2.1 성능 평가 지표
| 버전 | F1  | EM | EM_no_space | 주요 변경사항 |
|-----|----------|------------|------------------|--------------|
| v1 | 76.7090 | 36.1671 | 36.1671 | 기본 구현 |
| v2 | 80.6816 | 50.0000 | 50.0000 | 데이터 처리 개선, 학습 파라미터 조정 |
| v3 | 80.4974 | 49.7839 | 49.7839 | max_length 확장 시도 |
| v4 | 84.9540 | 50.9366 | 50.9366 | 추론 시 Sliding Window 적용 |
| v5 | 84.9650 | 50.4323 | 50.4323 | 학습 시 Sliding Window 적용 |
| v6 | 83.7073 | 52.0173 | 52.0173 | 데이터 라벨링 품질 개선 |

### 2.2 결과 분석
1. **성능 향상 추이**
   - 데이터 처리 개선으로 초기 F1 점수 76.7에서 80.6으로 향상
   - Sliding Window 도입으로 f1 score - 84.96까지 개선
   - 데이터 품질 개선으로 em score - 52.01 달성

2. **주요 개선점**
   - 데이터 전처리 최적화
   - 토큰 처리 제한 문제 해결
   - 라벨링 품질 향상

## 3. Instructions

### 3.1 Hardware/Software 환경
- **개발 플랫폼**: Google Colab
  - 데이터 전처리: CPU 환경
  - 모델 학습/추론: GPU(L4) 환경 

### 3.2 주요 라이브러리
```python
- transformers
- datasets
- torch
```

### 3.3 프로젝트 구조
```
Visual_information/extraction/
├── look_data.ipynb - 데이터 상태를 확인하기 위해 해당 노트북을 생성하였습니다.
├── making_dataset&modify_dataset.ipynb - 제공된 *.txt 파일과 img 데이터를 활용해서 Dataset을 생성하였습니다. 또한 훈련 데이터 개선을 위한 작업을 진행하였습니다.
├── model_training_main.ipynb - 데이터 훈련 과정에 대한 코드입니다.
└── data/
    ├── train/
        └── entities/
        └── img/
    ├── test/
        └── entities/
        └── img/

    ├── op_test.txt
    ├── op_test_box.txt
    ├── op_test_image.txt

    ├── train.txt
    ├── train_box.txt
    ├── train_image.txt

    ├── test.txt
    ├── test_box.txt
    └── test_image.txt

```

## 4. 기술적 접근 방식

### 4.1 초기 분석 및 모델 선정
1. **데이터 분석**
   - 데이터 구조 및 품질 분석 진행 - look_data.ipynb를 활용하여 직접 데이터 라벨링 확인
       - 데이터 label의 정확도가 낮은 점을 확인
   - 라벨링 패턴 분석

| 특성 | LayoutLM | LayoutLMv2 | LayoutLMv3 |
|-----|----------|------------|------------|
| **기본 구조** | BERT 기반 + 2D 위치 임베딩 | LayoutLM + 시각적 임베딩 | Transformer 기반 통합 아키텍처 |
| **주요 특징** | - 텍스트와 레이아웃 정보 통합<br>- 단순한 구조 | - 시각적 백본 도입<br>- 텍스트-이미지 정렬 | - 단일 다중 모달 인코더<br>- WPA(Word-Patch Alignment) |
| **장점** | - 학습 효율성 높음<br>- 빠른 추론 속도 | - 이미지 특징 활용<br>- 향상된 성능 | - 효율적인 통합 처리<br>- 최고 수준의 성능<br>- 계산 자원 효율성 |
| **한계점** | - 이미지 특징 미활용<br>- 제한적 모델링 | - 복잡한 구조<br>- 높은 학습 비용 | - 큰 모델 크기<br>- 높은 메모리 요구량 |
| **성능** | Form Understanding<br>FUNSD: 79.3% | Form Understanding <br> FUNSD: 82.8% | Form Understanding <br> FUNSD: 85.4% |
| **선정 여부** | ❌ | ❌ | ✅ |
| **선정 이유** | - | - | - 단일 인코더 효율성<br>- 향상된 정보 통합<br>- 최신 사전학습 기법<br>- SOTA 성능 |

### 4.2 개발 과정

#### 1) 기본 구현 (v1-v2)
```python
training_args = TrainingArguments(
    output_dir="test",
    max_steps=1000,  # v2: 1500
    learning_rate=1e-5,  # v2: 2e-5
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```
- 데이터셋 구축 및 기본 학습
- 파라미터 최적화 진행

#### 2) 토큰 제한 문제 해결 (v3-v5)
- **문제 발견**
  - 긴 텍스트 처리 시 토큰 손실
  - Position Embedding 제한 (514 토큰)

- **Sliding Window 구현**

| 구분 | 적용 전 | 적용 후 |
|-----|---------|---------|
| **처리 방식** | 단일 패스로 전체 문서 처리 | 중첩 윈도우로 분할 처리<br>`window_size=384, stride=192` |
| **장점** | - 구현 단순<br>- 문맥 유지 용이<br>- 메모리 효율적 | - 긴 문서 처리 가능<br>- 토큰 손실 방지<br>- Position Embedding 제한 극복 |
| **단점** | - 긴 문서 절단<br>- Position Embedding 제한<br>- 토큰 손실 발생 | - 구현 복잡도 증가<br>- 중복 처리 필요<br>- 경계 부분 문맥 유실 가능성 |
| **성능** | F1: 80.4974<br>EM: 49.7839 | F1: 84.9650<br>EM: 50.4323 |
| **메모리 사용** | 낮음 | 중복 처리로 인한 증가 |
| **처리 속도** | 빠름 | 중복 영역 처리로 인한 지연 |
| **활용 사례** | - 짧은 문서<br>- 단순한 레이아웃 | - 긴 문서<br>- 복잡한 레이아웃<br>- 정밀한 정보 추출 필요 시 |


  ```
  Window Size: 384 tokens
  Stride: 192 tokens
  Overlap: 192 tokens
  ```

#### 3) 데이터 품질 개선 (v6)
- **라벨링 개선**
  - entities 정보 활용
  - 정확한 매칭 규칙 적용
  - 일관성 검증

### 4.3 성능 최적화
```python
training_args = TrainingArguments(
    output_dir="test",
    max_steps=1500,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    gradient_accumulation_steps=4
)
```

## 5. 결론 및 향후 과제

### 5.1 주요 성과
1. **모델 성능**
   - F1 점수: 84.9650
   - 정확도(EM): 50.4323

2. **기술적 성과**
   - Sliding Window 기법 성공적 구현
   - 데이터 품질 개선 방법론 확립

### 5.2 한계점
1. 긴 텍스트 처리의 제약
2. 필드별 성능 편차

### 5.3 향후 개선 방향
1. **데이터 처리 고도화**
   - 필드별 패턴 인식 규칙 확장
   - 예외 케이스 처리 강화

2. **모델 최적화**
   - Sliding Window 파라미터 튜닝
   - 필드별 특화 모델 검토

3. **시스템 안정성**
   - 자동화된 품질 검증 시스템
   - 필드별 검증 규칙 체계화

---
