# Visual Information Extraction Task 수행 보고서

## 작성자: 강동욱

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [실험 결과](#2-실험-결과)
3. [개발 환경](#3-개발-환경)
4. [기술적 접근 방식](#4-기술적-접근-방식)
5. [결론 및 향후 과제](#5-결론-및-향후-과제)

## 1. 프로젝트 개요

### 1.1 과제 목표
본 프로젝트는 SROIE 데이터셋을 활용하여 영수증 문서에서 주요 정보(company, date, address, total)를 자동으로 추출하는 시스템을 개발하는 것을 목표로 합니다. 이는 문서 처리 자동화의 핵심 과제로, 정확하고 효율적인 정보 추출이 요구됩니다.

### 1.2 데이터셋 구성
- **입력 데이터**
  - 단어 및 bounding box 좌표 정보
  - BIO 태깅 정보 (conll format)
  - 정규화된 좌표 데이터

- **추출 대상 정보**
  - Company (회사명)
  - Date (날짜)
  - Address (주소)
  - Total (총액)

### 1.3 접근 방법
- [LayoutLMv3 모델](https://huggingface.co/microsoft/layoutlmv3-base) Fine-tuning
- Sliding Window 기법 적용
- 데이터 라벨링 품질 개선

## 2. 실험 결과

### 2.1 성능 평가 지표
| 버전 | F1 점수 | 정확도(EM) | 정확도(공백 제외) | 주요 변경사항 |
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
   - Sliding Window 도입으로 84.96까지 개선
   - 데이터 품질 개선으로 최종 정확도 52.01 달성

2. **주요 개선점**
   - 데이터 전처리 최적화
   - 토큰 처리 제한 문제 해결
   - 라벨링 품질 향상

## 3. 개발 환경

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
project/
├── look_data.ipynb // 데이터 상태를 확인하기 위해 해당 노트북을 생성하였습니다.
├── making_dataset&modify_dataset.ipynb // 제공된 *.txt 파일과 img 데이터를 활용해서 Dataset을 생성하였습니다. 또한 훈련 데이터 개선을 위한 작업을 진행하였습니다.
├── model_training_main.ipynb // 데이터 훈련 과정에 대한 코드입니다.

```

## 4. 기술적 접근 방식

### 4.1 초기 분석 및 모델 선정
1. **데이터 분석**
   - 데이터 구조 및 품질 평가
   - 라벨링 패턴 분석

2. **모델 선정**
   - LayoutLMv3 선택 이유
     - 최신 아키텍처 적용
     - 공간 정보 활용 가능
     - 우수한 성능 기록

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
   - F1 점수: 83.7073
   - 정확도(EM): 52.0173

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
