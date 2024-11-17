# Visual Information Extraction Task

SROIE 데이터셋을 활용한 문서 정보 추출 태스크 구현

## 1. Summary
본 프로젝트는 영수증 이미지에서 주요 정보(회사명, 날짜, 주소, 총액)를 추출하는 과제를 수행했습니다. [**LayoutLMv3 모델**](https://huggingface.co/microsoft/layoutlmv3-base) Fine-tuning을 진행하여 구현하였으며, 데이터 처리 방식의 개선과 Sliding Window 기법 도입을 통해 성능을 점진적으로 향상시켰습니다. 또한 데이터 라벨 수정 작업을 통해 데이터 품질을 개선하였으며, 최종적으로 F1 점수 84.965, 정확도 50.4323을 달성했습니다.

[Dongwooks HF-repo](https://huggingface.co/Dongwookss)

## 2. Experimental Results

### 성능 평가 결과
| 버전 | F1 점수 | 정확도(EM) | 정확도(공백 제외) | 주요 변경사항 |
|-----|----------|------------|------------------|--------------|
| v1 | 76.7090 | 36.1671 | 36.1671 | 기본 구현 |
| v2 | 80.6816 | 50.0000 | 50.0000 | 데이터 처리 개선, 학습 파라미터 조정 |
| v3 | 80.4974 | 49.7839 | 49.7839 | max_length 확장 시도 |
| v4 | 84.9540 | 50.9366 | 50.9366 | 추론 시 Sliding Window 적용 |
| v5 | 84.9650 | 50.4323 | 50.4323 | 학습 시 Sliding Window 적용 |

## 3. Instructions

### 개발 환경
- Google Colab
  - 데이터 전처리: CPU 환경
  - 모델 학습 및 추론: GPU(L4) 환경
- 주요 라이브러리
  ```
  - transformers
  - datasets
  - torch
  ```

### 코드 구조
```
project/
├── preprocessing/
│   ├── data_loader.py     # SROIE 데이터셋 로드
│   ├── text_processor.py  # 텍스트 및 좌표 처리
│   └── label_modifier.py  # 라벨 수정 처리
├── model/
│   ├── train.py          # 모델 학습
│   └── inference.py      # 추론
└── utils/
    └── evaluation.py     # 성능 평가
```

### 실행 방법
1. 데이터 전처리
   ```python
   # SROIE 데이터셋 로드 및 처리
   ```

2. 모델 학습
   ```python
   python modeltrain.ipynb
   ```

3. 평가 및 추론
   ```python
   python modeltrain.ipynb
   ```

## 4. Approach

### 구현 과정 및 개선사항

#### v1 (기본 구현)
```python
training_args = TrainingArguments(
    output_dir="test",
    max_steps=1000,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-5,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```
- 문제점: 데이터 처리 시 공백이 있는 단어 분할 문제 발생

#### v2 (파라미터 최적화)
- 데이터 처리 개선
- 학습 파라미터 최적화 (max_steps: 1500, learning_rate: 2e-5)
- 문제점: 긴 문자열 절단 현상 발생

#### v3-v5 (Sliding Window 도입)
```python
training_args = TrainingArguments(
    output_dir="test",
    max_steps=1500,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    gradient_accumulation_steps=4
)
```
- Sliding Window 기법 도입으로 성능 향상
- 학습 및 추론 단계 모두에 적용

### 데이터 라벨 수정 프로세스

1. 공간 정보 활용
   ```python
   def group_by_lines(words, bboxes, y_threshold=5)
   ```
   - Bounding box 좌표 기반 분석
   - 동일 행 단위 그룹화

2. 구조적 분석
   ```python
   def find_company_line(lines, company)
   def find_address_line(lines, address)
   def find_date_line(lines, target_date)
   def find_total_line(lines, total)
   ```

3. 주소 인식 규칙 개선
   - 시작점: NO., LOT, JALAN, JLN
   - 종료점: MALAYSIA, DARUL EHSAN
   - 중단점: TEL:, FAX:, EMAIL:

### 향후 개선 방안
1. 데이터 전처리 고도화
   - 필드별 패턴 인식 규칙 확장
   - 예외 케이스 처리 강화

2. 모델 성능 최적화
   - Sliding Window 파라미터 튜닝
   - 필드별 특화 모델 검토



중요하게 생각하는 **Points** 
- train_data Quality -> entities 에 json과 비교하여 라벨수정
- parameters

Troubles
- inference decoding 으로 인한 inference_result 문제 encoding.word_ids() 사용
- max_length 지정으로 인한 inference_result 문제
  -   // ## 모델  position embeddings이 514 -> 2048   : embedding max값 제한
  -   sliding window 기법 적용하여 추론 과정 진행 -> 개선
