# Visual Information Extraction Task 수행 보고서

## 작성자: 강동욱

[🤗 Dongwooks HF-repo](https://huggingface.co/Dongwookss) 

*본 테스트를 노출시키지 않기 위해 데이터셋은 Private로 저장하였습니다.*     
*따라서 ipynb상에 repo 저장부분이 있으나 실제 private 로 저장되었습니다*    
*결과 모델은 용량으로 인해 git upload가 불가하여 huggingface에 업로드 하였습니다*  

*진행 과정에서 쉬운 모델 저장을 위해 Huggingface에 private상태로 업로드 되어있으며 평가를 위해 필요할 경우 public으로 하겠습니다!*    
Result_Model : [VIE_TASK_v5](https://huggingface.co/Dongwookss/vie_task_v5)


![HuggingFace](https://img.shields.io/badge/huggingface-yellow?style=for-the-badge&logo=HuggingFace)
![Colab](https://img.shields.io/badge/Colab-black?style=for-the-badge&logo=GoogleColab)


## 목차
1. [Summary](#1-summary)
2. [Experimental Results](#2-experimental-results)
3. [Instructions](#3-instructions)
4. [Approach](#4-approach)
5. [결론 및 향후 과제](#5-결론-및-향후-과제)

## 1. Summary

### 1.1 과제 목표

본 프로젝트는 문서 이해 분야의 핵심 과제인 정보 추출(Information Extraction) 태스크를 다룹니다. 구체적으로는 영수증 문서에서 회사명, 날짜, 주소, 총액과 같은 핵심 정보를 자동으로 추출하는 딥러닝 모델을 개발하여, 문서 처리 자동화의 정확성과 효율성을 갖춘 모델을 목표로 하였습니다.

이를 위해 최신 문서 이해 모델인 LayoutLMv3를 기반으로 하여, 텍스트 정보와 공간 정보를 효과적으로 활용하는 방법을 연구했습니다.  Sliding Window 기법의 도입과 데이터 품질 개선을 통해, 실제 업무 환경에서 활용 가능한 수준의 성능을 달성하는 것을 목표로 진행하였습니다. 최종적으로 **F1 점수 84.96**, **정확도(EM) 50.43**을 달성하였습니다.

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
- transformers==4.46.2
- torch==2.5.1+cu121
- datasets==3.1.0  
- huggingface-hub==0.26.2
- seqeval==1.2.2
- sentence-transformers==3.2.1
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

### 3.4 making_dataset&modify_dataset.ipynb
making_dataset.ipynb
<details> 
<summary>데이터셋 생성 및 전처리 과정</summary>

#### 1. 환경 설정 및 데이터 검증
```python
# 필요 라이브러리 설치
!pip install transformers datasets seqeval
!git lfs install

# 데이터 무결성 검사
def validate_data():
    for target in ['train', 'test', 'op_test']:
        # 각 데이터 파일 비교 검증
        # - image_path: 바운딩 박스 좌표
        # - text_path: 텍스트 및 라벨
        # - box_path: 정규화된 좌표
```

#### 2. 데이터셋 생성
```python
def create_dataset(target_type):
    """학습/테스트 데이터셋 생성"""
    dataset = []
    
    # 파일 경로 설정
    txt_path = f"data/{target_type}.txt"
    box_path = f"data/{target_type}_box.txt"
    image_path = f"data/{target_type}_image.txt"
    
    # 데이터 로드 및 처리
    - words: 텍스트 정보
    - bboxes: 바운딩 박스 좌표
    - norm_bboxes: 정규화된 좌표
    - labels: BIO 태깅 정보
    
    # JSON 형식으로 저장
    save_to_json(f"{target_type}_dataset.json")
```

#### 3. 데이터셋 품질 개선
```python
def enhance_dataset():
    # 1. 공간 정보 활용
    def group_by_lines(words, bboxes, y_threshold=5):
        """세로 위치 기반 라인 그룹화"""
        # y좌표 기반 텍스트 라인 식별
        # 동일 라인 내 단어 정렬
    
    # 2. 엔티티별 매칭 규칙
    def find_company_line(lines, company):
        """회사명 식별 규칙"""
        # 문서 상단 위치
        # 특정 키워드 활용
    
    def find_address_line(lines, address):
        """주소 식별 규칙"""
        # 시작점: NO., LOT, JALAN 등
        # 종료점: MALAYSIA, DARUL EHSAN
        # 중단점: TEL, FAX, EMAIL
    
    def find_date_line(lines, target_date):
        """날짜 식별 규칙"""
        # 날짜 포맷 패턴 매칭
        
    def find_total_line(lines, total):
        """총액 식별 규칙"""
        # TOTAL, AMOUNT 키워드
        # 숫자 포맷 검증
```

#### 4. Hugging Face 데이터셋 변환
```python
def convert_to_hf_dataset():
    # 데이터셋 특성 정의
    features = Features({
        'image': Image(),
        'label': Sequence(...),
        'words': Sequence(...),
        'bbox': Array2D(...),
    })
    
    # Dataset 객체 생성
    train_dataset = Dataset.from_dict(...)
    eval_dataset = Dataset.from_dict(...)
    
    # Hugging Face Hub 업로드
    dataset.push_to_hub("Dongwookss/SROIE")
```
</details>

model_training_main.ipynb
<details>
<summary>모델 학습 및 추론 과정</summary>

#### 1. 데이터 및 모델 준비
```python
# 데이터셋 로드
dataset = load_dataset("Dongwookss/SROIE_lb1")
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")

# 라벨 정의
label_list = ["S-COMPANY", "S-DATE", "S-ADDRESS", "S-TOTAL", "O"]
id2label = {k: v for k,v in enumerate(label_list)}
```

#### 2. 데이터 전처리
```python
def prepare_examples(examples, window_size=384, stride=192):
    """Sliding Window 적용 데이터 처리"""
    
    # 입력 처리
    - 이미지 포맷 변환 (RGB)
    - 텍스트 토큰화
    - 바운딩 박스 정규화
    
    # Sliding Window 적용
    - window_size=384로 문서 분할
    - stride=192로 중첩 영역 설정
    - 토큰화 및 패딩 처리
    
    # 결과 포맷
    - pixel_values
    - input_ids
    - attention_mask
    - bbox
    - labels
```

#### 3. 모델 학습
```python
# 학습 설정
training_args = TrainingArguments(
    output_dir="test",
    max_steps=1500,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    evaluation_strategy="steps",
    eval_steps=100,
    metric_for_best_model="f1"
)

# 평가 메트릭
def compute_metrics(p):
    """seqeval 기반 성능 평가"""
    - precision
    - recall
    - f1
    - accuracy

# 모델 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
```

#### 4. 추론 및 결과 처리
```python
def process_with_sliding_window(example, model, processor):
    """Sliding Window 기반 추론"""
    
    # 윈도우 처리
    def safe_process_window():
        # 단일 윈도우 처리
        # word_ids 추적
        # 예측 결과 저장
    
    # 전체 문서 처리
    - 첫 번째 윈도우 처리
    - 미처리 단어 확인
    - 두 번째 윈도우 처리
    - 결과 병합
    
    # 후처리
    - 예측 라벨 정리
    - 결과 포맷팅
    - CSV 파일 저장
```

#### 5. 모델 저장 및 배포
```python
def save_and_upload():
    """모델 저장 및 Hugging Face Hub 업로드"""
    
    # 모델 저장
    trainer.save_model()
    
    # Hugging Face Hub 업로드
    upload_folder_to_huggingface(
        folder_path="test/checkpoint-1500",
        repo_id="Dongwookss/vie_task",
        token=HF_TOKEN
    )
```
</details>


## 4. Approach

### 4.1 초기 분석 및 모델 선정
1. **데이터 분석**
   - 좋은 모델을 위해서는 좋은 데이터셋이 필요하다고 생각하였습니다.
   - txt 파일 기반으로 데이터셋을 생성하고 직접 보며 label이 잘 되어있는지 확인하였으나<br>초기에 발견보다 나중에 데이터셋의 문제점을 발견하게 되었습니다.
   - 데이터 구조 및 품질 분석 진행 - look_data.ipynb를 활용하여 직접 데이터 라벨링 확인
       - 데이터 label의 정확도가 낮은 점을 확인
   - 라벨링 패턴 분석

    
2. **모델 선정**  

| 특성 | LayoutLM | LayoutLMv2 | LayoutLMv3 |
|-----|----------|------------|------------|
| **기본 구조** | BERT 기반 + 2D 위치 임베딩 | LayoutLM + 시각적 임베딩 | Transformer 기반 통합 아키텍처 |
| **주요 특징** | - 텍스트와 레이아웃 정보 통합<br>- 단순한 구조 | - 시각적 백본 도입<br>- 텍스트-이미지 정렬 | - 단일 다중 모달 인코더<br>- WPA(Word-Patch Alignment) |
| **장점** | - 학습 효율성 높음<br>- 빠른 추론 속도 | - 이미지 특징 활용<br>- 향상된 성능 | - 효율적인 통합 처리<br>- 최고 수준의 성능<br>- 계산 자원 효율성 |
| **한계점** | - 이미지 특징 미활용<br>- 제한적 모델링 | - 복잡한 구조<br>- 높은 학습 비용 | - 큰 모델 크기<br>- 높은 메모리 요구량 |
| **성능** | Form Understanding<br>FUNSD: 79.3% | Form Understanding <br> FUNSD: 82.8% | Form Understanding <br> FUNSD: 85.4% |
| **선정 여부** | ❌ | ❌ | ✅ |
| **선정 이유** | - | - | - 단일 인코더 효율성<br>- 향상된 정보 통합<br>- 최신 사전학습 기법<br>- SOTA 성능 |


    

3. **Fine-tuning의 필요성**

| 특성 | LayoutLMv3 (base) | LayoutLMv3 (fine-tuned) |
|-----|----------|------------|
| **성능** | F1: 9.64<br>EM: 0.00<br>EM_no_space: 0.00 | F1: 84.96<br>EM: 50.43<br>EM_no_space: 50.43 |
| **분석** | - 사전학습만 된 상태<br>- SROIE 태스크에 대한 훈련 없음<br>- Zero-shot 성능 낮음 | - SROIE 데이터로 Fine-tuning<br>- 태스크에 특화된 학습 완료<br>- 높은 성능 달성 |

Base 모델과 Fine-tuned 모델의 성능 차이로 도메인 특화 학습의 중요성을 볼 수 있었습니다.     
사전학습된 LayoutLMv3 base 모델은 일반적인 문서 이해 능력을 보유하고 있으나, 특정 도메인(영수증)과 태스크(정보 추출)에 대한 fine-tuning 없이는 실용적인 성능을 달성하기 어렵다는 것을 확인할 수 있었습니다.

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

#### 2) Sliding Window 도입 (v3-v5)
- **문제 발견**
  - v3 
    - output.csv 와 input 데이터를 비교하였을때 갯수가 맞지 않는것을 발견하였습니다. 
    - 원인을 분석하던 중 파일별 단어 처리갯수를 print문을 통해 비교하였습니다.
    - 이 결과, X51007846283 이미지 및 단어정보에 대한 토큰값이 512를 넘어서게 되어 output으로 나오지 않는 것을 발견하였습니다.
    - Position Embedding 514 토큰 제한 이슈
    - processor의 max_length = 2024로 수정하여 모델 훈련 진행하였습니다.
  - v4 
    - max_lenght를 높여도 단어의 토큰수가 일정 이상 올라가면 인식이 안되는 것을 발견하였습니다.
    - position embedding에 있어서도 토큰수가 제한되어 있을 수 있다는 것을 발견하고 새로운 기법을 적용시켰습니다.
    - v3 결과 모델에 inference 를 진행하며 sliding_window기법을 적용시킨 결과 결과물 평가지표가 향상됨을 발견하였습니다.
    - v3 f1 : 80.4974 -> v4+sliding_window f1 : 84.9540
    - Sliding window 기법 적용을 위해 훈련 단계에서부터 적용을 목표로 v5를 진행하였습니다.
  - v5
    - Dataset의 train과 test set에 대해서도 sliding_window를 적용시켜 훈련을 진행하였습니다.   

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



- **구현 세부사항**
  ```python
  # Sliding Window 파라미터
  window_size = 384  # 단일 윈도우 크기
  stride = 192      # 윈도우 이동 간격
  overlap = 192     # 중첩 영역
  
  # window 처리 로직
  def process_with_sliding_window(text):
      windows = []
      for i in range(0, len(text), stride):
          window = text[i:i + window_size]
          windows.append(window)
      return windows 


#### 3) 데이터 품질 개선 (v6)
- **라벨링 개선**
  - entities 정보 활용하여 label 수정
  - label 매칭 규칙 적용하여 기존 txt 기반 데이터셋 수정

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
1. **모델 결과 및 성능**
   - 제출모델 : [v5](https://huggingface.co/Dongwookss/vie_task_v5)
   - F1 점수: 84.9650
   - 정확도(EM): 50.4323

3. **기술적 성과**
   - Sliding Window 기법 성공적 구현
   - 데이터 품질 개선 방법론 확립

### 5.2 한계점
1. 긴 텍스트 처리의 제약
2. 필드별 성능 편차로 인한 새로운 데이터 처리과정 필요

### 5.3 향후 개선 방향
1. **데이터 처리 고도화**
   - 예외처리 혹 규칙기반 외 다른 방법론 적용이 필요함.

2. **모델 최적화**
   - Sliding Window 파라미터 튜닝을 통한 일부 성능개선
   - 필드별 특화 모델을 통한 원하는 label에 대한 quality가 보장되는 훈련데이터 구축

3. **시스템 안정성**
   - 필드별 검증 규칙 체계화
---
