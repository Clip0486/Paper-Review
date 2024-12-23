# XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model

- Ho Kei Cheng and Alexander G. Schwing
- University of Illinois Urbana-Champaign
- ECCV 2022

--- 

- 본 논문은 Video Object Segmentation (VOS) 작업을 위한 모델을 제안한다.
- Atkinson-Shiffrin Memory Model에서 영감을 받은 메모리 구조를 가지고 있어 다른 모델과 달리 Long-Term Video를 잘 처리할 수 있는 구조를 가지고 있으며, 새로운 Similarity Function을 제안한다.
- Keyword : Video Object Segmentation (VOS), Atkinson-Shiffrin Memory Model, Feature Memory Stores, Similarity Function

---

## 0. Abstract

**XMem**

Video Object Segmentation (VOS)

VOS는 첫 번째 프레임에서 마스크(Ground Truth)가 주어지고, 이후 프레임에서 객체를 Segmentation하는 작업이다.

![image](https://github.com/user-attachments/assets/76bb19bf-f29c-4eaf-8503-972648fd9bb3)


Atkinson-Shiffrin memory model을 참고하여 3개의 메모리를 사용함

sensory memory – 매 프레임 업데이트

working memory – r 프레임 마다 업데이트

long-term memory – working 메모리가 가득 차면 업데이트

![image](https://github.com/user-attachments/assets/d8ecb3b1-98a1-4301-bb20-efc20b1f2acd)


## 1. Introduction

Video Object Segmentation (VOS)

대부분 VOS 모델은 feature memory 사용함

최근 VOS 모델은 이전 프레임과 현재 프레임을 연결하기 위해 attention 방법 사용

→ 많은 프레임의 정보를 저장하기 때문에 GPU 메모리를 많이 사용함

→ 일반적인 소비자 하드웨어에서 1분 이상의 비디오를 처리하는 것이 어려움

XMem은 Atkinson-Shiffin 메모리 모델을 이용해 여러 독립적이지만, 깊이 연결된 메모리를 사용해 긴 비디오에서 좋음

(왼쪽 : 짧은 비디오, 오른쪽 : 짧은 비디오 (y축), 긴 비디오 (x축), error bar : 메모리 샘플링 표준편차

![image](https://github.com/user-attachments/assets/f2c08a9e-ba4d-45c3-ae20-821220a4a388)


## 2. Related Works

## 3. XMem

### 3.1 Overview

1. 첫번째 프레임에서 객체 마스크 주어짐
   
3. 각각 feature memory를 초기화함
   
4. 이후 프레임에 대해, 각각 메모리를 Memory reading 수행
   
5. 읽어온 특징을 마스크 생성에 사용함
   
6. 메모리 저장소를 각각의 빈도로 업데이트함

    sensory – 매 프레임 업데이트
   
    working – r번째 프레임 업데이트
   
    long-term – working 메모리가 최대치가 되면 특징을 압축하여 저장
   
7. long-term 메모리가 가득차면 (수천 프레임 이후)

오래된 특징을 버리며 GPU 메모리 제한함

![image](https://github.com/user-attachments/assets/c5d46908-ba17-4455-a9e4-f03bb4a286f1)


### 3.2 Memory Reading

모델의 구조는 다음과 같음 

Query encoder : 이미지 특징을 추출 (ResNet-50)

Decoder : Memory reading 단계의 출력을 받아 마스크 생성

Value encoder : 이미지와 생성된 마스크에서 새로운 메모리 특징을 추출 (ResNet-18)

![image](https://github.com/user-attachments/assets/f2ea2132-ffe1-4082-821d-298ee770ea44)

Memory Reading

Query encoder 이미지 특징 추출

𝑊(𝑘, 𝑞)= 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑆(𝑘, 𝑞)) 

Affinity matrix 𝑊(𝑘, 𝑞) = Similarity matrix 𝑆(𝑘, 𝑞) 에 softmax 수행한 것
    
Affinity matrix에 Memory Value 곱하여 𝐹 생성

![image](https://github.com/user-attachments/assets/b17a7e22-ca2d-421e-b213-3a0c321638bc)

Sensory memory ℎ_(𝑡−1)와 𝐹를 이용해 Decoder 마스크 생성

![image](https://github.com/user-attachments/assets/14806e66-8080-4633-90ad-71f6c9e614b6)

Similarity matrix 𝑆(𝑘, 𝑞)

기존의 L2 similarity는 안정적이지만, 표현력이 떨어짐

두 개의 새로운 스케일링 항을 도입해 새로운 Similarity Function (anisotropic L2, 비등방성 L2) 제안함

![image](https://github.com/user-attachments/assets/3fe2fb46-09d8-461c-8e85-4268c9675414)

shrinkage term s ∈[1, ∞) – 신뢰도를 인코딩함

  낮은 Confidence score = 높은 shrinkage term을 가짐 → 영향이 작아짐
        
selection term e ∈[0, 1] – Query 인코더를 통해 q와 함께 생성됨

  키 공간에서 각 채널의 중요성을 제어하여, 더 중요한 채널에 attention 하게 됨
        
s = e = 1 이면 기존 L2 similarity와 동일함

s, e를 추출하는 것은 논문에 포함되지 않음

![image](https://github.com/user-attachments/assets/ecefe82e-a8ff-482b-8bf9-78d73d2f26eb)


### 3.3 Long-Term Memory

Long-Term Memory

긴 비디오를 처리하기 위해, GPU 메모리 사용량을 최소화, 높은 분할 품질은 유지해야 함

1. key 메모리에서 프로토타입 (별표) 선택 (affinity score W가 높은 P개 선택 = 메모리 사용 빈도가 높은 것 선택)

2. 모든 후보들에서 key 값을 집계함 (Memory Potential algorithm)

    Weighted average를 사용함. weight는 ‘후보 key’와 ‘프로토타입 key’를 attention하여 사용함

3. 최종적으로 프로토타입 key, value가 long-term memory에 추가됨

4. 메모리가 가득 차면, 사용량이 적은 메모리가 제거됨

![image](https://github.com/user-attachments/assets/e99a7641-ea6f-457f-a3d4-c66e6cb3a5da)

![image](https://github.com/user-attachments/assets/4983645b-c929-4449-87b4-5a45d43bfee8)


### 3.4 Working Memory

Working Memory

매 r번째 프레임마다 query를 새로운 key로 복사

새로운 key와 Value encoder로 생성한 value를 working memory에 추가

![image](https://github.com/user-attachments/assets/eda743c5-4177-484f-aa89-7bb7f6f974b3)


### 3.5 Sensory Memory

Sensory Memory

디코더는 Query encoder와 Skip-connections 연결되어 Unet과 유사하게 마스크를 생성함

디코더의 다중 스케일 특징을 사용해 GRU로 Sensory Memory를 업데이트

짧은 시간동안 유지, 객체 위치와 같은 저수준 정보를 보유함

GRU (Gated Recurrent Unit)을 통해 매 프레임마다 업데이트 됨

매 r번째 프레임마다 working memory가 업데이트되면, deep update 수행

  (Value encoder를 사용해 또 다른 GRU를 업데이트함)

  1. 이미 Working memory에 저장된 중복 정보를 버림

  2. 최신 정보를 유지하며, 효율적으로 작동할 수 있음

![image](https://github.com/user-attachments/assets/222c1c36-7553-480a-b17e-9bf25635dd61)

GRU (Gated Recurrent Unit)

LSTM (Long Short-Term Memory) 모델을 개선한 모델로, 빠른 학습 시간, 낮은 계산 복잡성을 가짐

리셋 게이트, 업데이트 게이트를 가짐

𝑟_𝑡 : 0에 가까우면 이전 상태를 잊고, 1에 가까우면 이전 상태를 기억함

𝑧_𝑡 : 1에 가까우면 이전 상태를 많이 가져오고, 0에 가까우면 새로운 정보(x)를 많이 가져옴

![image](https://github.com/user-attachments/assets/17ab3b8c-4162-41c9-8414-afd056686405)


### 3.6 Implementation Details

## 4. Experiments

### 4.1 Long-Time Video Dataset

Long-Time Video Dataset

3X : 비디오를 3번 연속하여 재생

대부분 긴 비디오를 처리할 수 없기 때문에, 성능이 떨어짐

![image](https://github.com/user-attachments/assets/e261bd94-9419-4d70-988d-9eebd45befef)


### 4.2 Short Video Datasets

![image](https://github.com/user-attachments/assets/6bade7d5-0c78-491f-b9ec-df1d9a1c8182)

![image](https://github.com/user-attachments/assets/b9947e0f-0223-4edd-83c0-f748478357c3)


### 4.3 Ablations

Ablations

YouTubeVOS 2018 – Y18

DAVIS 2017 – D17

Long-time Video – Lnx

Working X → Long-term memory X

Long-term X → 모든 메모리가 working에 저장

  → 긴 비디오 X, 속도 저하

![image](https://github.com/user-attachments/assets/2148322d-5dbc-48b2-8939-582b5e37c2d0)

![image](https://github.com/user-attachments/assets/67e33fce-f4d7-45c8-90b0-97477ec48810)


### 4.4 Limitations

## 5. Conclusion


