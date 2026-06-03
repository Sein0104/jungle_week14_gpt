# mini GPT 구현 과제 보고서

## 0. 반·팀원

| 항목 | 내용 |
| --- | --- |
| 반 | 301 |
| 팀명 | 6조 |
| 팀원 | 김규태, 김세인, 김정환, 김현옥 |

---

## 1. 구현 현황

| 단계 | 구현 내용 | 구현 파일 | 
| --- | --- | --- | 
| 1 | UTF-8 byte-level BPE tokenizer | `src/bpe.py` | 
| 2 | GPTDataset, create_dataloader, InputEmbedding | `src/dataset.py`, `src/embeddings.py` |
| 3 | MultiHeadAttention, causal mask | `src/attention.py` | 
| 4 | LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, generate_text_simple | `src/model.py` | 
| 5 | loss 계산, checkpoint 저장/로드, generate, train_model | `src/train.py` | 
| 6 | NSMC 감성 분류 Dataset과 classifier | `src/finetune.py` | 
| 7 | Colab 사전 학습/미세 조정 실행 셀과 결과 시각화 셀 | `gpt-lab.ipynb` |

---

## 2. 테스트 통과 현황

테스트는 로컬 `.venv` 환경에서 실행했다. 시스템 기본 `python3`에는 `torch`가 설치되어 있지 않아 테스트 수집 단계에서 실패할 수 있으므로, 아래 결과는 `.venv/bin/python` 기준이다.

| 실행 명령 | 결과 | 비고 |
| --- | --- | --- |
| `.venv/bin/python -m pytest tests/test_bpe.py -v` | 통과 | BPE 특수 토큰, 저장/로드, encode/decode, vocab 학습 확인 |
| `.venv/bin/python -m pytest tests/test_dataset.py -v` | 통과 | Dataset, DataLoader, InputEmbedding shape 확인 |
| `.venv/bin/python -m pytest tests/test_attention.py -v` | 통과 | MHA output shape와 causal mask 확인 |
| `.venv/bin/python -m pytest tests/test_model.py -v` | 통과 | GPT 구성 요소, forward loss, simple generation 확인 |
| `.venv/bin/python -m pytest tests/test_train.py -v` | 통과 | loss 계산, checkpoint, generate, plot 호출 확인 |
| `.venv/bin/python -m pytest tests/test_finetune.py -v` | 통과 | 감성 분류 Dataset, classifier, train/eval 함수 확인 |
| `.venv/bin/python -m pytest tests/ -v` | 28 passed, 1 warning | `plt.show()` 관련 non-interactive warning 1개 |



---

## 3. 데이터

| 항목 | 내용 |
| --- | --- |
| 원본 데이터 | NAVER Sentiment Movie Corpus(NSMC) |
| 원본 경로 | `data/ratings_train.txt`, `data/ratings_test.txt` |
| 사전 학습 데이터 | `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt` |
| 미세 조정 데이터 | `data/nsmc_sentiment_train.jsonl`, `data/nsmc_sentiment_val.jsonl`, `data/nsmc_sentiment_test.jsonl` |
| 전처리 방식 | 빈 리뷰 제거, 공백 정리, train/validation/test 분리 |
| 사전 학습 train 크기 | 약 1,379,486자 |
| 사전 학습 validation 크기 | 약 120,560자 |
| 감성 분류 데이터 크기 | train 137,996개, validation 11,999개, test 49,997개 |
| 제출 실험 설정 | Basic |

---

## 4. BPE

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/bpe.py` |
| BPE 방식 | UTF-8 byte-level BPE |
| 특수 토큰 ID | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |
| byte token ID 범위 | 4~259 |
| merge token ID 범위 | 260 이상 |
| vocab_size | 3000 |
| 학습 corpus 크기 | `corpus[:1_500_000]` |
| vocabulary 저장 경로 | `data/nsmc_bpe_vocab_3000.json` |
| 인코딩/디코딩 복원 예시 | `decode(encode("이 영화는 정말 좋았다! English 123")) == 원문` |

한국어 리뷰는 글자 단위로 직접 자르면 UTF-8 byte 경계가 깨질 수 있으므로, 먼저 `text.encode("utf-8")`로 byte sequence를 만든 뒤 BPE merge rule을 적용했다. decode 시에는 merge token을 재귀적으로 byte token까지 펼친 뒤 마지막에 `bytes(...).decode("utf-8")`를 한 번만 수행하도록 구현했다.

---

## 5. 모델 구조

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/model.py` |
| 전체 구조 | InputEmbedding -> 2 x TransformerBlock -> LayerNorm -> LM head |
| vocab_size | 3000 |
| context_length | 128 |
| emb_dim | 128 |
| n_heads | 4 |
| head_dim | 32 |
| n_layers | 2 |
| drop_rate | 0.1 |
| qkv_bias | False |
| 총 파라미터 수 | 1,180,416 |

각 TransformerBlock은 pre-norm 구조로 구성했다. 입력은 LayerNorm 후 causal self-attention을 통과하고 residual connection으로 더해진다. 이후 두 번째 LayerNorm과 FeedForward를 거쳐 다시 residual connection을 적용한다. causal mask를 사용해 현재 token이 미래 token을 참조하지 못하도록 했다.

---

## 6. 사전 학습

### 6.1 하이퍼파라미터

| 구분 | 항목 | 값 |
| --- | --- | --- |
| 데이터 | preset | Basic |
| 데이터 | corpus_size | 1,500,000 |
| 모델 | vocab_size | 3000 |
| 모델 | context_length | 128 |
| 모델 | emb_dim | 128 |
| 모델 | n_heads | 4 |
| 모델 | n_layers | 2 |
| 모델 | drop_rate | 0.1 |
| 모델 | qkv_bias | False |
| 학습 | batch_size | 8 |
| 학습 | num_epochs | 20 |
| 학습 | eval_freq, eval_iter | 100, 10 |
| 학습 | ckpt_freq | 500 |
| 최적화 | optimizer | AdamW |
| 최적화 | learning_rate | 3e-4 |

### 6.2 결과

| 항목 | 내용 |
| --- | --- |
| global step | 약 16,000 |
| train loss | 약 7.3에서 시작해 약 5.0까지 감소 |
| validation loss | 약 7.3에서 시작해 약 5.5까지 감소 |
| 손실 그래프 | Colab 노트북의 `Basic Pretraining Loss by Step`, `Basic Pretraining Loss by Epoch` 그래프 |
| checkpoint 경로 | `outputs/pretrain_basic/final_checkpoint.pt` |
| 학습 기록 경로 | `outputs/pretrain_basic/pretrain_history.json` |

epoch 기준 손실 요약은 다음과 같다. 정확한 history JSON 파일이 로컬에 남아 있지 않아, Colab에서 확인한 그래프와 로그를 바탕으로 근사값을 기록했다.

| epoch | train loss | validation loss | 해석 |
| --- | --- | --- | --- |
| 1 | 약 7.3 | 약 7.3 | 초기에는 train/validation 모두 높은 손실 |
| 5 | 약 5.6 | 약 5.8 | 빠르게 다음 token 예측 패턴을 학습 |
| 10 | 약 5.3 | 약 5.6 | train loss는 계속 감소, validation 개선은 완만해짐 |
| 15 | 약 5.1 | 약 5.5 | train/validation 간 격차 증가 |
| 20 | 약 5.0 | 약 5.5 | 후반부 과적합 가능성 관찰 |

생성 샘플은 학습 초반에는 조사, 어미, 빈번한 한국어 token이 반복되는 경향이 있었다. 예를 들어 `"이 영화는"`을 시작 문맥으로 넣었을 때 `"이 영화는를의은..."`처럼 의미 연결보다 토큰 빈도 패턴이 먼저 나타났다. 이는 작은 모델과 제한된 학습 시간에서 자연스러운 현상이며, 더 긴 학습, 더 나은 regularization, 학습률 스케줄링으로 개선할 수 있다.

---

## 7. 미세 조정

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/finetune.py` |
| 과제 | NSMC 리뷰 긍정/부정 이진 분류 |
| 데이터 포맷 | JSONL, `text`, `label` |
| backbone | 사전 학습된 `GPTModel` |
| classifier | GPT backbone 위에 sequence classification head 추가 |
| max_length | 128 |
| batch_size | 16 |
| learning rate | 1e-4 |
| quick run train 샘플 | 5,000개 |
| quick run validation 샘플 | 1,000개 |
| quick run test 샘플 | 1,000개 |
| validation loss / accuracy | Colab에서 fine-tuning 셀 실행 후 기입 |
| test loss / accuracy | Colab에서 fine-tuning 셀 실행 후 기입 |
| 저장 경로 | `outputs/pretrain_basic/best_sentiment_model.pt`, `outputs/pretrain_basic/finetune_history.json` |

노트북 Step 6 아래에 감성 분류 fine-tuning 실행 셀과 loss/accuracy 그래프 셀을 추가했다. 처음 확인할 때는 작은 샘플 수로 빠르게 동작을 검증하고, 제출용 최종 실험에서는 `MAX_TRAIN_SAMPLES`, `MAX_VAL_SAMPLES`, `MAX_TEST_SAMPLES`를 `None`으로 바꿔 전체 데이터 기준 결과를 기록한다.

오류 분석은 최종 미세 조정 실행 후 틀린 리뷰를 추출해 작성할 예정이다. 예상되는 오류 유형은 짧고 반어적인 리뷰, 긍정/부정 단어가 섞인 리뷰, 맥락 없이 별점 표현만 있는 리뷰이다.

---

## 8. 실험 환경

| 항목 | 내용 |
| --- | --- |
| Python | 3.12.13 (`.venv` 로컬 테스트 기준) |
| PyTorch | 2.12.0 (`.venv` 로컬 테스트 기준) |
| 사전 학습 실행 환경 | Colab GPU |
| 테스트 실행 환경 | 로컬 `.venv` |
| GPU/CPU 정보 | Colab CUDA 사용 로그 확인, GPU 종류는 실행 런타임에 따라 달라짐 |
| 총 학습 소요 시간 | Colab 실행 환경별로 상이, Basic 실험은 약 16,000 global step까지 진행 |

---

## 9. 고찰

- 한국어 byte-level BPE는 UTF-8 byte를 기준으로 다뤄야 하므로, encode/decode 과정에서 byte 경계를 깨뜨리지 않는 것이 중요했다.
- `context_length=128`은 영화 리뷰의 짧은 문맥을 학습하기에는 충분히 실험 가능한 크기였지만, 긴 문장 구조를 안정적으로 반영하기에는 한계가 있다.
- Basic 사전 학습에서 train loss와 validation loss가 모두 감소했으므로 모델이 다음 token 예측 패턴을 학습하고 있음을 확인했다.
- 후반부에는 train loss가 validation loss보다 더 많이 감소해 과적합 가능성이 보였다. `drop_rate` 증가, weight decay, learning rate decay, early stopping을 적용하면 개선 여지가 있다.
- `n_heads=4`, `n_layers=2`, `emb_dim=128`은 빠른 실험에는 적절했지만 생성 문장의 의미 일관성은 아직 부족했다. 모델 용량을 키우거나 학습 corpus를 늘리는 실험이 필요하다.
- 감성 분류 미세 조정은 사전 학습 checkpoint를 backbone으로 재사용하는 구조까지 준비했다. 최종 제출 전에는 전체 감성 분류 데이터로 실행해 validation/test accuracy를 기록해야 한다.
- 다음 개선 방향은 BPE 학습 속도 최적화, cosine learning rate decay, gradient clipping, dropout/weight decay 비교 실험, 생성 샘플 정성 평가 자동화이다.
