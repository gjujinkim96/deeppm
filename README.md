## 설치 조건
### 필요 library
- https://github.com/tatp22/multidim-positional-encoding
- https://docs.wandb.ai/quickstart

- scikit-learn 필요
- 최소 pytorch 버전 1.13.1


## Tokenizer
* 주어진 basic block을 있는 그대로 tokenize하기 위해 사용
* 기존에 사용하던 tokenizer는 instruction에 canonicalized을 적용해서 암시적인 정보를 추가로 제공함, 하지만 필요한 정보를 추출하는 과정에서 정보 손실이 일어남.

### Tokenize 과정

- 각각 명령어의 시작과 끝에 ```<START```, ```<END>``` 추가
    - 첫 번째 명령어에는 ```<START>``` 대신에 ```<BLOCK_START>```
    - 마지막 명령어에는 ```<END>``` 대신 ```<BLOCK_END>```
- 특수기호들도 하나의 토큰으로([ ] +’, ‘-’, ‘,’)
    - ```[```
    - ```]```
    - ```+```
    - ```-```
    - ```*```
    - ```,```
    - ```:```
- 0x로 시작하는 constant:
    - constant = 0: ```<ZERO_{constant_byte_size}_BYTES>```로 변경
    - constant ≠ 0: ```<NUM_{constant_byte_size}_BYTES>```로 변경
    - 예시) 0x03 ⇒ ```<NUM_1_BYTES>```
 
        0x00000012 ⇒ ```<NUM_4_BYTES>```
        
        0x0000 ⇒ ```<ZERO_2_BYTES>```
        
        0x00 ⇒ ```<ZERO_1_BYTES>```
        
- tokenizer 사용시 처음 보는 토큰은 ```<UNK>``` 토큰으로 변경

예시)
```
push   rbx
test   byte ptr [rdi+0x0e], 0x01

tokenizer 사용시

<BLOCK_START> push rbx <END>
<START> test byte ptr [ rdi + <NUM_1_BYTES> ] , <NUM_1_BYTES> <BLOCK_END>
```