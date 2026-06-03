# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []
        self._init_special_tokens()

    # 파이썬 내장 클래스 bytes를 이용해 255개의 토큰을 미리 생성합니다.
    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다. 머야 위에 등록이 되어 있잖아? 
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        # 먼저 특수 토큰부터 등록합니다. 
        for token, idx in SPECIAL_IDS.items() :
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx
        
        #그 후 일반 토큰을 등록합니다.
        for byte_value in range(256) :
            self.id_to_token[byte_value + BYTE_OFFSET] = bytes([byte_value])
            self.token_to_id[bytes([byte_value])] = byte_value + BYTE_OFFSET

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.
        BPE merge rule은 자주 등장하는 토큰 쌍을 합쳐서 새 토큰으로 만드는 규칙을 말합니다.

        구현 힌트:
        - 1. `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 2. 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 3. 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - 4. `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        #1. `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        token_ids = list(corpus.encode("utf-8"))

        while (self.vocab_size - len(self.id_to_token) > 0 and len(token_ids) > 1 ) :

            #2. 가장 자주 등장하는 이웃 token pair를 찾습니다.
            # while문 안에 있어야 하는 이유는, 이전 merge로 생성된 새 토큰이 다음 pair 후보가 될 수 있기 때문입니다.
            token_pair = {}
            for i in range(len(token_ids)-1) :            
                if (token_ids[i], token_ids[i+1]) in token_pair :
                    token_pair[(token_ids[i], token_ids[i+1])] += 1
                else : 
                    token_pair[(token_ids[i], token_ids[i+1])] = 1
            best_pair = max(token_pair, key=lambda x: token_pair[x])

            #3. 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
            new_id = len(self.id_to_token)
            new_token_ids = []

            i = 0
            while i < len(token_ids) - 1 :
                if (token_ids[i], token_ids[i+1]) == best_pair :
                    new_token_ids.append(new_id)
                    i += 2                    
                else : 
                    new_token_ids.append(token_ids[i])
                    i += 1

            token_ids = new_token_ids

            #4. `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
            self.merges.append(best_pair)
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id
            

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """

        serializable = {} 
        for k, v in self.id_to_token.items():
            if isinstance(v, bytes) :
                serializable[k]=list(v)
            elif isinstance(v, tuple):
                serializable[k]= list(v)
            else : 
                serializable[k] = v


        temp_merges = []
        for k in self.merges :
            temp_merges.append(list(k))

        data = {
            "merges" : temp_merges,
            "id_to_token" : serializable
        }
        
        with open(path, "w") as f :
            json.dump(data, f, indent=4)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        with open(path, "r") as f :
            temp = json.load(f)
            temp_merges = temp['merges']
            temp_id_to_token = temp['id_to_token']
                        
        for i in range(len(temp_merges)) :
            self.merges.append(tuple(temp_merges[i]))
        
        for i in range(len(temp_id_to_token)) :
            if type(temp_id_to_token[str(i)]) is str : # 특수토큰 (<unk> 등)인 경우
                self.id_to_token[i] = temp_id_to_token[str(i)]
            elif type(temp_id_to_token[str(i)]) is list and len(temp_id_to_token[str(i)]) == 1 : #일반토큰 bytes인 경우
                #TODO : temp_id_to_token을 bytes로 반환한 후, id_to_token을 수정합니다.
                self.id_to_token[i] = bytes(temp_id_to_token[str(i)])
            else : #merge 토큰일 경우 
                #TODO : temp_id_to_token을 tuple로 변환한 후, id_to_token을 수정합니다.)
                self.id_to_token[i] = tuple(temp_id_to_token[str(i)])


    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        
        token_ids = list(text.encode("utf-8"))

        for merge in self.merges :
            new_token_ids = []        
            i = 0
            while i < len(token_ids) - 1 : 
                if (token_ids[i], token_ids[i+1]) == merge :
                    new_token_ids.append(self.token_to_id[merge])
                    i += 2
                else :
                    new_token_ids.append(token_ids[i])
                    i += 1   
            
            #현재 페어를 찾기 위해 len - 1까지만 반복하기 때문에, 마지막 원소를 따로 추가해주어야 합니다.
            new_token_ids.append(token_ids[i])
            token_ids = new_token_ids

        if add_bos_eos :
            token_ids = [self.token_to_id[BOS_TOKEN]] + token_ids + [self.token_to_id[EOS_TOKEN]]

        return token_ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        byte_tokens = []
        result = []
        
        #재귀용 헬퍼 함수. 
        def _flatten(ids, result) -> list[int] :
            for token_id in ids :
                if token_id >= 260 :
                    _flatten(list(self.id_to_token[token_id]), result)
                elif skip_special and token_id in (0, 1, 2, 3) :
                    continue
                else :
                    result.append(token_id)
            return result

        # 
        byte_tokens = _flatten(ids, result)

        return bytes(result).decode("utf-8")
        
        
        
# """
# smoke Test
# TODO: 구현을 모두 완료하고 난 뒤엔 아래 코드를 지워 주세요!
# """
# if __name__ == "__main__" :
#     tokenizer = BPETokenizer(vocab_size=300)
#     tokenizer._init_special_tokens()

#     with open("data/ratings_train.txt", "r") as f:
#         corpus = f.read()
    
#     tokenizer.train(corpus[:5000])
#     tokenizer.save("test.json")
#     tokenizer.load("test.json")
#     print(tokenizer.decode(tokenizer.encode("난 괴로워.. 네가 나 아니라 다른 사람에게만 웃고 사랑을 말하고 또 그렇게 미워해 날", add_bos_eos= True)))
