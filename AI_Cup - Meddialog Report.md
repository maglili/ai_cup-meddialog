# AI_Cup - Meddialog Report
**隊伍名稱:** 阿財專業檳榔攤(雙子星、結冰水)
**組員1:** 電機所 碩一 N26091194 鄧立昌
**組員2:** 電機所 碩一 N26092116 宋士莆
[colab 連結 ](https://colab.research.google.com/drive/1dBkluZ-jpx_sgbNa-Hglck2954ApDB9J?usp=sharing)

## 分析問題與模型介紹
### 分析問題
本次的資料去識別化比賽，就是進行 NER 任務來判斷隱私詞彙。
為了進行 NER 任務，在傳統機器學習有 HMM、CRF...等模型。
但近年由於 bert 它在各個 NLP 任務的 Benchmark 中，都有著不錯的表現，因此我們組決定使用 bert 來作為我們的 model。

### BERT 模型介紹
在原始論文中，BERT 可以在下游接線性分類器來進行 NER 任務，如下圖(d)
![](https://i.imgur.com/tA22jRk.png)

[出處:BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

由圖(d)可知將句子輸入至 bert 後，會輸出每個 token 的 embedding，我們能看成是分類問題，利用線性分類器將 embedding 進行分類 。

### 程式流程
#### 一、載入所需的套件
#### 二、處理原始數據
1. 把原始數據(e.g. train_2.txt)中的所有醫病對話儲存到　list 中
2. 把文章跟每個字的標籤存起來，data_text用來儲存文章，data_label用來儲存每個文章的每個字的label
    ```python
    # 例如
    data_text = ['醫師:啊回去還好嗎?']
    data_label = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    ```
3. 把 label 轉換成 id
    ```python
    {'B-ID': 13,
     'B-clinical_event': 15,
     'B-contact': 21,
     'B-education': 17,
     'B-family': 11,
     'B-location': 9,
     'B-med_exam': 3,
     'B-money': 19,
     'B-name': 7,
     'B-organization': 23,
     'B-others': 25,
     'B-profession': 5,
     'B-time': 1,
     'I-ID': 14,
     'I-clinical_event': 16,
     'I-contact': 22,
     'I-education': 18,
     'I-family': 12,
     'I-location': 10,
     'I-med_exam': 4,
     'I-money': 20,
     'I-name': 8,
     'I-organization': 24,
     'I-others': 26,
     'I-profession': 6,
     'I-time': 2,
     'O': 0,
     'PAD': 27}
    ```
4. 把文章(一段字串)切成字元(character)
    ```
    before = '醫師:啊回去還好嗎?'
    after = '醫','師',':','啊','回','去','還','好','嗎',''?'
    ```
6. 把文章切成好幾個句子，以"。"為分界，總共 200 篇文章可以切出 20379 個句子。
    ```python
    print(all_chunks_text[0])
    ['醫', '師', ':', '啊', '回', '去', '還', '好', '嗎', '?', '民', '眾', ':', '欸', ',', '還', '是', '虛', '虛', '的', ',', '但', '。']
    
    ```

    ```
    句子分布:
    
    50> len: 19016
    100> len >= 50: 1190
    100> len >= 50: 127
    200> len >= 150: 26
    250> len >= 200: 11
    len >= 250: 9
    ```
6. 將句子切成長度只剩 250，若句子長度超過250的話就取前250個字。
7. 使用 bert_tokenizer 將文字轉成轉成 id，還有建立 attention_mask (建立 bert 輸入格式)
8. 分割 train / test 資料
9. 建立 dataloader 以丟入模型訓練

#### 三、丟入模型訓練
**模型架構:**
使用 huggingface 的 pretrained model。
但是 loss function 的部分，使用 torchcrf 的 CRF funtion。
```python=
class Bert_CRF(nn.Module):
    def __init__(self, output_dim):
        super(Bert_CRF, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese",output_attentions = False, output_hidden_states = False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, output_dim)
        self.crf = CRF(output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)
        return emission
```
**超參數:**
optimizer: AdamW
Learning rate: 3e-5
scheduler = get_linear_schedule_with_warmup
num_warmup_steps = total_steps * 0.1
batch size: 32

```python=
FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(mymodel.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(mymodel.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)
```
```python=
from transformers import get_linear_schedule_with_warmup

epochs = 10
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    #num_warmup_steps=0,
    num_warmup_steps=total_steps*0.1,
    num_training_steps=total_steps
)
```
最後一個epoch數據:
![](https://i.imgur.com/dgpR4MQ.png)

各類別 f1 score:
![](https://i.imgur.com/RR6i4lX.png)

Learning curve:
![](https://i.imgur.com/Q5k3fFb.png)

F1 score per epoch:
![](https://i.imgur.com/fR9Rckf.png)


預測結果:
丟入以下句子進行預測查看效果:
>醫師：榮總醫院在十月21日於泰山區辦老人免費健檢活動。民眾:好，我明天會去看看，謝謝。

```
[CLS] O
醫 O
師 O
: O
榮 B-location
總 I-location
醫 I-location
院 I-location
在 O
十 B-time
月 I-time
2 I-time
1 I-time
日 I-time
於 O
泰 B-location
山 I-location
區 I-location
辦 O
老 O
人 O
免 O
費 O
健 O
檢 O
活 O
動 O
。 O
民 O
眾 O
: O
好 O
, O
我 O
明 B-time
天 I-time
會 O
去 O
看 O
看 O
, O
謝 O
謝 O
。 O
[SEP] O
```
#### 四、輸出結果至csv
```
# 例子

[article_id = 0]
start_pos: [198, 227, 237]
3
end_pos: [200, 229, 239]
3
entity_text: ['新樓', '麻豆', '麻豆']
3
entity_type: ['location', 'location', 'location']
3
```


## 實驗環境與結果
### 環境與套件
**實驗環境:**
本程式是在 google colab 上執行
1. GPU: Tesla T4
2. RAM: 12 GB
3. CPU: Intel(R) Xeon(R) CPU @ 2.30GHz

**使用套件:**
1. transformers
3. seqeval
4. pytorch-crf
5. torch
6. pandas
7. numpy
8. tqdm
9. keras
10. sklearn

### 實驗過程
為了快速進行實驗，在過程中我們準備了 2 種 data。
1. small-data: 去除只含有 O-tag 的句子。
2. all-data: 全部的句子

#### 一、增加 CRF 作為 loss function

    Data type: small-data
    Model: bert-base-chinese + crf
    epoch: 4
    batchsize: 32

![](https://i.imgur.com/owL2YMF.png)
>原本只用 bert 的情況，若只訓練 4 次，則 f1 score 是 66.48，
增加了 crf 作為 loss function 後，則進步到了 71.04，效果顯著。

**若增加訓練次數到 10 次:**

    Data type: small-data
    Model: bert-base-chinese + crf
    epoch: 10
    batchsize: 32

![](https://i.imgur.com/Cdc7Otw.png)
>使用 small data 訓練到 10 次後，結果反而從 71.05 下降到 68.22。
若有加 warm-up，則下降幅度會比較小，變成 68.71。

**使用全部資料(all-data):**

    Data type: all-data
    Model: bert-base-chinese + crf
    epoch: 10
    batchsize: 32
    
![](https://i.imgur.com/525r0rp.png)
>若使用 all-data 訓練 10 次，則 f1 score 上升到 76.64，是我們組最好的成績。

#### 二、 資料集使用 Sliding window

    Data type: small-data
    Model: bert-base-chinese + sliding window
    epoch: 4
    batchsize: 32
    window size: 25
    window step: 5

![](https://i.imgur.com/wDHEE9y.png)
>相比於單純用 bert 得到的 66.48，增加 sliding window 後 f1 score 進步到 70.18， 效果顯著。

**從 small-data 改成 full-data:**

    Data type: full-data
    Model: bert-base-chinese + sliding window
    epoch: 4
    batchsize: 32
    window size: 25
    window step: 5

![](https://i.imgur.com/GDRrsp6.png)
>若使用全部資料，一樣只 train 4次，則可以進步到 74.53，比起單純 bert 的 f1 score進步了約 8。
>比起 sliding window 使用 small-data train 4 次的 f1 score 進步了約 4 。

**若增加訓練次數到 10 次:**

    Data type: full-data
    Model: bert-base-chinese + sliding window
    epoch: 10
    batchsize: 32
    window size: 25
    window step: 5

![](https://i.imgur.com/fp5BRYn.png)
> 訓練 10 次後，f1 score 進步到 76.12，很逼近前面 bert + crf 訓練 10 次的結果。

#### 三、 CRF + Sliding window

    Data type: small-data
    Model: bert-base-chinese + sliding window + crf
    epoch: 4
    batchsize: 32
    window size: 25
    window step: 5

![](https://i.imgur.com/cDQQHxu.png)
> 將較於 bert + sliding wondow 的 70.17，bert + sliding wondow + crf 的 f1 僅提升至 71.76，進步幅度反而沒有那麼顯著了 (將比於 bert 與 bert + crf 的進步)，提升約 1.6。

**使用 full-data:**

    Data type: full-data
    Model: bert-base-chinese + sliding window + crf
    epoch: 4
    batchsize: 32
    window size: 25
    window step: 5
    
![](https://i.imgur.com/r6aB7zx.png)
> bert-base-chinese + sliding window + crf 使用全部資料跑 4 次後，進步幅度相比 bert-base-chinese + sliding window 進步約 1。

**若增加訓練次數到 10 次:**

    Data type: full-data
    Model: bert-base-chinese + sliding window + crf
    epoch: 10
    batchsize: 32
    window size: 25
    window step: 5
    
![](https://i.imgur.com/xqpeuIE.png)
> 訓練 10 次以後，進步幅度反而不大。比起單純用 bert + crf 的成績還低。

#### 四、 調整 Batch-size

    Data type: small-data
    Model: bert-base-chinese  + crf
    epoch: 4
    batchsize: 32 / 16 / 8
    
![](https://i.imgur.com/uRMfwmq.png)
> 在 small-data 的情況下， batchsize(bs)越小，f1 score 越高，且分割train set / dev set 時，若給越多 train，則 f1 score 的表現也越好。

**在 full-data 調整 batch-size:**

    Data type: full-data
    Model: bert-base-chinese  + crf
    epoch: 10
    batchsize: 8

![](https://i.imgur.com/ZBj73TO.png)
> 但是在 full-data 的情況下，把batch-size調小之後，反而表現變差。

#### 最終分數
由於經過幾次實驗後，發現 bert + crf 用 full-data 訓練 10 次所得到的f1 score  76.64 是最好的成績，因此就使用這個結果上傳。

**Public Leaderboard:**
26 / 174
![](https://i.imgur.com/tQKyNIY.png)

**Private Leaderboard:**
32 / 174
![](https://i.imgur.com/BJ9kBqO.png)

## 賽後檢討
有些在 small-data 的趨勢在 full-data 不一定有，但是要以 full-data 為主才對，因為 full-data 比較貼近實際的資料分布。當初區分出 small-data 與 full-data 只是為了讓訓練時間減少，但實際上對於調參數的幫助其實不是很大。
我們有得出幾點檢討:
1. 分出部分 (small-data) or 全部資料(all-data) 並無幫助
2. 可以訓練多個 model ( e.g. BERT / RoBERTa / Electra)，然後使用 bagging 投票
3. 不同類別分別訓練 (e.g. 模型專門辨識 time、location、data…)

## 心得
### N26091194 鄧立昌 心得 
這是我第一次參加 ai_cup，在比賽的過程中，我覺得透過比賽能夠提升很多自己的經驗，讓自己能更快上手一個新的技術，原本我連 NLP 是甚麼都沒有概念，但是因為要參加這次 AI_CUP，所以也看了蠻多相關的資料，就這樣從 0 開始累積相關知識，也是在找資料的過程中發現了 BERT 這個模型，本來只期待說程式能夠跑就好，沒想到做出來的結果還算不錯，比起機器學習的方法好上不少。

在過程中也會遇到許多問題，每次成功 debug 就會覺得自己又更進步了，或是在嘗試新方法時，看到結果有進步時都會感到收穫。比較印象深刻的是原本只有單純使用 bert，後來想幫 bert 再增加 CRF ，但是找了很多資料都沒有我想要的應用，最後才找到一個 pytorchcrf 這個套件，它的用法也蠻特別的，是把 crf 作為 loss function 來使用，原本對於這個套件是半信半疑，但是沒想到在使用 crf 後， f1 score 有顯著的增加，讓我感到很驚訝。 

總結來說我在這次比賽學習到了很多實務上的經驗，雖然最後結果並沒有到很前面，但對於我自己來說是一個很大的進步，之後還會想繼續參加 ai_cup 來磨練自己。

### N26092116 宋士莆 心得
參加比賽後才發現，就算看懂baseline，也不一定能改良baseline進而獲得好的結果，大部分時候還是需要外查閱其他相關文件，找尋新知識。再來是花最多時處理的通常是在fine-tuning的過程中，對數據有大膽的假設，但求證過程通常不是很理想，有可能是專業知識不夠，但在廣漠的文件中，要找關鍵因素本就也是不太容易發生的事情，總結來說比賽很有意思，但還是想多補充相關知識再繼續前行。

###### tags: `report`






