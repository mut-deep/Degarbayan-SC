# Degarbayan-SC: A Colloquial Paraphrase Farsi Subtitles Dataset
Paraphrase generation and detection are important tasks in Natural Language Processing (NLP), such as information retrieval, text simplification, question answering, and chatbots. The lack of comprehensive datasets in the Persian paraphrase is a major obstacle to progress in this area. Despite their importance, no large-scale corpus has been made available so far, given the difficulties in its creation and the intensive labor required. In this paper, the construction process of Degarbayan-SC uses movie subtitles. As you know, movie subtitles are in Colloquial language. It is different from formal language.  To the best of our knowledge, Degarbayan-SC is the first freely released large-scale (in the order of a million words) Persian paraphrase corpus. Furthermore, this newly introduced dataset will help the growth of Persian paraphrase. 
| dataset | Number of pair sentences | Date modified | details
| :---: | :---: | :---: | :---: |
| PPDB | 100M | Version.1 2013  Version.2 2015 | Phrasal paraphrases are extracted via bilingual pivoting |
| Wiki answer | 18M | 2014 | Paired questions that the users of the wikianswer website considered similar and paired them |
| MSCOCO | 500K | 2014 | Based on the annotation of 238K photos in 91 classes by 5 people |
| QQP | 400K | 2017 | Based on the Kaggle competition (identifying similar questions) |
| ParaNMT-50 | 50M | 2018 | Sentential paraphrase pairs are generated automatically by using neural machine translation |
| **ours** | **1.5M** | **2022** | **Based on aligning sentences in hundreds of movie subtitles** |
 
 
## Dataset

### Access and Download
You can find the dataset under this link of [Google Drive](https://drive.google.com/file/d/1-0B-t9MISKmymaBn88ay4EUS1X7awwLL/view?usp=sharing).
- Dataset is in .csv format
- our dataset has 2 columns the first column is for source sentences and the second is for targets.

Alternatively, you can also access the data through the HuggingFace🤗 datasets library.
For that, you need to install datasets using this command in your terminal:
(We will share it on HuggingFace after our paper is published)
```sh
pip install -q datasets
```

Afterward, import the `Degarbayan-SC` dataset using `load_dataset`:

```python
from datasets import load_dataset
dataset = load_dataset("m0javad/Degarbayan-SC-dataset")
```
or you can **fine-tune** the model using 'transformers': 
```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("m0javad/Degarbayan-SC")
model = AutoModelForSequenceClassification.from_pretrained("m0javad/Degarbayan-SC")
```
or you can **test** the fine-tuned model using 'pipeline': 
```python
from transformers import pipeline
pipe = pipeline("text2text-generation", model="m0javad/Degarbayan-SC")
```
### Statistic
![Lenght of sentences](https://i.ibb.co/C1RJhTZ/lenght.jpg")

our sentence length distribution is between 3 and 19 words and sentences are an average of 8 words. This makes sense because in the movie subtitles, sentences are shown in a range of times and we matched them with timespans. Humans can say a certain number of words in a certain period. Our collected sentences have 128,699 unique words.

### Examples

| Source sentence | Target sentence |
| ---: | ---: |
| باقی زندگی بچه هاتو تغییر میده | این تصمیم قراره زندگی بچه هاتو عوض کنه| 
| خب انگار که داریم انجامش میدیم | فکر کنم داریم این کارو می‌کنیم |
| به من بگو کی پشت این جریانه | میخوای به من بگی که کی پشت همه ایناست | 
| بیدار شو   باهات کار دارم | از اون تو بیا بیرون رفیق بهت نیاز دارم |
|به هر حال  این سرزمین مال اونه | بااینکه این سرزمین متعلق به اونه  | 
|باک  ما الان همه با هم گره خوردیم |  باک  ما همگی بهم وصل هستیم| 

as you see in the table above, our dataset contains a large number of paraphrasing sentences in various forms such as syntactic, semantic, and conceptual paraphrases.

### contact

contact me for contribution and future possible works at: mjaghajani.ai@gmail.com

### acknowledgment

I would like to thank my dear teacher Dr.Keyvanrad and my colleagues, [Zahra Ghasemi](https://github.com/ZahraGhasemi-AI), [Ali sadeghian](https://github.com/alisdnn)
