# Degarbayan-SC: A Colloquial Paraphrase Farsi Subtitles Dataset
Paraphrase generation and paraphrase detection are important tasks in Natural Language Processing (NLP), such as information retrieval, text simplification, question answering, and chatbots. The lack of comprehensive datasets in the Persian paraphrase is a major obstacle to progress in this area. In spite of their importance, no large-scale corpus has been made available so far, given the difficulties in its creation and the intensive labor required. In this paper, the construction process of Degarbayan-SC using movie subtitles. As you know, movie subtitles are in Colloquial language. It is different from formal language.  To the best of our knowledge, Degarbayan-SC is the first freely released large-scale (in the order of a million words) Persian paraphrase corpus. Furthermore, this newly introduced dataset will help the growth of Persian paraphrase. 
| dataset | Number of pair sentences | Date modified | details
| :---: | :---: | :---: | :---: |
| PPDB | 100M | Version.1 2013  Version.2 2015 | Phrasal paraphrases are extracted via bilingual pivoting |
| Wiki answer | 18M | 2014 | Paired questions that the users of the wikianswer website considered similar and paired them |
| MSCOCO | 500K | 2014 | Based on the annotation of 238K photos in 91 classes by 5 people |
| QQP | 400K | 2017 | Based on the Kaggle competition (identifying similar questions) |
| ParaNMT-50 | 50M | 2018 | sentential paraphrase pairs are generated automatically by using neural machine translation |
| **ours** | **1.5M** | **2022** | **Based on aligning sentences in hundreds of movie subtitles** |
 
 - our dataset has 2 columns that the first column is for source sentences and the second is for target sentences
 - 
## Dataset

### Access and Download
You can find the dataset under this link of [Google Drive](https://drive.google.com/file/d/1-0B-t9MISKmymaBn88ay4EUS1X7awwLL/view?usp=sharing).

Alternatively, you can also access the data through the HuggingFace🤗 datasets library.
For that, you need to install datasets using this command in your terminal:

```sh
pip install -q datasets
```

Afterwards, import `persian_qa` dataset using `load_dataset`:

```python
from datasets import load_dataset
dataset = load_dataset("m0javad/Degarbayan-SC")
```
### Examples



### Statistic
![Lenght 0f sentences](https://ibb.co/DV0G5Sy)


