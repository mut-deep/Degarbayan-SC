#T5 Fine tuning with PyTorch
#install modules
## install sentencepiece
## install transformers
## install rich[jupyter]

# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

from rich.table import Column, Table
from rich import box
from rich.console import Console

# define a rich console logger
console=Console(record=True)

def display_df(df):
  """display dataframe in ASCII format"""

  console=Console()
  table = Table(Column("source_text", justify="center" ), Column("target_text", justify="center"), title="Sample Data",pad_edge=False, box=box.ASCII)

  for i, row in enumerate(df.values.tolist()):
    table.add_row(row[0], row[1])

  console.print(table)

training_logger = Table(Column("Epoch", justify="center" ), 
                        Column("Steps", justify="center"),
                        Column("Loss", justify="center"), 
                        title="Training Status",pad_edge=False, box=box.ASCII)

#import and preprocess dataset
parphrases = pd.read_csv("Pad to Dataset")
## for delete nan values
dataa = parphrases.values.tolist()
h = []
for i in range(0,len(dataa)):
  dat = dataa[i]
  try:
    if 'NoneType' in dat[0] or 'NoneType' in dat[1]:
      print(i)
  except:
    h.append(i)
    continue
print(h)
m = 0
for ha in h:
  ha = ha-m
  del(dataa[ha])
  m += 1
#creat new dataset representation
da = []
for i in range(0,len(dataa)):
  add = []
  dat = dataa[i]
  text1 = da[0]
  if bool(re.findall('([ادرزژوذ][\u200c])', text1)) is True :
   text1 = re.sub(r"[\u200c]+", '', text1)
  if bool(re.findall('([^ادرزژوذ][\u200c])', text1)) is True:
   text1 = re.sub(r"[\u200c]+", ' ', text1) 
  text2 = da[1]
  if bool(re.findall('([ادرزژوذ][\u200c])', text2)) is True :
   text2 = re.sub(r"[\u200c]+", '', text2)
  if bool(re.findall('([^ادرزژوذ][\u200c])', text2)) is True:
   text2 = re.sub(r"[\u200c]+", ' ', text2) 
  add.append(text1)
  add.append(text2)
  da.append(add)
parphrases = pd.DataFrame(da)

#change dataset format to initialize the Model
parphrases = parphrases.rename(columns={1:"text"})
parphrases = parphrases.rename(columns={0:"headlines"})
parphrases["text"] = "paraphrase: "+parphrases["text"]

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

"""
  Creating a custom dataset for reading the dataset and 
  loading it into the dataloader to pass it to the neural network for finetuning the model
  """
class YourDataSetClass(Dataset):
  
  def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
    self.tokenizer = tokenizer
    self.data = dataframe
    self.source_len = source_len
    self.summ_len = target_len
    self.target_text = self.data[target_text]
    self.source_text = self.data[source_text]

  def __len__(self):
    return len(self.target_text)

  def __getitem__(self, index):
    source_text = str(self.source_text[index])
    target_text = str(self.target_text[index])

    #cleaning data so as to ensure data is in string type
    source_text = ' '.join(source_text.split())
    target_text = ' '.join(target_text.split())

    source = self.tokenizer.batch_encode_plus([source_text], max_length= self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
    target = self.tokenizer.batch_encode_plus([target_text], max_length= self.summ_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

    source_ids = source['input_ids'].squeeze()
    source_mask = source['attention_mask'].squeeze()
    target_ids = target['input_ids'].squeeze()
    target_mask = target['attention_mask'].squeeze()

    return {
        'source_ids': source_ids.to(dtype=torch.long), 
        'source_mask': source_mask.to(dtype=torch.long), 
        'target_ids': target_ids.to(dtype=torch.long),
        'target_ids_y': target_ids.to(dtype=torch.long)
    }
     

"""
  Function to be called for training with the parameters passed from main function
  """
def train(epoch, tokenizer, model, device, loader, optimizer):

  model.train()
  for _,data in enumerate(loader, 0):
    y = data['target_ids'].to(device, dtype = torch.long)
    y_ids = y[:, :-1].contiguous()
    lm_labels = y[:, 1:].clone().detach()
    lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
    ids = data['source_ids'].to(device, dtype = torch.long)
    mask = data['source_mask'].to(device, dtype = torch.long)

    outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
    loss = outputs[0]

    if _%10==0:
      training_logger.add_row(str(epoch), str(_), str(loss))
      console.print(training_logger)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


"""
  Function to evaluate model for predictions
  """
def validate(epoch, tokenizer, model, device, loader):

  
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=100, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=False
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          if _%10==0:
              console.print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals


"""
  T5 trainer
  """
def T5Trainer(dataframe, source_text, target_text, model_params, output_dir="./outputs/" ):
  
  # Set random seeds and deterministic pytorch for reproducibility
  torch.manual_seed(model_params["SEED"]) 
  # pytorch random seed^^
  np.random.seed(model_params["SEED"]) 
  # numpy random seed^^
  torch.backends.cudnn.deterministic = True

  # logging
  console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

  # tokenzier for encoding the text
  tokenizer = AutoTokenizer.from_pretrained(model_params["MODEL"])

  # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
  # Further this model is sent to device (GPU/TPU) for using the hardware.
  model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
  model = model.to(device)
  
  # logging
  console.log(f"[Data]: Reading data...\n")

  # Importing the raw dataset
  dataframe = dataframe[[source_text,target_text]]
  display_df(dataframe.head(2))

  
  # Creation of Dataset and Dataloader
  # Defining the train size. So 80% of the data will be used for training and the rest for validation. 
  train_size = 0.999
  train_dataset=dataframe.sample(frac=train_size,random_state = model_params["SEED"])
  val_dataset=dataframe.drop(train_dataset.index).reset_index(drop=True)
  train_dataset = train_dataset.reset_index(drop=True)

  console.print(f"FULL Dataset: {dataframe.shape}")
  console.print(f"TRAIN Dataset: {train_dataset.shape}")
  console.print(f"TEST Dataset: {val_dataset.shape}\n")


  # Creating the Training and Validation dataset for further creation of Dataloader
  training_set = YourDataSetClass(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
  val_set = YourDataSetClass(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)


  # Defining the parameters for creation of dataloaders
  train_params = {
      'batch_size': model_params["TRAIN_BATCH_SIZE"],
      'shuffle': True,
      'num_workers': 0
      }


  val_params = {
      'batch_size': model_params["VALID_BATCH_SIZE"],
      'shuffle': False,
      'num_workers': 0
      }


  # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
  training_loader = DataLoader(training_set, **train_params)
  val_loader = DataLoader(val_set, **val_params)


  # Defining the optimizer that will be used to tune the weights of the network in the training session. 
  optimizer = torch.optim.Adam(params =  model.parameters(), lr=model_params["LEARNING_RATE"])


  # Training loop
  console.log(f'[Initiating Fine Tuning]...\n')

  for epoch in range(model_params["TRAIN_EPOCHS"]):
      train(epoch, tokenizer, model, device, training_loader, optimizer)
      
  console.log(f"[Saving Model]...\n")
  #Saving the model after training
  path = os.path.join(output_dir, "model_files")
  model.save_pretrained(path)
  tokenizer.save_pretrained(path)


  # evaluating test dataset
  console.log(f"[Initiating Validation]...\n")
  for epoch in range(model_params["VAL_EPOCHS"]):
    predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    final_df.to_csv(os.path.join(output_dir,'predictions.csv'))
  
  console.save_text(os.path.join(output_dir,'logs.txt'))
  
  console.log(f"[Validation Completed.]\n")
  console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")
  console.print(f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n""")
  console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")

model_params={
    "MODEL":"erfan226/persian-t5-paraphraser",  # model_type: erfan226/persian-t5-paraphraser
    "TRAIN_BATCH_SIZE":32,         # training batch size
    "VALID_BATCH_SIZE":32,         # validation batch size
    "TRAIN_EPOCHS":3,              # number of training epochs
    "VAL_EPOCHS":1,                # number of validation epochs
    "LEARNING_RATE":1e-4,          # learning rate
    "MAX_SOURCE_TEXT_LENGTH":100,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH":50,   # max length of target text
    "SEED": 42                     # set seed for reproducibility 
}

# for vacant cuda cache to run better
import torch
torch.cuda.empty_cache()

# set model name and tokenizer
from transformers import T5ForConditionalGeneration,AutoTokenizer
model_name = "erfan226/persian-t5-paraphraser"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#FineTuning the model with our dataset
hist = T5Trainer(dataframe=parphrases, source_text="text", target_text="headlines", model_params=model_params, output_dir="pad to output Direction")

#loss chart 
loss = pd.DataFrame(
    {"train loss": hist.history["loss"], "test loss": hist.history["val_loss"]}
).melt()
loss["epoch"] = loss.groupby("variable").cumcount() + 1
sns.lineplot(x="epoch", y="value", hue="variable", data=loss).set(
    title="Model loss",
    ylabel="",
    xticks=range(1, loss["epoch"].max() + 1),
    xticklabels=loss["epoch"].unique(),
)







# Evaluate the fine tuned Model 
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

def set_seed(seed):
  torch.manual_seed(seed)
#  if torch.cuda.is_available():
#   torch.cuda.manual_seed_all(seed)
set_seed(42)
best_model_path = "Pad to out Direction that created before"
model = T5ForConditionalGeneration.from_pretrained(best_model_path)
from transformers import T5ForConditionalGeneration,AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(best_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)


#get sentences and see how this Network work
sents=["Your entrance sentences divided by , among each sentence"]
w = 0 
for sentt in sents:
  text =  "paraphrase: " + sentt
  max_len = 100
  encoding = tokenizer.encode_plus(text,padding='longest', return_tensors="pt")
  input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
  # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
  beam_outputs = model.generate(
      input_ids=input_ids, attention_mask=attention_masks,
      do_sample=True,
      max_length=120,
      top_k=120,
      top_p=0.98,
      early_stopping=True,
      num_return_sequences=1
  )
  print ("\nOriginal sentence: ")
  print (sentt)
  print ("\n")
  print ("Paraphrased sentences: ")
  final_outputs =[]
  for beam_output in beam_outputs:
      sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
      if sent.lower() != sentt.lower() and sent not in final_outputs:
          final_outputs.append(sent)
  for i, final_output in enumerate(final_outputs):
      print("{}: {}".format(i, final_output))


#be payan amad in daftar hekayat hamchenan baghiist :)