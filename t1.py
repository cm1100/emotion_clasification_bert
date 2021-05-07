import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json


import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks



load_data={}
load_data['df0'] = pd.read_csv("dataset1/train.txt",delimiter=";",names=["sentence","emotion"])
#print(df.head())
load_data['df1']= pd.read_csv("dataset1/val.txt",delimiter=";",names=["sentence","emotion"])
load_data['df2']=pd.read_csv("dataset1/test.txt",delimiter=";",names=["sentence","emotion"])

data={}

for i in range(3):
    data["train"+str(i)]=np.array(load_data["df"+str(i)]["sentence"])
    data["test"+str(i)]=np.array(load_data["df"+str(i)]["emotion"])

print(list(data))

data_to_tokenize = np.concatenate((data["train0"],data["train1"],data["train2"]))
print(len(data_to_tokenize))



# dictionary for getting meaning of a sequence

def preprocessing(data_tokenize):
    additional_filters = '-''""'
    token = Tokenizer(num_words=None,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' + additional_filters,
                      lower=True,
                      split=" ",
                      char_level=False,
                      oov_token="UNK",
                      document_count=0)

    token.fit_on_texts(data_tokenize)
    return token

token = preprocessing(data_to_tokenize)



new_data={}
for i in range(3):

    new_data["train"+str(i)]=token.texts_to_sequences(data["train"+str(i)])
    new_data["test"+str(i)]=data["test"+str(i)]

print(list(new_data))


for i in range(3):
    new_data["train"+str(i)] = np.array(pad_sequences(new_data["train"+str(i)],padding="pre",truncating="pre",maxlen=128))

tokenizer_config= token.get_config()

word_index = json.loads(tokenizer_config["word_index"])
index_word = {value:key for key,value in word_index.items()}


print(set(new_data["test0"]))

model = tf.keras.models.load_model("model_new")
print(model.summary())



