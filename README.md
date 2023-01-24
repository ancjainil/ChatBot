# Chatbot using seq2seq model

### Dataset

Cornell Movie Dialogue Corpus
``https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html``


Preprocessed data are uploaded in the directory.

*Current Stage* - Upto Training the model

#### Requirement

``Tensorflow==1.0``
``NumpPy==13.x``

## Input / Output

![image](https://user-images.githubusercontent.com/80884488/214215921-08c3ef43-62dc-41ae-ae94-433c6003d358.png)


## Flow:

![image](https://user-images.githubusercontent.com/80884488/214216321-72f7ed87-efec-4dbe-941a-b9aaafb83057.png)

## Seq2Seq model Working


The Sequence to Sequence model (seq2seq) consists of two RNNs — an encoder and a decoder. 
The encoder reads the input sequence, word by word and emits a context (a function of final hidden state of encoder), which would ideally capture the essence (semantic summary) of the input sequence. 
Based on this context, the decoder generates the output sequence, one word at a time while looking at the context and the previous word during each timestep. 
This is a ridiculous oversimplification, but it gives you an idea of what happens in seq2seq.




![image](https://user-images.githubusercontent.com/80884488/214216986-755e679d-5c92-44fc-8fc3-556dede90314.png)



## MethodoLogy
# Prepare the Dataset:
First we need to prepare the dataset. We had prepared the dataset of question for the subjects like Data Structures, Algorithms, Operating System. The dataset contain the question and answers of these subjects. The better the dataset, the more accurate and efficient conversational results can be obtained.

# Pre-Processing:
Lowercase all the charcters and remove unwanted charcter like - or # or $ etc.
Filter the dataset with max question length and max answer length Here we are use 20 for both qmax and amax.
Tokenization and Vectorization
Add zero padding
Split into train,validation,test data

# Creation of LSTM,Encoder and Decoder Model:
LSTM are a special kind of RNN which are capable of learning long-term dependencies. Encoder-Decoder model contains two parts- encoder which takes the vector representation of input sequence and maps it to an encoded representation of the input. This is then used by decoder to generate output.


# Train and Save Model:
I trained the model with 100 epochs and batch size of 32, Learning rate-0.001, word embedding size was set to 1024, we took categorical cross entropy as our loss function and optimiser used was AdamOptimizer. We got the best results with these parameters. Training accuracy obtained was approximately 99% and validation accuracy of about 80%.


# Testing:
Finally the user can input questions and bot will reply the answer of the Question. The results obtained are satisfactory according to review analysis.

