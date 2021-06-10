# END2.0_Assignment6

- Take the last code  (+tweet dataset) and convert that in such a way that:
- encoder: an RNN/LSTM layer takes the words in a sentence one by one and finally converts them into a single vector. VERY IMPORTANT TO MAKE THIS SINGLE VECTOR
this single vector is then sent to another RNN/LSTM that also takes the last prediction as its second input. Then we take the final vector from this Cell
and send this final vector to a Linear Layer and make the final prediction. 
This is how it will look:
- embedding
--word from a sentence +last hidden vector -> encoder -> single vector
--single vector + last hidden vector -> decoder -> single vector
--single vector -> FC layer -> Prediction


# Conceptual Understanding

- Encoder-It accepts a single element of the input sequence at each time step, process it, collects information for that element and propagates it forward.
- Context/Intermediate vector- This is the final internal state produced from the encoder part of the model. It contains information about the entire input sequence to help the decoder make accurate predictions.
- Decoder- given the entire sentence, it predicts an output at each time step.


# Model Architechture
![Model_Architecture](Model_Architecture.png)

- Encoder

class Encoder(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim):
    super().__init__()          
    
    # Embedding layer
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    # LSTM layer
    self.rnn = nn.LSTM(embedding_dim, 
                       hidden_dim,
                       #dropout=dropout,
                       batch_first=True)


  def forward(self, text, text_lengths,debug=False):
    
    # text = [batch size, sent_length]
    embedded = self.embedding(text)
    # embedded = [batch size, sent_len, emb dim]
  
    # packed sequence
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
    
    packed_output, (hidden_encoder, cell_encoder) = self.rnn(packed_embedded)

    encoder_output, encoder_output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    
    if debug:
      print(encoder_output)

    return encoder_output

- Decoder

class Decoder(nn.Module):

#Define all the layers used in model
  def __init__(self, vocab_size, encoder_output_dim, hidden_dim):
    
    super().__init__()          
    
    # LSTM layer
    self.rnn = nn.LSTM(encoder_output_dim, 
                       hidden_dim, 
                       batch_first=True)

    self.fc = nn.Linear(hidden_dim, output_dim)


  def forward(self, encoder_output,debug=False):
    
    output, (hidden_decoder, cell_decoder) = self.rnn(encoder_output)

    # Linear
    dense_outputs = self.fc(hidden_decoder)   
    
    if debug:
      print(dense_outputs)

    return dense_outputs

# Define hyperparameters
- vocab_len = len(Tweet.vocab)
- embedding_dim = 324
- hidden_dim = 144
- num_classes=3
- encoder_dim = 144
- debug=False
- encoder = Encoder(vocab_len,embedding_dim,hidden_dim)
- decoder = Decoder(vocab_len, 144,64)
- model = EncoderDecoder(encoder,decoder,16,num_classes)

# Accuracy logs

Train Loss: 1.070 | Train Acc: 68.11%
	 Val. Loss: 1.047 |  Val. Acc: 68.30% 

	Train Loss: 0.991 | Train Acc: 69.12%
	 Val. Loss: 0.908 |  Val. Acc: 68.30% 

	Train Loss: 0.870 | Train Acc: 69.12%
	 Val. Loss: 0.855 |  Val. Acc: 68.30% 

	Train Loss: 0.835 | Train Acc: 71.15%
	 Val. Loss: 0.840 |  Val. Acc: 71.43% 

	Train Loss: 0.814 | Train Acc: 76.56%
	 Val. Loss: 0.828 |  Val. Acc: 72.32% 

	Train Loss: 0.795 | Train Acc: 79.31%
	 Val. Loss: 0.814 |  Val. Acc: 72.77% 

	Train Loss: 0.766 | Train Acc: 81.59%
	 Val. Loss: 0.804 |  Val. Acc: 76.79% 

	Train Loss: 0.736 | Train Acc: 84.54%
	 Val. Loss: 0.780 |  Val. Acc: 81.25% 

	Train Loss: 0.703 | Train Acc: 86.66%
	 Val. Loss: 0.765 |  Val. Acc: 80.80% 

	Train Loss: 0.686 | Train Acc: 87.50%
	 Val. Loss: 0.756 |  Val. Acc: 80.80%

# Encoder Decoder Output

## Encoder Outputs

![encode](time_step0.PNG)
![encode](time_step1.PNG)
![encode](time_step2.PNG)
![encode](time_step3.PNG)
![encode](time_step4.PNG)
![encode](time_step5.PNG)
![encode](time_step6.PNG)
![encode](time_step7.PNG)
![encode](time_step8.PNG)
![encode](time_step9.PNG)
![encode](time_step10.PNG)
![encode](time_step11.PNG)
![encode](time_step12.PNG)
![encode](time_step13.PNG)
![encode](time_step14.PNG)

## Decoder Output

![decode](decod_0_1.PNG)
![decode](decod_2_3.PNG)
![decode](decod_4_5.PNG)
![decode](decod_6_7.PNG)
![decode](decod_8_9.PNG)
![decode](decod_10_11.PNG)
![decode](decod_12_13.PNG)
![decode](decod_14.PNG)

# Examples of correctly classified tweets

Tweet - A valid explanation for why Trump won't let women on the golf course.
Sentiment - Negative

# Examples of misclassified tweets

![missclassified](missclass_tweet.PNG)

# Training Validation loss

![loss](train_valid_loss.png)

# Training Validation accuracy

![loss](train_valid_acc.png)

Thank You 😁
