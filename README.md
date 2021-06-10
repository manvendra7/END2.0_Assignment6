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

# Model Architechture

- Encoder

- Decoder

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

# Examples of correctly classified tweets

# Examples of misclassified tweets

