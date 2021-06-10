# END2.0_Assignment6

- Take the last code  (+tweet dataset) and convert that in such a war that:
- encoder: an RNN/LSTM layer takes the words in a sentence one by one and finally converts them into a single vector. VERY IMPORTANT TO MAKE THIS SINGLE VECTOR
this single vector is then sent to another RNN/LSTM that also takes the last prediction as its second input. Then we take the final vector from this Cell
and send this final vector to a Linear Layer and make the final prediction. 
This is how it will look:
- embedding
--word from a sentence +last hidden vector -> encoder -> single vector
--single vector + last hidden vector -> decoder -> single vector
--single vector -> FC layer -> Prediction
- Your code will be checked for plagiarism, and if we find that you have copied from the internet, then -100%. 
