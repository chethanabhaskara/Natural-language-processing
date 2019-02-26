This is the implementation of a Naive Bayes model that performs Natural language processing to classify reviews. 
The reviews can classified as Positve or Negative. The model also has the capability to distinguish between truthful and deceptive reviews.

# Training 
The Naive bayes classifier was trained based on 960 reviews containing four categories of data - Truthful positive, Deceptive positive reviews and Truthful negative and truthful positive reviews. 

# Pre processing
Each review was converted to lower case, numbers and punctutations were eliminated. 
A list of stop words were further removed from the review sentences. 

# Classifier
The classifier is implemented as a multinomial naive bayes classifier that classifies any given review into the aforementioned classes. 

# Accuracy 
The model has an average F1_score of 0.85 over the development set used. 

