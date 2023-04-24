# Naive-Bayes-Algorithm-with-Python

Before we dive deeper into this topic we need to understand what is “Conditional probability”, what is “Bayes’ theorem” and how conditional probability help’s us in Bayes’ theorem.

## Conditional Probability for Naive Bayes

Conditional probability is defined as the likelihood of an event or outcome occurring, based on the occurrence of a previous event or outcome. Conditional probability is calculated by multiplying the probability of the preceding event by the updated probability of the succeeding, or conditional, event.

Consider a random experiment of tossing 2 coins. The sample space here will be:

S = {HH, HT, TH, TT}

If a person is asked to find the probability of getting a tail his answer would be 3/4 = 0.75

Now suppose this same experiment is performed by another person but now we give him the condition that both the coins should have heads. This means if event A: ‘Both the coins should have heads’, has happened then the elementary outcomes {HT, TH, TT} could not have happened. Hence in this situation, the probability of getting heads on both the coins will be 1/4 = 0.25

From the above examples, we observe that the probability may change if some additional information is given to us. This is exactly the case while building any machine learning model, we need to find the output given some features.

Mathematically, the conditional probability of event A given event B has already happened is given by:

conditional probability | Naive Bayes Algorithm 

https://editor.analyticsvidhya.com/uploads/530437.1.png![image](https://user-images.githubusercontent.com/13853670/233920194-7ade914a-7fc4-4b84-ad9f-ef563d68efeb.png)
 

## Bayes’ Rule

Now we are prepared to state one of the most useful results in conditional probability: Bayes’ Rule.

Bayes’ theorem which was given by Thomas Bayes, a British Mathematician, in 1763 provides a means for calculating the probability of an event given some information.

Mathematically Bayes’ theorem can be stated as:

 

bayes rule 
Basically, we are trying to find the probability of event A, given event B is true.

Here P(B) is called prior probability which means it is the probability of an event before the evidence

P(B|A) is called the posterior probability i.e., Probability of an event after the evidence is seen.

With regards to our dataset, this formula can be re-written as:

formula | Naive Bayes Algorithm 
Y: class of the variable

X: dependent feature vector (of size n)

Bayes rule use
Image Source: Author
