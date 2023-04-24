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

![image](https://user-images.githubusercontent.com/13853670/233920631-b4a699f4-bc01-40e0-b22e-695ae48db6f1.png)

## Bayes’ Rule


Now we are prepared to state one of the most useful results in conditional probability: Bayes’ Rule.

Bayes’ theorem which was given by Thomas Bayes, a British Mathematician, in 1763 provides a means for calculating the probability of an event given some information.

Mathematically Bayes’ theorem can be stated as:

 ![image](https://user-images.githubusercontent.com/13853670/233920765-d8a3cef7-37b4-4d7d-81de-5034b7e7c02a.png)


bayes rule 
Basically, we are trying to find the probability of event A, given event B is true.

Here P(B) is called prior probability which means it is the probability of an event before the evidence

P(B|A) is called the posterior probability i.e., Probability of an event after the evidence is seen.

With regards to our dataset, this formula can be re-written as:
![image](https://user-images.githubusercontent.com/13853670/233920940-e4d599a0-7ed7-4d61-8383-b5eaae35f4f5.png)


Y : class of the variable

X: dependent feature vector (of size n)

Bayes rule use
![image](https://user-images.githubusercontent.com/13853670/233921035-19cca12f-2d6f-45f4-b58c-584215f7fb24.png)

What is Naive Bayes?
Bayes’ rule provides us with the formula for the probability of Y given some feature X. In real-world problems, we hardly find any case where there is only one feature.

When the features are independent, we can extend Bayes’ rule to what is called Naive Bayes which assumes that the features are independent that means changing the value of one feature doesn’t influence the values of other variables and this is why we call this algorithm “NAIVE”

Naive Bayes can be used for various things like face recognition, weather prediction, Medical Diagnosis, News classification, Sentiment Analysis, and a lot more.

When there are multiple X variables, we simplify it by assuming that X’s are independent, so
![image](https://user-images.githubusercontent.com/13853670/233922758-7a344b2f-c114-40cc-a7b5-12ff7ad653f5.png)
For n number of X, the formula becomes Naive Bayes:

![image](https://user-images.githubusercontent.com/13853670/233922849-686c62cd-1736-4db8-a7eb-5ac3716d2769.png)
Which can be expressed as:
![image](https://user-images.githubusercontent.com/13853670/233922967-dcdc396f-3a2a-4bce-b032-3a6b009f6afb.png)

 Since the denominator is constant here so we can remove it. It’s purely your choice if you want to remove it or not. Removing the denominator will help you save time and calculations.
 ![image](https://user-images.githubusercontent.com/13853670/233923050-622ef86c-1c13-411c-b6ed-0bcbd47bdbad.png)
This formula can also be understood as:

![image](https://user-images.githubusercontent.com/13853670/233923125-e3385ee2-b14c-404c-a18f-57af5ac496f4.png)

There are a whole lot of formulas mentioned here but worry not we will try to understand all this with the help of an example.

## Naive Bayes Example
Let’s take a dataset to predict whether we can pet an animal or not.
![image](https://user-images.githubusercontent.com/13853670/233923202-4ad6aa6a-1d5f-4376-b376-94cd3ecb1e6f.png)

##### Assumptions of Naive Bayes
· All the variables are independent. That is if the animal is Dog that doesn’t mean that Size will be Medium

· All the predictors have an equal effect on the outcome. That is, the animal being dog does not have more importance in deciding If we can pet him or not. All the features have equal importance.

We should try to apply the Naive Bayes formula on the above dataset however before that, we need to do some precomputations on our dataset.

We need to find P(xi|yj) for each xi in X and each yj in Y. All these calculations have been demonstrated below:
![image](https://user-images.githubusercontent.com/13853670/233923362-c5b20162-c22e-49a1-9e58-06b3d0c9324b.png)

We also need the probabilities (P(y)), which are calculated in the table below. For example, P(Pet Animal = NO) = 6/14.
![image](https://user-images.githubusercontent.com/13853670/233923519-bdfd05fa-a7e1-4807-bb7e-e6cece5542c0.png)
Now if we send our test data, suppose test = (Cow, Medium, Black)

Probability of petting an animal :
![image](https://user-images.githubusercontent.com/13853670/233923576-327e6b64-4ffb-4423-a740-5900dac0da79.png)
![image](https://user-images.githubusercontent.com/13853670/233923675-37e003f1-ac09-41bc-93bf-4bd209e49eee.png)

And the probability of not petting an animal:
![image](https://user-images.githubusercontent.com/13853670/233923781-3f85d5f2-e4a8-4ea0-8b02-28eac58dbde4.png)
![image](https://user-images.githubusercontent.com/13853670/233923803-47e5863b-8975-4a55-bec1-48525ac218f4.png)

We know P(Yes|Test)+P(No|test) = 1

So, we will normalize the result:
![image](https://user-images.githubusercontent.com/13853670/233923947-ed41ca47-0ce2-42d6-963e-b74ae324d71a.png)
![image](https://user-images.githubusercontent.com/13853670/233923974-0d7f4343-a191-4bd9-9005-a2ee85244203.png)

We see here that P(Yes|Test) > P(No|Test), so the prediction that we can pet this animal is “Yes”.

## Gaussian Naive Bayes
So far, we have discussed how to predict probabilities if the predictors take up discrete values. But what if they are continuous? For this, we need to make some more assumptions regarding the distribution of each feature. The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of P(xi | y). Here we’ll discuss Gaussian Naïve Bayes.

Gaussian Naïve Bayes is used when we assume all the continuous variables associated with each feature to be distributed according to Gaussian Distribution. Gaussian Distribution is also called Normal distribution.

The conditional probability changes here since we have different values now. Also, the (PDF)  probability density function of a normal distribution is given by:
![image](https://user-images.githubusercontent.com/13853670/233924132-ac8a818f-6fd2-4f51-a538-35280c1a713f.png)
We can use this formula to compute the probability of likelihoods if our data is continuous.
