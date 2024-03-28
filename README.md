# Natural Language Processing Medical Question & Question Type

**Data Sciencist:** Elina Rankova

<div style="width: 100%; text-align: center;">
  <img src="https://altheia.com/wp-content/uploads/2020/12/0KbmpXTpCCIt1TR1B-1280x720.png" width="720" height="450" style="margin: 0 auto;"/>
</div>

<u>image source</u>: <a href="https://altheia.com/recommended-tools-for-data-scientists-in-the-medical-field/">Altheia; Recommended Tools for Data Scientists in the Medical Field</a>

## Business Problem and Understanding

**Stakeholders:** Healing Hearts Partners, Lead Receptionist, Lead Nurse, Lead Medical Assistant

Healing Hearts is growing and is noticing that patients are not recieving response to common questions regarding specific diseases in a timely manner. They are interseted in automating the Q&A process for both patients and the front desk staff who have a hard time distinguishing where to direct questions.

For Phase 1 of this Natural Language Processing (NLP) task, we will be classifying the questions based on question type, aiming to predict the question type of questions being asked of reception.

**The goal:** Create classification model that predicts question type from which reception can determine who is best equiped to answer patient, improving productivity and patient satisfaction.

## Data Understanding and Exploration

For this the <a href="https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset/code">Comprehensive Medical Q&A Dataset</a> sourced from Kaggle.

Our target will the be `qtype` or question type and we will try to match the `Question` column to be able to advice on appropriate next steps accoridng to the type of question being asked. Questions consist of common inquiries about specific diseases. 

### Observations
- No missingness to take care of
- 16407 records
- 16 total classes

**Question Example per Label** 
susceptibility :  Who is at risk for Lymphocytic Choriomeningitis (LCM)?
symptoms :  What are the symptoms of Lymphocytic Choriomeningitis (LCM)?
exams and tests :  How to diagnose Lymphocytic Choriomeningitis (LCM)?
treatment :  What are the treatments for Lymphocytic Choriomeningitis (LCM)?
prevention :  How to prevent Lymphocytic Choriomeningitis (LCM)?
information :  What is (are) Parasites - Cysticercosis?
frequency :  how common are these diseases for Marine Toxins?
complications :  are there complications from botulism?
causes :  What causes Chronic Fatigue Syndrome (CFS)?
research :  what research is being done for Tuberculosis (TB)?
outlook :  What is the outlook for Striatonigral Degeneration?
considerations :  What to do for Lactose Intolerance?
inheritance :  Is Ovarian Epithelial, Fallopian Tube, and Primary Peritoneal Cancer inherited?
stages :  What are the stages of Ovarian Epithelial, Fallopian Tube, and Primary Peritoneal Cancer?
genetic changes :  What are the genetic changes related to Chronic Myelogenous Leukemia?
support groups :  Where to find support for people with Alcohol Use and Older Adults?

**Class Imbalance**

Since we have some question types that have very few instances, we will have to drop that row before spliting our data if we want to keep class proportions. We will also have to consider our class imbalance when creating our models.

We can also use `LabelEncoder` to transform this variable before the train test split if we choose.

**Note:** There class `support groups` only has one record, if we plan to keep our class proportions when we plit our data, we will have to drop this class

<p align="center">
  <img src="https://github.com/erankova/Phase_4_Project/assets/155934070/3d551dac-d9f1-43bf-bebc-8aea9d27ebc6" alt="Class Count Chart">
</p>

### Preprocess Analysis

First, we should take a look at what the word frequency looks like before cleaning our data for things like stopwords and punctuation. It looks like we have a lot of stop words as well as other words that typically would be meaningful but are not in our medical context.

<p align="center">
  <img src="https://github.com/erankova/Phase_4_Project/assets/155934070/611af996-01b0-4f38-9cf7-26414e6ec83b" alt="Top 20 Raw Tokens">
</p>

Now let's see what our tolkens look like after removing stopwords by visualizing them in a wordcloud. It looks like some of the words from our raw tolken analysis made it into the word cloud giving us an idea of the features we might see after we vectorize.

<p align="center">
  <img src="https://github.com/erankova/Phase_4_Project/assets/155934070/ada2e525-7403-4e08-936d-2c2baa4223f2" alt="Wordcloud">
</p>

## Data Preparation

To help us normalize our text, we implemented a `TextPreprocessor class`
- Calls on `BaseEstimator` and `TransformerMixin` to be able to add class into pipeline
- Makes text lowercase
- Tokenizes the text and removes stop words
- Tags with parts of speech
- Lemmatizes the text with `WordNetLemmatizer`

Before we split our data, we first drop the `support groups` class which contains only one record since this interferes with our stratifying the split to keep class distributions consistant.

### Base Model Pipeline

As our base model, we will try a `MultinomialNB` model as it is great for text classification problems. Before fitting, we have to add the `TFidVectorizer` to our pipeline to vectorize our text with class weights in mind.

We set our minimum document frequency and maximum document frequency to .03 and .98 respectively to make sure that we don't penalize rare/frequent words too much but also don't loosen the thresholds too much and create noisy predictions.

![Base Model Pipeline](https://github.com/erankova/Phase_4_Project/assets/155934070/864fac0a-1824-4379-a51b-7145abf7c2b5)

_**We are left with 18 features after preprocessing our data in full**_

<ins>**Baseline Results**</ins>
Prior to hyperparameter tuning and cross validation we have:
- **F1 Score** = .86
- **Accuracy** = .90
- **Precision** = .92
- **Recall** = .90

Since we have a class imbalance we will be focusing on a **weighted F1 score** as our primary metric.

## Naive Bayes Models

Now that we have our base model, we can finetune our `MultiNomialNB` to see if we can improve our scores.

> For each subsequent model we try going forward we will utilize `RandomizedSearchCV` to tune our parameters and perform cross validation since it is less computationally costy and will save us some time while giving us direction about our model performance.

We also try a `ComplementNB` model since it is supposed to be great for class imbalance. After hyperparameter tuning and cross validation however, we see our tuned `MultiNomialNB` performs significantly better on our F1 score.

![MNB vs CNB Metrics](https://github.com/erankova/Phase_4_Project/assets/155934070/90a117f9-5f2f-43ef-a08f-b5c20311d572)

## Tree Models

Next we try some tree models, specifically the `DecisionTreeClassifier` and `RandomForestClassifier`.

After applying `RandomizedSearchCV` to both, we can see that our `DecisionTreeClassifier` gives us the best F1 score on unseen data so far! This could be due to the type of questions we have and their formulaic nature in relation to the `qtype`.

![DEC vs NB Metrics](https://github.com/erankova/Phase_4_Project/assets/155934070/f17e4af3-2eaa-4b0b-ac35-034b06a2f1c8)

## Boosting Models

Lastly, we try some boosting algorithms since cost sensitive learning could help us with or class imbalance.

First, we try the `GradientBoostingClassifier`, then `XGBoostingClassifier`, and lastly the `AdaBoostingClassfier. After hyperparamter tuning and cross validation, our boosting models do pretty well! 

Interestingly enough, the `DecisionTreeClassifier` still produces the best F1 score on unseen data! Close behind is the `AdaBoostingClassifier`, beating out the `GradientBoostingClassifier by a small amount. 

![DEC vs BOOSTING Metrics](https://github.com/erankova/Phase_4_Project/assets/155934070/47364970-09b1-4fa0-a9e0-44f3ae0d63c4)







