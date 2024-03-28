# Natural Language Processing Medical Question & Question Type

**Data Sciencist:** Elina Rankova

 <p align="center"><img src="https://altheia.com/wp-content/uploads/2020/12/0KbmpXTpCCIt1TR1B-1280x720.png" width="720" height="450" style="margin: 0 auto;"/></p>

<u>image source</u>: <a href="https://altheia.com/recommended-tools-for-data-scientists-in-the-medical-field/">Altheia; Recommended Tools for Data Scientists in the Medical Field</a>

## Business Problem and Understanding

**Stakeholders:** Healing Hearts Partners, Lead Receptionist, Lead Nurse, Lead Medical Assistant

Healing Hearts medical practice is growing and is noticing that patients are not recieving response to common questions regarding specific diseases in a timely manner. They are interested in automating the Q&A process for both patients and the front desk staff who have a hard time distinguishing where to direct questions.

For Phase 1 of this Natural Language Processing (NLP) task, we will be classifying the questions based on question type, aiming to predict the question type of questions being asked by patients.

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

Since we have some question types that have very few instances, we will also have to consider our class imbalance when creating our models.

We can also use `LabelEncoder` to transform this variable before the train test split if we choose.

**Note:** There class `support groups` only has one record, if we plan to keep our class proportions when we split our data, we will have to drop this class

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

We set our minimum document frequency and maximum document frequency to .03 and .98 respectively to make sure that we don't penalize rare/frequent words too much but also don't loosen the thresholds so much that we create noisy predictions.

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

> For each subsequent model we try going forward, we will utilize `RandomizedSearchCV` to tune our parameters and perform cross validation since it is less computationally costy than `GridSearchCV` and will save us some time while giving us direction about our model performance.

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

## Best Model Evaluation

Since our DecisionTree and AdaBoost are so close in F1 scores, we should compare the two further to make sure we are making wholistic decision when it comes to our predictive model.

When we look at the feature importances of both models, the similarity between the two is further confirmed. They are both demonstrating the same top 10 features at very similar levels!

<p align="center">
  <img src="https://github.com/erankova/Phase_4_Project/assets/155934070/aeb0634d-980b-4c95-a6dd-7d63d761ef15" alt="Final Models Feature Importances">
</p>

Looking at our classification report we can tell that the `AdaBoostingClassifier` while doing mostly as well as the `DecisionTreeClassifier`, it has better scores for accuracy, recall, and precision. This means that it is balancing false negatives and false positives well and producing largly accurate predictions for us!

<div style="display: flex; justify-content: center;">
  <img src="https://github.com/erankova/Phase_4_Project/raw/main/assets/155934070/69ea77ce-0ce0-4771-82d1-9a6c627b0064.png" alt="DecisionTree Classification Report" style="width: 50%;">
  <img src="https://github.com/erankova/Phase_4_Project/raw/main/assets/155934070/eefe984f-cab5-4b25-83d2-daff027f4342.png" alt="AdaBoost Classification Report" style="width: 50%;">
</div>

## Final Evaluation & Conclusion

Based on a wholistic evaluation, we can conclude that the `AdaBoostingClassifier` provides us with the best results and overall most reliable predictions. This model not only has a comprable F1 score to our `DecisionTreeClassifier` but does better in the other relevant metrics, making it the overall best choice.

**Recommendations:**

For _<ins>Phase 1</ins>_ of this NLP task, we recommend implementing the model for the front desk staff, where the Healing Hearts team can coordinate around a process that pairs predicted class with the corresponding best professional equipped to answer these patient inquiries. This process can be implemented via phone, email, and SMS - routing the questions based on their predicted subject.

**Positive Implications:** 

Since the metrics are strong, we can be confident in routing questions to the right person, making the patient care more efficient for both the office staff and patients alike.

**Negative Implications:**

The model is not perfect, and there will be the rare occasion that a question is paired with the wrong class. In such cases, the office staff should align on protocol so that patients are not provided misleading medical advice. Data on these instances should also be collected for future iterations.

**Data Limitation and Future Considerations:**

As seen in our exploration, our dataset is limited in record number as well as the amount of records available in each class. For future improvements, we would recommend to collect more data on both questions, answers, and associated categories to further improve model reliability.

In addition, by increasing the amount of available data, our model can afford to add more strict constraints on minimum and maximum document frequency of tokens. This will make our predictions stronger and create a strong baseline to expand on how we use this information.

Ultimately, Healing Hearts would like to implement a Retrieval-Augmented Generation (RAG) chatbot. This chatbot would enable patients and potentially medical staff to not only recieve the general category of the questions being asked, but an answer to those questions well, all based on similarity metrics created from existing data. Since Healing Hearts is first and foremost concerned about their patients satisfaction and wellbeing, this chatbot can be implemented where appropriate and provide an option to speak to a person if the patient so chooses.


