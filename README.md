
# Hi, I'm Ibrahim! 👋 and this is a
Machine Learning Project on Detecting Fake News

  
## Dataset
The Dataset used is obtained from Kaggle. It is uploaded by the user "jainpooja". 
The dataset title is: "Fake News Detection".
Dataset Link:

 - [Kaggle Fake News Detection](https://www.kaggle.com/jainpooja/fake-news-detection)

## 🛠 Packages Used
pandas, numpy, sklearn
## Text Preprocessing
### Text Cleaning Function
```bash
def preprocess(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text
```
### Feature Extraction
Bag-Of-Words TF-IDF


## Machine Learning Models
**Logistic Regression**

**Decision Tree**

**Ensemble Methods**

- Random Forest
- Gradient Boositng 

  
## Authors

- [@96ibman](https://www.github.com/96ibman)

  
## 🚀 About Me
Ibrahim M. Nasser, a Software Engineer, Usability Analyst, 
and a Machine Learning Researcher.


  
## 🔗 Links
[![GS](https://img.shields.io/badge/-Google%20Scholar-blue)](https://scholar.google.com/citations?user=SSCOEdoAAAAJ&hl=en&authuser=2/)

[![linkedin](https://img.shields.io/badge/-Linked%20In-blue)](https://www.linkedin.com/in/ibrahimnasser96/)

[![Kaggle](https://img.shields.io/badge/-Kaggle-blue)](https://www.kaggle.com/ibrahim96/)

  
## Contact

96ibman@gmail.com

  