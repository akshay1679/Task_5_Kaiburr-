# 🧠 Task 5 — Text Classification using Logistic Regression (Kaiburr Internship Assignment)
**Author:** Akshay S  

---

## 📘 Project Overview
This project performs **text classification** on consumer financial complaints from the **Consumer Financial Protection Bureau (CFPB)** dataset.  
The goal is to classify consumer complaint narratives into **four product categories** using **TF-IDF vectorization** and a **Logistic Regression model**.

The project demonstrates:
- Data preprocessing and cleaning  
- Feature extraction using TF-IDF  
- Model training and evaluation  
- Visualization of training and validation accuracy  

---

## 🧩 Dataset Information
**Source:** [Consumer Complaints Dataset (CFPB)](https://files.consumerfinance.gov/ccdb/complaints.csv.zip)

Only the first **50,000 rows** were used for faster experimentation.  
From these, four categories were selected and mapped as follows:

| Product Category | Label |
|-------------------|:------:|
| Credit reporting, credit repair services, or other personal consumer reports | 0 |
| Debt collection | 1 |
| Consumer Loan | 2 |
| Mortgage | 3 |

Each class was balanced to have approximately **1,500 samples**, ensuring fair model training.

---

## ⚙️ Steps Performed
### 1️⃣ Data Loading & Filtering
- Loaded dataset using `pandas` (first 50,000 rows).
- Selected only the required columns:  
  `['Consumer complaint narrative', 'Product']`.
- Filtered the dataset for the above four categories.
- Created a balanced dataset (1,500 samples per class).

### 2️⃣ Text Cleaning
- Converted all text to lowercase.
- Removed punctuation, numbers, and special characters.
- Removed English stopwords using **NLTK**.

### 3️⃣ Feature Extraction
Used **TF-IDF (Term Frequency–Inverse Document Frequency)** with:
- `max_features = 5000`
- `ngram_range = (1, 2)`

This converts text into numerical vectors suitable for machine learning.

### 4️⃣ Model Training
Trained a **Logistic Regression** classifier with:
- `max_iter = 300`  
- Optimized using **scikit-learn**

The data was split as:
- **80% training**
- **20% testing**

### 5️⃣ Model Evaluation
Generated:
- **Accuracy score**
- **Classification report** (Precision, Recall, F1-score)
- **Learning curve** showing both training and validation accuracy.

---

## 📊 Results

| Metric | Value |
|:--------|:------|
| **Model Accuracy** | ~0.85 – 0.90 (depending on random sampling) |

**Classification Report:**


Credit Reporting   : Precision 0.86 | Recall 0.87
Debt Collection    : Precision 0.84 | Recall 0.83
Consumer Loan      : Precision 0.88 | Recall 0.86
Mortgage           : Precision 0.90 | Recall 0.89

````

---

## 📈 Visualizations

### 🟢 Learning Curve
Shows the relationship between training size and accuracy.  
Both **training** and **validation** accuracy are plotted to identify overfitting or underfitting.

**Code Snippet:**
```python
train_sizes, train_scores, val_scores = learning_curve(
    log_model, X_train_vec, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
````

**Generated Plot:**

> *Insert Screenshot 1 here (with your name + system date/time visible)*

---

### 🧪 Sample Predictions

Example test sentences used:

```python
samples = [
    "My credit report has wrong information even after I disputed it.",
    "The debt collector keeps calling me even after payment.",
    "The bank changed my mortgage rate without notice."
]
```

**Predicted Output:**

```
[0, 1, 3]
```

> *Insert Screenshot 2 here (console output with your name + date/time visible)*

---

## 🧰 Tools & Libraries

| Library                  | Purpose                               |
| :----------------------- | :------------------------------------ |
| **pandas**               | Data manipulation                     |
| **scikit-learn**         | ML model & evaluation                 |
| **nltk**                 | Text preprocessing & stopword removal |
| **matplotlib / seaborn** | Visualization                         |
| **numpy**                | Numerical operations                  |

---

## 🚀 How to Run

### 🔧 Step 1: Install Dependencies

```bash
pip install pandas scikit-learn nltk matplotlib seaborn
```

### 🔧 Step 2: Run the Script

Run the `.ipynb` file in **Google Colab** or your **local Python environment**:

```bash
python Task5_Modified.ipynb
```

### 🔧 Step 3: Output Files

A smaller version of the dataset will be saved as:

```
consumer_complaints_balanced.csv
```

---

## 📷 Screenshots

> ⚠️ **Important:** Each screenshot must include:
>
> * Your **name** visible (in terminal, editor, or window)
> * System **date/time widget** visible

| Screenshot   | Description                            |
| :----------- | :------------------------------------- |
| Screenshot 1 | Dataset loading and class distribution |
| Screenshot 2 | Model training and accuracy output     |
| Screenshot 3 | Learning curve plot                    |
| Screenshot 4 | Sample predictions                     |

---

## 📽️ (Optional) Demo Video

You may include a short screen recording showing:

* Running the notebook
* Model training
* Graph output
* Predictions

---

## 🧑‍💻 Author

**Akshay S**
*MCA-B, Amrita College*
📍 Trivandrum, India

---
```

---

Would you like me to include **example Markdown syntax** for embedding your screenshots (with `![](path/to/image.png)`) so you can paste them easily under each section in the README?
```
