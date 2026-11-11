# Predicting Student Dropout and Academic Success

A machine learning app that predicts whether a university student is likely to **graduate**, **drop out**, or **still be enrolled** at the end of their programme, so advisers can spot at-risk students early.

Built with Python, scikit-learn and Streamlit.

## Dataset

[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) from the UCI Machine Learning Repository: 4,424 students from a Portuguese polytechnic, with demographic, enrolment, financial, academic and economic information.

| Outcome | Students |
| --- | --- |
| Graduate | 2,209 |
| Dropout | 1,421 |
| Enrolled | 794 |

## Output: What the App Predicts

| Outcome | Meaning |
| --- | --- |
| **Graduate** | The student is likely to complete the programme. |
| **Dropout** | The student is at risk of leaving before completing. |
| **Enrolled** | The student is likely to still be studying at the end of the normal programme length. |

For each student, the app shows:

1. **Predicted outcome:** the single most likely class, e.g. `Predicted Academic Outcome: Dropout`.
2. **Probability chart:** the model's probability for each outcome, so you can see how confident the prediction is. A student shown as *Graduate* at 0.48 against *Dropout* at 0.41 is a much less certain case than one at 0.90.

## How the Prediction Is Made

- **Model:** a scikit-learn pipeline of `StandardScaler` + `RandomForestClassifier` (100 trees, `max_depth=10`, `class_weight="balanced"` to handle the smaller *Enrolled* and *Dropout* classes).
- **Inputs (19 features):** demographics (marital status, gender, age at enrolment, nationality), enrolment details (course, daytime/evening attendance, displaced, special needs), financial status (tuition fees up to date, scholarship holder), first- and second-semester performance (units enrolled, units approved, average grade), and economic context (unemployment rate, inflation rate, GDP).
- **Label encoding:** `Enrolled = 0`, `Dropout = 1`, `Graduate = 2`.

## Model Performance

Evaluated on a stratified 20% hold-out test set (885 students, `random_state=42`):

| Metric | Score |
| --- | --- |
| Accuracy | 0.739 |
| Macro F1 | 0.696 |
| Dropout recall | 0.67 |

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Graduate | 0.83 | 0.83 | 0.83 |
| Dropout | 0.86 | 0.67 | 0.75 |
| Enrolled | 0.43 | 0.60 | 0.50 |

> Dropout recall matters most here: it shows how many at-risk students the model actually catches. The model finds about two in three students who go on to drop out.

## How to Read the Results

- **Use it as an early-warning signal, not a verdict.** A *Dropout* prediction means a student shares patterns with past dropouts. It is a prompt for an adviser to check in, not a judgement about the student.
- **Semester performance drives the prediction.** Units approved and grades in the first and second semesters account for about 57% of the model's feature importance, with second-semester units approved the single strongest signal (21%). Predictions are most reliable once semester results are available.
- **The *Enrolled* class is the hardest to predict** (F1 0.50), because these students sit between graduating and dropping out and have the fewest examples in the data.

## Run It Locally

```bash
git clone https://github.com/Teninlanimi/predict-students-dropout-and-academic-success.git
cd predict-students-dropout-and-academic-success
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model, open `deploy_pipeline.ipynb` and run all cells. It writes a new `multiclass_classification_model.pkl` and `final_feature_list.txt`.

## Project Structure

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit app: input form, prediction and probability chart |
| `deploy_pipeline.ipynb` | Loads the data, trains and evaluates the model, and saves it |
| `data.csv` | The UCI dataset |
| `multiclass_classification_model.pkl` | The trained pipeline loaded by the app |
| `final_feature_list.txt` | The 19 input features, in the order the model expects |
| `requirements.txt` | Python dependencies |
