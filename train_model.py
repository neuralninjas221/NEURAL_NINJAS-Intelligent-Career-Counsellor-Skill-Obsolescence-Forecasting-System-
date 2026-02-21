import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

print("Loading data...")

occup = pd.read_excel("data/Occupation Data.xlsx")
skills = pd.read_excel("data/Skills.xlsx")

occup = occup[['O*NET-SOC Code', 'Title']]
skills = skills[['O*NET-SOC Code', 'Element Name', 'Data Value']]

occup.columns = ['code', 'career']
skills.columns = ['code', 'skill', 'importance']

# Clean skill text
skills['skill'] = skills['skill'].str.lower().str.strip()

df = pd.merge(skills, occup, on='code')

# Remove low-importance noise
df = df[df['importance'] > 2]

print("Creating feature matrix...")

pivot = df.pivot_table(
    index='career',
    columns='skill',
    values='importance',
    fill_value=0
)

X = pivot
y = pivot.index

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Splitting train/test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print("Training model...")

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Evaluating...")

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

joblib.dump(model, "career_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model saved successfully.")