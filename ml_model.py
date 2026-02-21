import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

occup = pd.read_excel("data/Occupation Data.xlsx")
skills = pd.read_excel("data/Skills.xlsx")

occup = occup[['O*NET-SOC Code', 'Title']]
skills = skills[['O*NET-SOC Code', 'Element Name', 'Data Value']]

occup.columns = ['code', 'career']
skills.columns = ['code', 'skill', 'importance']

df = pd.merge(skills, occup, on='code')

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

model = RandomForestClassifier()
model.fit(X, y_encoded)

def predict_career(user_skills):
    user = pd.DataFrame(0, index=[0], columns=X.columns)

    for s in user_skills:
        if s in user.columns:
            user[s] = 1

    pred = model.predict(user)
    return le.inverse_transform(pred)[0]