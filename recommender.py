import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data once
occup = pd.read_excel("data/Occupation Data.xlsx")
skills = pd.read_excel("data/Skills.xlsx")

occup = occup[['O*NET-SOC Code', 'Title']]
skills = skills[['O*NET-SOC Code', 'Element Name', 'Data Value']]

occup.columns = ['code', 'career']
skills.columns = ['code', 'skill', 'importance']

# Clean skill text
skills['skill'] = skills['skill'].str.lower().str.strip()

# Merge
df = pd.merge(skills, occup, on='code')

# Create career-skill matrix
pivot = df.pivot_table(
    index='career',
    columns='skill',
    values='importance',
    fill_value=0
)

def recommend(user_skills, top_n=5):
    user_vector = pd.DataFrame(0, index=[0], columns=pivot.columns)

    for skill in user_skills:
        skill = skill.lower().strip()
        if skill in user_vector.columns:
            user_vector[skill] = 5

    similarities = cosine_similarity(user_vector, pivot)[0]
    scores = pd.Series(similarities, index=pivot.index)

    return scores.sort_values(ascending=False).head(top_n)