import seaborn as sns
import pandas as pd

titanic_df=sns.load_dataset('titanic')
print(titanic_df)

# Find the average age of passengers for each class (1st, 2nd, and 3rd).
avg_age_by_class=titanic_df.groupby('class')['age'].mean()
print(f"average age of passengers for each class: {avg_age_by_class}")

# Create a new DataFrame that contains the count of male and female passengers in each age group (e.g., 0-10, 11-20, etc.).
age_gp=pd.cut(titanic_df['age'],bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 100])
passenger_cnt_by_age=titanic_df.groupby(['sex',age_gp]).size().unstack(fill_value=0)
print(f"DataFrame that contains the count of male and female passengers in each age group : {passenger_cnt_by_age}")

# Calculate the survival rate for passengers who were traveling alone (without any siblings, spouses, parents, or children) versus those who were traveling with family members.
titanic_df['family_size']=titanic_df['sibsp']+titanic_df['parch']+1
alone_sur_rate=titanic_df[titanic_df['family_size']==1]['survived'].mean()
with_family_sur_rate=titanic_df[titanic_df['family_size']>1]['survived'].mean()
print("Survival rate for passengers traveling alone and with family members:")
print("Alone:",alone_sur_rate)
print("with family:",with_family_sur_rate)

# For each passenger, calculate the age difference with the oldest sibling (if any) and the age difference with the youngest sibling (if any).
def get_age_diff_with_oldest_sib(row):
    # Convert the 'age' column to numeric to avoid the categorical error
    oldest_sib = pd.to_numeric(titanic_df[(titanic_df['sibsp'] > 0) & (titanic_df['pclass'] == row['pclass']) & (titanic_df['sibsp'] != row['sibsp'])]['age'], errors='coerce').max()
    return row['age'] - oldest_sib if pd.notna(oldest_sib) else pd.NA

def get_age_diff_with_youngest_sib(row):
    # Convert the 'age' column to numeric to avoid the categorical error
    youngest_sib = pd.to_numeric(titanic_df[(titanic_df['sibsp'] > 0) & (titanic_df['pclass'] == row['pclass']) & (titanic_df['sibsp'] != row['sibsp'])]['age'], errors='coerce').min()
    return row['age'] - youngest_sib if pd.notna(youngest_sib) else pd.NA
titanic_df['age_diff_with_oldest_sib']=titanic_df.apply(get_age_diff_with_oldest_sib,axis=1)
titanic_df['age_diff_with_youngest_sib']=titanic_df.apply(get_age_diff_with_youngest_sib,axis=1)
print("Age difference with the oldest sibling and youngest sibling:")
print(titanic_df[['age_diff_with_oldest_sib','age_diff_with_youngest_sib']])

# Find the most common deck letter (A, B, C, etc.) for each passenger class.
def cabin_lettets(cabin):
    return cabin[0] if not pd.isnull(cabin) else pd.NA
titanic_df['cabin_letter']=titanic_df['deck'].apply(cabin_lettets)
most_common_cabin_by_class=titanic_df.groupby('pclass')['cabin_letter'].apply( lambda x: x.mode().iloc[0])
print("most common deck letter:",most_common_cabin_by_class)

# Group the Titanic DataFrame by 'Embarked' (port of embarkation) and find the percentage of passengers who survived in each group.
surv_per_by_embarked=titanic_df.groupby('embarked')['survived'].mean()*100
print("the percentage of passengers who survived in each group:",surv_per_by_embarked)

# Calculate the correlation matrix for the 'Age', 'Fare', and 'Survived' columns in the Titanic dataset and find the feature with the highest absolute correlation with 'Survived'.
corr_mat=titanic_df[['age', 'fare', 'survived']].corr()
higest_corr_feature=corr_mat['survived'].abs().nlargest(2).index[1]
print("Correlation matrix for 'Age', 'Fare', and 'Survived' columns: ")
print(corr_mat)
print("Feature with the highest absolute correlation with 'Survived': ",higest_corr_feature)

# Create a new DataFrame that contains the 'Pclass', 'Sex', 'Age', and 'Fare' columns from the Titanic dataset and pivot it to have 'Pclass' as the index, 'Sex' as the columns, and 'Fare' as the values, with 'Age' as the weights.
pivot_df=titanic_df[['pclass', 'sex', 'age', 'fare']].pivot_table(index='pclass', columns='sex', values='fare', aggfunc='mean', fill_value=0, margins=True, margins_name='Total', dropna=False)
pivot_df.to_csv('pivot.csv')