import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load Dataset
df = pd.read_csv("train.csv")
df.head()

# Data types
print(df.dtypes)

# Missing values
print(df.isnull().sum())

#suevival rate by sex
survival_sex = df.groupby("Sex")["Survived"].mean()
print(survival_sex)

survival_sex.plot(kind="bar")
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.show()

# survival rate by passenger class
survival_class = df.groupby("Pclass")["Survived"].mean()
print(survival_class)

survival_class.plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.show()


#survival rate by age buckets
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 80],
    labels=["Child", "Teen", "Adult", "Middle Age", "Senior"]
)

age_survival = df.groupby("AgeGroup")["Survived"].mean()
print(age_survival)

age_survival.plot(kind="bar")
plt.title("Survival Rate by Age Group")
plt.ylabel("Survival Rate")
plt.show()

# visualization (boxplot/violin plot)
sns.boxplot(x="Survived", y="Age", data=df)
plt.title("Age Distribution by Survival")
plt.show()

sns.violinplot(x="Survived", y="Age", data=df)
plt.title("Violin Plot of Age vs Survival")
plt.show()