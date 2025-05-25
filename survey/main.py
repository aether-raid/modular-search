import pandas as pd
from pandas.core.arrays import boolean
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Predicted negatives (directly use question numbers)
predicted_negative_question_numbers = [
    1, 9, 10, 11, 39, 43, 54, 55, 59, 71, 72, 74, 76, 77, 87, 89, 90, 102, 109,
    113, 126, 127, 137, 140, 144, 145, 152, 162, 166, 171, 172, 174, 176, 178,
    180, 181, 184, 37
]

# Step 1: Download and preprocess the data
url = "https://web-raider.pages.dev/api/download-database"
headers = {"Authorization": "Bearer aether-raid-web-raider"}
response = requests.get(url, headers=headers)

if response.status_code == 200:
    with open("database_output.csv", "wb") as file:
        file.write(response.content)
else:
    print("Failed to download the CSV file. Status Code:", response.status_code)
    exit()

# Read and process database_output.csv
try:
    with open("database_output.csv", "r") as file:
        lines = file.readlines()

    # Extract total respondents and clean file
    total_respondents = int(lines[0].split(",")[1].strip().strip('"'))
    with open("cleaned_database.csv", "w") as file:
        file.writelines(lines[2:])
except (IndexError, ValueError, FileNotFoundError) as e:
    print("Error processing database_output.csv:", e)
    exit()

# Load cleaned data into a DataFrame
try:
    df = pd.read_csv("cleaned_database.csv")
    df["NOT_RELEVANT"] = df["NOT_RELEVANT"].fillna(0).replace("", 0)
    df["RELEVANT"] = df["RELEVANT"].fillna(0).replace("", 0)
    df = df.astype({"RELEVANT": int, "NOT_RELEVANT": int})
except (KeyError, ValueError) as e:
    print("Error reading cleaned_database.csv:", e)
    exit()

# Add Actual Output column: 1 if RELEVANT > NOT_RELEVANT (True Positive), otherwise 0 (True Negative)
df.loc[df['QN_NO'].isin(predicted_negative_question_numbers), 'NOT_RELEVANT'] -= 3
df['Actual_Output_Relevant'] = (df['NOT_RELEVANT']/(df['RELEVANT'] + df['NOT_RELEVANT']) >= 0.25).astype(int)
df['Actual_Output_Relevant'] = abs(df['Actual_Output_Relevant']-1)


# Step 2: Create a predicted output (Predicted_Negative)
# Mark as predicted negative (0) if it's in the list of predicted negatives, else predicted positive (1)
df['Predicted_Negative'] = df['QN_NO'].isin(predicted_negative_question_numbers).astype(int)
df['Predicted_Positive'] = abs(df['Predicted_Negative']-1)

# save df as a csv file to debug
df.to_csv("df.csv")

# Print a sample of Predicted Negative to debug
print("Sample of Predicted Negative:\n", df[['QN_NO', 'Predicted_Negative']].head())

#print response rate
response_rate = len(df) / total_respondents
print("Response Rate:", response_rate)
# print total respondents
print("Total Respondents:", total_respondents)
#print relavant and not relevant votes
print("Relevant Votes:", df['RELEVANT'].sum())
print("Not Relevant Votes:", df['NOT_RELEVANT'].sum())
# Step 3: Calculate confusion matrix and evaluation metrics
conf_matrix = confusion_matrix(df['Actual_Output_Relevant'], df['Predicted_Positive'])
accuracy = accuracy_score(df['Actual_Output_Relevant'], df['Predicted_Positive'])
f1 = f1_score(df['Actual_Output_Relevant'], df['Predicted_Positive'])
precision = precision_score(df['Actual_Output_Relevant'], df['Predicted_Positive'])
recall = recall_score(df['Actual_Output_Relevant'], df['Predicted_Positive'])

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

# Step 4: Visualize the confusion matrix and save it as a PNG file
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

# print TP, TN, FP, FN
TP = conf_matrix[1][1]
TN = conf_matrix[0][0]
FP = conf_matrix[0][1]
FN = conf_matrix[1][0]
print("True Positives:", TP)
print("True Negatives:", TN)
print("False Positives:", FP)
print("False Negatives:", FN)

import json
import pandas as pd
# read the jsonl file
with open("query_results.jsonl", "r") as file:
    lines = file.readlines()
# create a list of dictionaries
data = []
for line in lines:
    json_data = json.loads(line)
    data.append(json_data)
# create a dataframe
df_json = pd.DataFrame(data)
# delete the question_title, question_body, and model_answer fields
df_json = df_json.drop(columns=['question_title', 'question_body', 'model_answer'])
# for each question, include a question number, the model_answer_score
df_json['model_answer_score'] = df_json['model_answer_score'].astype(float)
# save the dataframe as a csv file
df_json.to_csv("df_json.csv")


# create a auc curve with df_json and df
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(df['Actual_Output_Relevant'], df_json['model_answer_score'])
print("ROC AUC Score:", roc_auc)
# save the roc curve as a png file
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(df['Actual_Output_Relevant'], df_json['model_answer_score'])
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig("roc_curve.png")
