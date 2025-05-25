# read from results.json and generate a report
# sample of a json line from results.jsonl {"question": "How do you run CMD.exe under the Local System Account?", "justification": "The most suitable platform to find an answer is a forum, as it allows for community-driven discussions and Q&A, which can provide specific guidance on running CMD.exe under the Local System Account.", "choices": ["forum"], "confidence": "80.00"}
# save the data into a pandas dataframe llm_categories_df
# read from query_results.json and generate a report
# sample of a json line from query_results.jsonl {"question_title": "Java lib or app to convert CSV to XML file?", "question_body": "Is there an existing application or library in Java which will allow me to convert a CSV data file to XML file? The XML tags would be provided through possibly the first row containing column headings.", "model_answer": "## Converting CSV to XML using Java\n\nThere are several libraries and approaches available for converting a CSV file to an XML file in Java.\n\n### Direct Answer\nYou can use the following libraries/applications to convert CSV to XML:\n\n* Apache Commons CSV: This library provides a simple way to read and write CSV files, which can be easily converted to XML.\n* OpenCSV: Similar to Apache Commons CSV, this library allows for easy reading and writing of CSV files.\n* JAXB (Java Architecture for XML Binding): This is a Java API for converting between Java classes and XML documents.\n\n### Approach 1: Using Apache Commons CSV\n\nAccording to the search results, you can use Apache Commons CSV to read the CSV file and then emit XML. For example:\n\n```java\nimport org.apache.commons.csv.CSVFormat;\nimport org.apache.commons.csv.CSVParser;\nimport org.apache.commons.csv.CSVPrinter;\n\n// Read the CSV file using Apache Commons CSV\nCSVParser parser = CSVParser.parse(\"input.csv\", CSVFormat.DEFAULT);\n\n// Emit XML from the parsed CSV data\nXMLWriter writer = new XMLWriter();\nwriter.write(parser);\n```\n\n### Approach 2: Using OpenCSV\n\nSimilarly, you can use OpenCSV to read the CSV file and then convert it to XML. For example:\n\n```java\nimport au.com.bytecode.opencsv.CSVReader;\n\n// Read the CSV file using OpenCSV\nCSVReader reader = new CSVReader(\"input.csv\");\n\n// Convert the CSV data to XML\nXMLWriter writer = new XMLWriter();\nwriter.write(reader);\n```\n\n### Approach 3: Using JAXB\n\nAlternatively, you can use JAXB to convert the CSV data to a Java object and then marshal it to XML. For example:\n\n```java\nimport javax.xml.bind.JAXBContext;\nimport javax.xml.bind.Marshaller;\n\n// Create a Java object from the CSV data\nCSVData csvData = new CSVData();\ncsvData.setValues(reader);\n\n// Marshal the Java object to XML using JAXB\nJAXBContext context = JAXBContext.newInstance(CSVData.class);\nMarshaller marshaller = context.createMarshaller();\nmarshaller.marshal(csvData, writer);\n```\n\n### Conclusion\n\nIn conclusion, there are several libraries and approaches available for converting a CSV file to an XML file in Java. You can use Apache Commons CSV, OpenCSV, or JAXB depending on your specific requirements.\n\nReferences:\n\n* [rgagnon.com/javadetails/\u2026](http://rgagnon.com/javadetails/\u2026) (Apache Commons CSV)\n* [codebrane](https://stackoverflow.com/users/123456/codebrane) (OpenCSV)\n* [ceving](https://stackoverflow.com/users/123456/ceving) (JAXB)\n\nNote: The above code snippets are for illustration purposes only and may require modifications to suit your specific use case.", "model_answer_score": 92.5}
# save the data into a pandas dataframe llm_answers_df
# merge the two dataframes on the question (question and question_title) column and save the result into llm_results_df
# show a pie chart of the count of each question (choices) by question type (choices) using the llm_results_df
# show a bar chart of the average score (model_answer_score) by question type (choices) using the llm_results_df
#%%
import pandas as pd
import json
import matplotlib.pyplot as plt
from pandas import json_normalize
#%%
# read from results.json and generate a report
llm_categories_df = pd.DataFrame()
with open('results.jsonl') as f:
    for line in f:
        data = json.loads(line)
        llm_categories_df = pd.concat([llm_categories_df, pd.json_normalize(data)], ignore_index=True)
llm_categories_df = llm_categories_df.reset_index(drop=True)
llm_categories_df
#%%
# read from query_results.json and generate a report
llm_answers_df = pd.DataFrame()
with open('query_results.jsonl') as f:
    for line in f:
        data = json.loads(line)
        llm_answers_df = pd.concat([llm_answers_df, pd.json_normalize(data)], ignore_index=True)
llm_answers_df = llm_answers_df.reset_index(drop=True)
llm_answers_df
#%%
# merge the two dataframes on the question (question and question_title) column and save the result into llm_results_df
llm_results_df = pd.merge(llm_categories_df, llm_answers_df, left_on='question', right_on='question_title', how='inner')
llm_results_df = llm_results_df.drop(columns=['question_title'])
llm_results_df
#%%
# show a pie chart of the count of each question (choices) by question type (choices) using the llm_results_df
llm_results_df['choices'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()
#%%
# show a bar chart of the average score (model_answer_score) by the first question type (choices is a list) using the llm_results_df\
llm_results_df['choices'] = llm_results_df['choices'].apply(lambda x: x[0])
# make a bloxplot of the model_answer_score by choices
llm_results_df.boxplot(column='model_answer_score', by='choices')
# scale the y-axis to 80-100
plt.ylim(92.6, 92.4)
plt.show()
#%%
llm_results_df['choices'].value_counts()
# %%
llm_results_df.boxplot(column='model_answer_score')
plt.ylim(92, 94)
# %%
# scatter plot of model_answer_score
llm_results_df.plot.scatter(y='model_answer_score', x='confidence')
plt.show()
# %%
