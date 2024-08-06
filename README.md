# EDA-Tool
**Exploratory Data Analysis (EDA) Tool:**

This project leverages advanced AI technologies to create an interactive tool for data analysis and machine learning problem-solving. Built using Streamlit, LangChain, and OpenAI, this tool guides users through exploratory data analysis (EDA), data visualization, and machine learning model selection based on their data and business problems.

**Key Features:**

1. **User Interface:**
   - **Streamlit Application:** Provides a user-friendly interface for uploading CSV files and interacting with the AI assistant.
   - **Sidebar Functionality:** Allows users to upload their datasets and access information about the project creator.

2. **Data Analysis and Exploration:**
   - **CSV Upload and Processing:** Users can upload CSV files, which are read into a Pandas DataFrame.
   - **Exploratory Data Analysis (EDA):** The tool performs EDA tasks, including data overview, data cleaning, missing values detection, duplicate values identification, correlation analysis, outlier detection, and feature engineering.
   - **Data Visualization:** Users can visualize the data and obtain summary statistics and trends for specific variables of interest.

3. **AI Integration:**
   - **LangChain LLMs:** Utilizes OpenAI's language models for natural language processing tasks, including generating prompts for data analysis and summarization.
   - **Pandas DataFrame Agent:** Implements a specialized agent to interact with the DataFrame, answering questions related to data columns, missing values, duplicates, and more.

4. **Advanced Problem Solving:**
   - **Business Problem to Data Science Problem:** Converts user-defined business problems into data science problems using LangChain's prompt templates and chains.
   - **Machine Learning Model Selection:** Provides recommendations on suitable machine learning algorithms based on the defined data science problem and relevant Wikipedia research.
   - **Python Script Generation:** Generates Python scripts to address the data science problem using the selected algorithm and the uploaded dataset.

5. **Interactive Workflow:**
   - **Exploratory Data Analysis:** Guides users through EDA steps and displays results directly in the Streamlit app.
   - **Variable Analysis:** Allows users to investigate specific variables, view their summary statistics, and analyze trends and outliers.
   - **Further Data Analysis:** Supports additional queries about the DataFrame, providing detailed insights and analysis.
   - **Data Science Problem Framing:** Assists users in reframing business problems into data science problems and selecting appropriate machine learning models.
   - **Solution Generation:** Provides a Python script solution based on the selected algorithm and dataset.

**Technologies Used:**
- **Streamlit:** For building the interactive web application.
- **LangChain:** For integrating language models and creating agents for data analysis and problem-solving.
- **OpenAI:** For leveraging advanced language models.
- **Pandas:** For data manipulation and analysis.
- **WikipediaAPIWrapper:** For researching information relevant to the data science problem.

This project aims to simplify the process of data analysis and machine learning model selection, making advanced data science techniques more accessible to users through an intuitive and interactive interface.

