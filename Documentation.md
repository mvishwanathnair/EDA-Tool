
### 1. Technology Overview
- **Introduction**: This tool is used to the Exploratory data analysis of a dataset.Exploratory Data Analysis (EDA) is a crucial initial step in data science projects. It involves analyzing and visualizing data to understand its key characteristics, uncover patterns, and identify relationships between variables refers to the method of studying and exploring record sets to apprehend their predominant traits, discover patterns, locate outliers, and identify relationships between variables. EDA is normally carried out as a preliminary step before undertaking extra formal statistical analyses or modeling.
- **Key Features**: This tool can analyse data and can process the data using ML algorithms
- **Strengths and Limitations**: 

 **Strengths:** Makes the work of user easier, by performing EDA on the data set, has the ability to look for new ML algorithms

**Limitations:** Might fail to implement ML algorithms if certain data are missing in the data set.

### 2. Technical Specifications
- **API Details**: 
The API's used are:
1.OpenAI 
2.Wikipedia API wrapper
3.Streamlit Framework
4.Langchain.experimental_agents (pandas agent and PythonREPLTool) 
- **Supported Languages and Platforms**: 
Languages: Python 3.9 or higher
Platforms: Windows, Mac OS, Web
- **Integration Requirements**: 
The requirements mentioned in the requirements.txt should be installed
- **Performance Metrics**: 

### 3. Use Cases
- **Industry Applications**:
Finance: Predictive analytics for stock market trends
Healthcare: Analyzing patient data for better diagnosis
Retail: Customer segmentation and demand forecasting
- **Success Stories**: Case studies showcasing successful implementations.
- **Potential Scenarios**:

Scenario 1: A retail company wants to optimize inventory levels based on sales forecasts.

Scenario 2: A healthcare provider needs to predict patient readmissions to improve care.

### 4. Implementation Guide
- **Step-by-Step Instructions**:
Install the requirements using pip install -r requirements.txt
import the modules and start working on them
- **Code Snippets**: 
Sample code to help with implementation.

EDA:

        @st.cache_data
        def fn_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head(7))
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create?")
            st.write(new_features)
            return

Analysing Dataset:

        @st.cache_data
        def func_qs_variable():
            st.line_chart(df, y=[user_qs])
            summary_statistics = pandas_agent.run(f"What are the mean, median, mode, standard deviation, variance, range, quartiles, skewness, and kurtosis of {user_qs}")
            st.write(summary_statistics)
            normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_qs}")
            st.write(normality)
            outliers = pandas_agent.run(f"Assess the presence of outliers of {user_qs}")
            st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_qs}")
            st.write(trends)
            missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_qs}")
            st.write(missing_values)
            return

User Prompts:

        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_qs_df)
            st.write(dataframe_info)
            return

        @st.cache_resource
        def wiki(prompt):
            wiki_research = WikipediaAPIWrapper().run(prompt)
            return wiki_research

        @st.cache_data
        def prompt_templates():
            data_problem_template = PromptTemplate(
                input_variables=['business_problem'],
                template='Convert the following business problem into a data science problem: {business_problem}.'
            )
            model_selection_template = PromptTemplate(
                input_variables=['data_problem', 'wikipedia_research'],
                template='Give a list of machine learning algorithms that are suitable to solve this problem: {data_problem}, while using this wikipedia research: {wikipedia_research}.'
            )
            return data_problem_template, model_selection_template
- **Configuration Settings**:
API Key: Set in the request headers for authentication
File Format: Ensure the dataset is in CSV or JSON format
Query Language: Use natural language for queries
- **Troubleshooting Tips**:
Invalid API Key: Ensure your API key is correct and not expired.
File Upload Issues: Check file format and size.
Slow Responses: Verify internet connection and server status.

### 5. Best Practices
- **Security Considerations**:
Use secure connections (HTTPS) for API requests.
Store API keys securely and do not expose them in client-side code.
Regularly rotate API keys and monitor usage.
- **Scalability Tips**: 
Implement caching for repeated queries.
Optimize data before uploading (e.g., remove unnecessary columns).
- **Optimization Techniques**:
Preprocess data to reduce size and complexity.
Use batch processing for large-scale data analysis.

### 6. Testing Procedures
- **Test Cases**: 

Upload Test: Verify that a dataset can be uploaded successfully.

EDA Test: Ensure EDA results are accurate and comprehensive.

Query Test: Test various natural language queries for correctness.

Model Recommendation Test: Validate the relevance of model recommendations.
- **Expected Outputs**:

Upload Test: { "status": "success", "data_id": "12345" }

EDA Test: Detailed analysis report.

Query Test: Correct data subset or summary.

Model Recommendation Test: List of recommended models.
- **Troubleshooting Tips**: 

Upload Failures: Check file format, size, and internet connection.

Inaccurate EDA: Ensure data quality and preprocessing.

Query Issues: Refine query language and parameters.

Model Recommendation Errors: Provide clear and detailed problem statements.


