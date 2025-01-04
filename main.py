'''
                                    Import the necessary Libraries
'''
import numpy as np
import random
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

'''
                                    Define the required functions
'''

#Define a function for text cleaning
def clean_text(Text):
    if pd.isnull(Text):  # Handling missing values
        return ""
    Text = re.sub('http\S+\s*', ' ', Text)  # remove URLs
    Text = re.sub('RT|cc', ' ', Text)  # remove RT and cc
    Text = re.sub('#\S+', '', Text)  # remove hashtags
    Text = re.sub('@\S+', '  ', Text)  # remove mentions
    Text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',Text)# remove punctuations
    Text = re.sub(r'[^\x00-\x7f]', r' ', Text) #remove non ascii characters
    Text = re.sub('\s+', ' ', Text)  # remove extra whitespace
    return Text

#Define a function to extract skills from the job description and resumes
def extract_skills(job_text, skill_set):
    job_skills = set()
    for skill in skill_set:
        # Use word boundaries to ensure standalone matches
        if re.search(rf'\b{re.escape(skill)}\b', job_text, re.IGNORECASE):
            job_skills.add(skill)
    return job_skills

#Define a function for computing cosine similarity between resume and job description
def compute_cosine_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]  # Return similarity score

#Define a function to compute skill match score between job and resume
def compute_skill_match(resume_skills, job_skills):
    intersect = len(resume_skills.intersection(job_skills))
    required = len(job_skills)
    score = (intersect / required) if required>0 else 0
    return score

'''
                                    Prepare the Resume Dataset
'''

#Import the resume dataset into the enviroment
resume_df = pd.read_csv('UpdatedResumeDataSet.csv')

#Find the duplicated resumes from the dataset and drop them
resume_df.drop_duplicates(subset=['Resume'], keep='first',inplace = True)
resume_df.reset_index(inplace=True,drop=True)

#Apply the clean_text() on the Resume column and store the cleaned resumes under the cleaned_resume column
resume_df['cleaned_resume'] = resume_df.Resume.apply(lambda x: clean_text(x))

'''
                                    Prepare the Job Postings dataset
'''

#Import the job postings dataset into the enviroment
ds_jobs_df = pd.read_csv('Cleaned_DS_Jobs.csv')

#Apply the clean_text() on the Job Description column and store the cleaned job descriptions under the cleaned_jobdescription column
ds_jobs_df['cleaned_jobdescription'] = ds_jobs_df['Job Description'].apply(lambda x: clean_text(x))

'''
Filter out unnecessary resumes from the resume dataset. Extract skills required for a job from a specific job description, 
and skills present in a specific resume 
'''
#Filter resumes by selected categories
selected_categories = [
    'Data Science', 'HR', 'Java Developer', 'Business Analyst',
    'Automation Testing', 'Python Developer', 'Database', 'Hadoop',
    'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing'
]
resume_df = resume_df[resume_df['Category'].isin(selected_categories)].reset_index(drop=True)

#Define a list to store the common skills extracted from multiple job descriptions
common_skills = ['Python', 'Machine Learning', 'Statistics', 'SQL', 'R', 'Data Analysis', 'Spark', 'Hadoop', 'Java',
                 'AWS', 'Big Data', 'Databases', 'Tableau', 'Data Visualization', 'TensorFlow', 'Deep Learning', 'Excel',
                 'NoSQL', 'Business Intelligence', 'Artificial Intelligence', 'Pandas', 'ETL', 'NumPy', 'PyTorch', 'NLP',
                 'MATLAB', 'Scikit-learn', 'Azure', 'Git', 'Cloud Computing', 'Keras', 'Power BI', 'Probability', 'API',
                 'Data Warehousing', 'Flask', 'Google Cloud', 'Django']

#Create a new column to store the skills required for a specific job
ds_jobs_df['skills_required'] = ds_jobs_df['cleaned_jobdescription'].apply(lambda x: extract_skills(x, common_skills))
#Create a new column to store the skills present in a specific resume
resume_df['skills_present'] = resume_df['cleaned_resume'].apply(lambda x: extract_skills(x, common_skills))

'''
Create a new dataframe to store resume-job pairs along with their cosine similarity, skill matching score, and average salary
'''
resume_job_df = []
for _, resume_row in resume_df.iterrows():
    resume_desc = resume_row['cleaned_resume']
    resume_skills = resume_row['skills_present']
    for _, job_row in ds_jobs_df.iterrows():
        job_desc = job_row['cleaned_jobdescription']
        job_skills = job_row['skills_required']

        cosine_sim = compute_cosine_similarity(resume_desc, job_desc)
        skill_score = compute_skill_match(resume_skills, job_skills)

        resume_job_df.append({
            'cleaned_resume': resume_desc,
            'cleaned_jobdescription': job_desc,
            'cosine_similarity': cosine_sim,
            'skill_overlap_score': skill_score,
        })

resume_job_df = pd.DataFrame(resume_job_df)

#Create heuristic labels which are required to train supervised models¶
#Generate dynamic weights for each sample
random.seed(1)
w1 = np.random.uniform(0.5, 0.8, len(resume_job_df))  #Weight range for cosine similarity
w2 = 1 - w1  #Complementary weight for skill overlap

#Calculate compatibility_index
resume_job_df['compatibility_index'] = (w1 * resume_job_df['cosine_similarity']) + (w2 * (resume_job_df['skill_overlap_score']))

'''
                                            Modelling
'''
X = resume_job_df[['cosine_similarity', 'skill_overlap_score']]
y = resume_job_df['compatibility_index']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on the test set
y_pred = lr_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Linear Regression Results:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Initialize Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print results
print("Random Forest Results:")
print(f"Mean Absolute Error (MAE): {mae_rf:.4f}")
print(f"Mean Squared Error (MSE): {mse_rf:.4f}")
print(f"R-squared (R²): {r2_rf:.4f}")

# Initialize XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate XGBoost
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost Results:")
print(f"Mean Absolute Error (MAE): {mae_xgb:.4f}")
print(f"Mean Squared Error (MSE): {mse_xgb:.4f}")
print(f"R-squared (R²): {r2_xgb:.4f}")