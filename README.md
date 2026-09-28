# AI Placement Readiness Analyzer

An AI-based machine learning system that analyzes student academic and skill-related factors to estimate placement readiness and provide personalized recommendations.

## Features

* Generates student placement-readiness data
* Calculates readiness score
* Categorizes students as Low, Average, Good, or Excellent
* Uses Random Forest Classifier for placement prediction
* Displays model accuracy and classification report
* Performs batch student analysis
* Identifies individual student skill gaps
* Generates personalized recommendations
* Creates data visualizations and reports

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Random Forest Classifier

## Machine Learning

The system uses a Random Forest Classifier to predict placement likelihood based on:

* CGPA
* Attendance
* Internships
* Projects
* Certifications
* Coding Skills
* Soft Skills
* Communication Score
* Aptitude Score
* Domain Knowledge

## Results

* Students analyzed: 300
* Training samples: 240
* Testing samples: 60
* Model accuracy: 90%

## Project Output

The system generates:

* Readiness distribution visualization
* CGPA vs readiness visualization
* Feature importance visualization
* Placement readiness report

## How to Run

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Run the application:

```bash
python app.py
```

## Internship Task

**Task:** Placement Readiness Analyzer
**Domain:** AI & Data Science
**Task ID:** AI-RP-008
