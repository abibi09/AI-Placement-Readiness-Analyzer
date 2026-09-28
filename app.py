import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class PlacementReadinessAnalyzer:

    def __init__(self):
        self.data = None
        self.model = None
        self.features = []

    # ==========================================
    # 1. GENERATE STUDENT DATA
    # ==========================================
    def load_data(self, n_students=300):

        np.random.seed(42)

        self.data = pd.DataFrame({
            "student_id": [f"STU{i:04d}" for i in range(1, n_students + 1)],
            "age": np.random.randint(20, 24, n_students),
            "cgpa": np.round(np.random.uniform(5.0, 9.5, n_students), 2),
            "attendance": np.round(np.random.uniform(60, 100, n_students), 1),
            "internships": np.random.randint(0, 4, n_students),
            "projects": np.random.randint(0, 7, n_students),
            "certifications": np.random.randint(0, 5, n_students),
            "coding_skills": np.random.choice(
                ["Beginner", "Intermediate", "Advanced"], n_students
            ),
            "soft_skills": np.random.choice(
                ["Beginner", "Intermediate", "Advanced"], n_students
            ),
            "communication_score": np.round(
                np.random.uniform(40, 100, n_students), 1
            ),
            "aptitude_score": np.round(
                np.random.uniform(40, 100, n_students), 1
            ),
            "domain_knowledge": np.random.choice(
                ["Beginner", "Intermediate", "Advanced"], n_students
            )
        })

        # Readiness calculation
        cgpa_score = (self.data["cgpa"] - 5.0) / 4.5

        coding_score = self.data["coding_skills"].map({
            "Beginner": 0.3,
            "Intermediate": 0.6,
            "Advanced": 1.0
        })

        soft_score = self.data["soft_skills"].map({
            "Beginner": 0.3,
            "Intermediate": 0.6,
            "Advanced": 1.0
        })

        domain_score = self.data["domain_knowledge"].map({
            "Beginner": 0.3,
            "Intermediate": 0.6,
            "Advanced": 1.0
        })

        self.data["readiness_score"] = (
            cgpa_score * 0.20
            + (self.data["attendance"] / 100) * 0.10
            + (self.data["internships"] / 3) * 0.15
            + (self.data["projects"] / 6) * 0.10
            + (self.data["certifications"] / 4) * 0.05
            + (self.data["communication_score"] / 100) * 0.10
            + (self.data["aptitude_score"] / 100) * 0.10
            + coding_score * 0.10
            + soft_score * 0.05
            + domain_score * 0.05
        ) * 10

        self.data["readiness_score"] = (
            self.data["readiness_score"]
            + np.random.normal(0, 0.25, n_students)
        ).clip(0, 10).round(2)

        self.data["readiness_category"] = pd.cut(
            self.data["readiness_score"],
            bins=[0, 4, 6, 8, 10.01],
            labels=["Low", "Average", "Good", "Excellent"],
            include_lowest=True
        )

        return self.data

    # ==========================================
    # 2. TRAIN RANDOM FOREST
    # ==========================================
    def train_model(self):

        df = self.data.copy()

        skill_map = {
            "Beginner": 0,
            "Intermediate": 1,
            "Advanced": 2
        }

        df["coding_skills"] = df["coding_skills"].map(skill_map)
        df["soft_skills"] = df["soft_skills"].map(skill_map)
        df["domain_knowledge"] = df["domain_knowledge"].map(skill_map)

        df["placed"] = (df["readiness_score"] >= 6.5).astype(int)

        self.features = [
            "age",
            "cgpa",
            "attendance",
            "internships",
            "projects",
            "certifications",
            "coding_skills",
            "soft_skills",
            "communication_score",
            "aptitude_score",
            "domain_knowledge"
        ]

        X = df[self.features]
        y = df["placed"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 55)
        print("           MACHINE LEARNING MODEL")
        print("=" * 55)

        print("\nModel: Random Forest Classifier")
        print("Training samples:", len(X_train))
        print("Testing samples:", len(X_test))
        print(f"Accuracy: {accuracy * 100:.2f}%")

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred,
            zero_division=0
        ))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nFeature Importance:")

        importance = pd.Series(
            self.model.feature_importances_,
            index=self.features
        ).sort_values(ascending=False)

        print(importance)

        return accuracy

    # ==========================================
    # 3. INDIVIDUAL STUDENT ANALYSIS
    # ==========================================
    def analyze_student(self, student_id):

        student = self.data[
            self.data["student_id"] == student_id
        ]

        if student.empty:
            print("\nStudent not found.")
            return

        student = student.iloc[0]

        skill_map = {
            "Beginner": 0,
            "Intermediate": 1,
            "Advanced": 2
        }

        values = pd.DataFrame([{
            "age": student["age"],
            "cgpa": student["cgpa"],
            "attendance": student["attendance"],
            "internships": student["internships"],
            "projects": student["projects"],
            "certifications": student["certifications"],
            "coding_skills": skill_map[student["coding_skills"]],
            "soft_skills": skill_map[student["soft_skills"]],
            "communication_score": student["communication_score"],
            "aptitude_score": student["aptitude_score"],
            "domain_knowledge": skill_map[student["domain_knowledge"]]
        }])

        prediction = self.model.predict(values)[0]
        probability = self.model.predict_proba(values)[0][1] * 100

        print("\n" + "=" * 55)
        print("             INDIVIDUAL STUDENT ANALYSIS")
        print("=" * 55)

        print("\nStudent ID:", student["student_id"])
        print("CGPA:", student["cgpa"])
        print("Attendance:", student["attendance"])
        print("Internships:", student["internships"])
        print("Projects:", student["projects"])
        print("Certifications:", student["certifications"])
        print("Coding:", student["coding_skills"])
        print("Soft Skills:", student["soft_skills"])
        print("Communication:", student["communication_score"])
        print("Aptitude:", student["aptitude_score"])
        print("Domain Knowledge:", student["domain_knowledge"])

        print("\nReadiness Score:", student["readiness_score"], "/ 10")
        print("Readiness Category:", student["readiness_category"])

        print(
            "\nPlacement Prediction:",
            "Likely" if prediction == 1 else "Needs Improvement"
        )

        print(
            f"Placement Probability: {probability:.2f}%"
        )

        self.skill_gap_analysis(student)

    # ==========================================
    # 4. SKILL GAP ANALYSIS
    # ==========================================
    def skill_gap_analysis(self, student):

        gaps = []

        if student["cgpa"] < 7:
            gaps.append("Improve CGPA")

        if student["attendance"] < 75:
            gaps.append("Improve attendance")

        if student["internships"] < 1:
            gaps.append("Gain internship experience")

        if student["projects"] < 2:
            gaps.append("Build more projects")

        if student["certifications"] < 2:
            gaps.append("Complete relevant certifications")

        if student["coding_skills"] == "Beginner":
            gaps.append("Improve coding skills")

        if student["soft_skills"] == "Beginner":
            gaps.append("Develop soft skills")

        if student["communication_score"] < 65:
            gaps.append("Improve communication")

        if student["aptitude_score"] < 65:
            gaps.append("Practice aptitude")

        if student["domain_knowledge"] == "Beginner":
            gaps.append("Strengthen domain knowledge")

        print("\nSkill Gaps:")

        if gaps:
            for gap in gaps:
                print("-", gap)
        else:
            print("- No major skill gaps identified.")

        print("\nPersonalized Recommendations:")

        if "Improve coding skills" in gaps:
            print("- Practice Python, Java or problem solving regularly.")

        if "Build more projects" in gaps:
            print("- Build practical projects and upload them to GitHub.")

        if "Gain internship experience" in gaps:
            print("- Apply for internships and gain practical experience.")

        if "Improve communication" in gaps:
            print("- Practice interviews and technical communication.")

        if "Practice aptitude" in gaps:
            print("- Practice quantitative and logical aptitude questions.")

        if "Complete relevant certifications" in gaps:
            print("- Complete relevant industry certifications.")

        if "Strengthen domain knowledge" in gaps:
            print("- Study core subjects and domain-specific concepts.")

        if not gaps:
            print("- Continue improving technical and professional skills.")

    # ==========================================
    # 5. BATCH ANALYSIS
    # ==========================================
    def batch_analysis(self):

        print("\n" + "=" * 55)
        print("                 BATCH ANALYSIS")
        print("=" * 55)

        print("\nTotal Students:", len(self.data))

        print(
            "\nAverage CGPA:",
            round(self.data["cgpa"].mean(), 2)
        )

        print(
            "Average Attendance:",
            round(self.data["attendance"].mean(), 2)
        )

        print(
            "Average Readiness:",
            round(self.data["readiness_score"].mean(), 2)
        )

        print("\nReadiness Distribution:")

        print(
            self.data["readiness_category"].value_counts()
            .reindex(
                ["Low", "Average", "Good", "Excellent"],
                fill_value=0
            )
        )

        print("\nTop 5 Students:")

        print(
            self.data[
                ["student_id", "readiness_score", "readiness_category"]
            ]
            .sort_values(
                "readiness_score",
                ascending=False
            )
            .head()
        )

    # ==========================================
    # 6. VISUALIZATIONS
    # ==========================================
    def create_visualizations(self):

        # Readiness distribution
        plt.figure(figsize=(8, 5))

        self.data["readiness_category"].value_counts().reindex(
            ["Low", "Average", "Good", "Excellent"],
            fill_value=0
        ).plot(kind="bar")

        plt.title("Student Readiness Distribution")
        plt.xlabel("Readiness Category")
        plt.ylabel("Number of Students")
        plt.tight_layout()
        plt.savefig("readiness_distribution.png")
        plt.show()

        # CGPA vs Readiness
        plt.figure(figsize=(8, 5))

        plt.scatter(
            self.data["cgpa"],
            self.data["readiness_score"]
        )

        plt.title("CGPA vs Placement Readiness")
        plt.xlabel("CGPA")
        plt.ylabel("Readiness Score")
        plt.tight_layout()
        plt.savefig("cgpa_vs_readiness.png")
        plt.show()

        # Feature importance
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.features
        ).sort_values(ascending=True)

        plt.figure(figsize=(8, 6))

        importance.plot(kind="barh")

        plt.title("Random Forest Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.show()

    # ==========================================
    # 7. REPORT GENERATION
    # ==========================================
    def generate_report(self):

        report = f"""
AI PLACEMENT READINESS ANALYZER
================================

Total Students: {len(self.data)}

Average CGPA:
{self.data["cgpa"].mean():.2f}

Average Attendance:
{self.data["attendance"].mean():.2f}%

Average Readiness Score:
{self.data["readiness_score"].mean():.2f}/10

READINESS DISTRIBUTION
----------------------
{self.data["readiness_category"].value_counts().reindex(
    ["Low", "Average", "Good", "Excellent"],
    fill_value=0
).to_string()}

MODEL
-----
Algorithm: Random Forest Classifier

The system analyzes academic performance,
attendance, internships, projects,
certifications, coding skills, soft skills,
communication, aptitude and domain knowledge.

Generated automatically by AI Placement Readiness Analyzer.
"""

        with open(
            "placement_readiness_report.txt",
            "w",
            encoding="utf-8"
        ) as file:
            file.write(report)

        print("\nReport generated:")
        print("placement_readiness_report.txt")


# =============================================
# MAIN PROGRAM
# =============================================

if __name__ == "__main__":

    analyzer = PlacementReadinessAnalyzer()

    # Generate data
    df = analyzer.load_data()

    print("\n" + "=" * 55)
    print("       AI PLACEMENT READINESS ANALYZER")
    print("=" * 55)

    print("\nStudents generated:", len(df))

    print("\nFirst 5 Students:")

    print(
        df[
            [
                "student_id",
                "cgpa",
                "attendance",
                "internships",
                "projects",
                "readiness_score",
                "readiness_category"
            ]
        ].head()
    )

    print("\nREADINESS CATEGORY DISTRIBUTION")

    print(
        df["readiness_category"]
        .value_counts()
        .reindex(
            ["Low", "Average", "Good", "Excellent"],
            fill_value=0
        )
    )

    print(
        "\nAverage Readiness Score:",
        round(df["readiness_score"].mean(), 2)
    )

    # Train ML model
    analyzer.train_model()

    # Batch analysis
    analyzer.batch_analysis()

    # Analyze one student
    analyzer.analyze_student("STU0001")

    # Generate visualizations
    analyzer.create_visualizations()

    # Generate report
    analyzer.generate_report()

    print("\n" + "=" * 55)
    print("             ANALYSIS COMPLETE")
    print("=" * 55)