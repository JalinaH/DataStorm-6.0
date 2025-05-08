# Data Storm v6.0 - Preliminary Round Submission

**Team Name:** TechSpark

**Members:** Jalina Hirushan, [@Vishwa](https://github.com/VishwaJaya01), [@Theekshana](https://github.com/th33k)

## Project Overview

This project addresses the "Predict & Improve Insurance Agent Performance" challenge for Data Storm v6.0. Our solution focuses on two main goals:
1.  Predicting which agents are at risk of "One Month NILL" (selling no policies) in the following month.
2.  Understanding and improving the performance of all current agents by categorizing them and recommending personalized improvement plans.

We utilized Python with libraries such as Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, and Streamlit.

## File Structure

-   `DataStorm_v6_Submission.ipynb`: Jupyter Notebook containing all EDA, model training, feature importance, action plan logic, agent classification, and intervention strategies.
-   `dashboard.py`: Streamlit script for the interactive dashboard.
-   `nill_prediction_model.pkl`: Saved trained XGBoost model for NILL prediction.
-   `submission_nill_prediction.csv`: Generated submission file for the Kaggle NILL prediction task.
-   `train_storming_round.csv`: Provided training data.
-   `test_storming_round.csv`: Provided test data.
-   `README.md`: This file.

## I. EDA - Exploratory Data Analysis

(Summarize key findings from your EDA notebook section. For example:)
-   **Key Metrics:** `new_policy_count` is heavily skewed towards zero, indicating a significant number of NILL instances (approx. XX%). Agent activity metrics like `unique_proposal` also show many low values.
-   **Sales Patterns:** Analysis of `year_month` showed [mention any trends, seasonality, or spikes/dips observed in total sales, average sales, or NILL rate].
-   **Multivariate Analysis:** Correlation heatmaps revealed [mention key correlations, e.g., positive correlation between activity metrics and policy count, negative with NILL status].
-   **Agent Trajectories:** Individual agent performance varies significantly over time, highlighting the need for personalized approaches.
-   **Innovative EDA:**
    -   Agent tenure analysis showed that [e.g., newer agents (0-3 months) have higher NILL rates, performance peaks around X months].
    -   Time to first sale: Agents achieving their first sale quicker tend to [e.g., have lower subsequent NILL rates].

*Refer to `DataStorm_v6_Submission.ipynb` for detailed EDA code, charts, and insights.*

## II. Part 1 - Predict NILL Agents

### 1. Trained Prediction Model (Kaggle Task)
-   **Model:** An XGBoost Classifier was trained to predict if an agent will sell any policies (1) or go NILL (0) in the next month.
-   **Features:** Key features included [list a few top ones like `unique_proposal`, `agent_tenure_at_record_months`, `months_since_first_sale_at_record`, various date-derived features, and interaction terms. Be specific based on your final model].
-   **Performance:** The model's performance on the Kaggle leaderboard is [Your Rank/Score, if available. Otherwise, mention local validation performance if you did a split].
-   **Output:** Predictions are submitted in `submission_nill_prediction.csv`. The trained model is saved as `nill_prediction_model.pkl`.

*Code for feature engineering and model training is in `DataStorm_v6_Submission.ipynb`.*

### 2. List of Top Factors Affecting Early Performance
Based on the XGBoost model's feature importances, the top factors influencing NILL risk (and thus, early performance) are:
1.  `[Feature 1 name from your model]` (Importance: [Score]) - Interpretation: [e.g., Higher proposal count significantly reduces NILL risk.]
2.  `[Feature 2 name]` (Importance: [Score]) - Interpretation: [...]
3.  `[Feature 3 name]` (Importance: [Score]) - Interpretation: [...]
    ... (List top 5-10)

*Detailed feature importance plot and list are in `DataStorm_v6_Submission.ipynb`.*

### 3. Personalized Action Plan Recommendation System for At-Risk Agents
-   **Identification:** Agents predicted as NILL by the model are flagged as "at-risk."
-   **System Logic:** For each at-risk agent, a personalized SMART (Specific, Measurable, Achievable, Relevant, Time-bound) action plan is generated.
-   **Personalization:** Recommendations are triggered by the agent's specific data points related to key performance drivers (e.g., low proposal count, new CUS, short tenure).
    -   Example Trigger: If `unique_proposal < 5`, suggest focusing on increasing proposal volume with specific targets.
    -   Example Trigger: If `agent_tenure < 3 months`, suggest foundational training and mentorship.
-   **Output:** The system provides a textual plan with actionable steps.

*The logic and example outputs are demonstrated in `DataStorm_v6_Submission.ipynb` and visualized for selected test agents in `dashboard.py`.*

## III. Part 2 - Monitor and Improve Existing Agent Performance

### 1. Method to Classify Current Agent Performance (Low, Medium, High)
-   **Data Source:** Historical performance data from `train_df`.
-   **Metric:** Average `new_policy_count` per month for each agent.
-   **Classification:**
    -   Agents are categorized into 'Low', 'Medium', or 'High' performers.
    -   Thresholds are determined using quantiles of the `avg_policies_per_month` distribution (e.g., Low: bottom 33%, Medium: middle 33%, High: top 33%).
    -   Low Threshold: `avg_policies_per_month` <= [value from your notebook]
    -   Medium Threshold: `avg_policies_per_month` > [low_value] AND <= [high_value from notebook]
    -   High Threshold: `avg_policies_per_month` > [high_value from notebook]

*Detailed classification logic and distribution are in `DataStorm_v6_Submission.ipynb`.*

### 2. Intervention Strategy Based on Performance Category
Custom intervention strategies are proposed for each category:
-   **Low Performers:**
    -   Focus: Foundational improvement and intensive support.
    -   Interventions: Mandatory retraining, dedicated mentorship, strict activity goal setting, shadowing.
-   **Medium Performers:**
    -   Focus: Skill enhancement and consistency.
    -   Interventions: Advanced training, peer learning groups, stretch goals, career pathing discussions.
-   **High Performers:**
    -   Focus: Leadership development, engagement, and retention.
    -   Interventions: Mentoring opportunities, specialized skill development, recognition programs, strategic input.

*Detailed strategies are outlined in `DataStorm_v6_Submission.ipynb` and summarized in `dashboard.py`.*

### 3. Optional: Progress Tracker to Measure Changes Over Time (Concept)
-   **Purpose:** To measure the effectiveness of interventions and track agent development.
-   **Method:**
    1.  Regularly (e.g., quarterly) re-calculate agent performance metrics and re-classify them.
    2.  Track movement between performance categories (e.g., % of Low performers moving to Medium).
    3.  Monitor changes in key KPIs like `avg_policies_per_month` and NILL rate post-intervention.
    4.  Visualize trends and category migration over time.

*Conceptual details are in `DataStorm_v6_Submission.ipynb`.*

## IV. Bonus: Dashboard / Visualization
-   **Tool:** A simple interactive dashboard was built using Streamlit.
-   **Functionality:**
    1.  **NILL Risk Prediction:** Users can select an agent record from the test data. The dashboard displays:
        -   The agent's key current-month metrics.
        -   The predicted probability of being NILL next month.
        -   If at risk, a personalized SMART action plan.
    2.  **Agent Performance Segments:** Users can view:
        -   The distribution of agents across Low, Medium, and High performance categories (based on historical training data).
        -   General intervention strategies for each selected category.
-   **Access:** Run `streamlit run dashboard.py` (ensure required files are present).

## How to Run
1.  Ensure Python and necessary libraries (see `requirements.txt` if provided, or install pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, streamlit) are installed.
2.  Place `train_storming_round.csv`, `test_storming_round.csv` in the same directory as the scripts.
3.  To run the main analysis and generate model/submission: Execute `DataStorm_v6_Submission.ipynb`. This will create `nill_prediction_model.pkl` and `submission_nill_prediction.csv`.
4.  To run the dashboard: Execute `streamlit run dashboard.py` in the terminal from the project directory.

## Challenges and Future Work
-   **Lagged Features:** More sophisticated lagged features (e.g., performance in previous 1, 2, 3 months) could improve NILL prediction.
-   **External Factors:** Incorporating external data (e.g., marketing campaigns, economic indicators) if available.
-   **Dynamic Thresholds for Action Plans:** Action plan triggers could be made more dynamic or based on anomaly detection within an agent's own historical performance.
-   **A/B Testing Interventions:** If implemented, A/B testing different interventions would provide more robust evidence of their effectiveness.
