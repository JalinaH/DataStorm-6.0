# Data Storm v6.0 - Preliminary Round Submission

**Team Name:** TechSpark
**Members:** Jalina Hirushan, [@VishwaJaya01](https://github.com/VishwaJaya01), [@th33k](https://github.com/th33k)

## Project Overview

This project addresses the "Predict & Improve Insurance Agent Performance" challenge for Data Storm v6.0. Our solution focuses on two main goals:

1.  Predicting which agents are at risk of "One Month NILL" (selling no policies) in the following month.
2.  Understanding and improving the performance of all current agents by categorizing them and recommending personalized improvement plans.

We utilized Python with libraries such as Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, and Streamlit.

## File Structure

-   `notebook/techspark.ipynb`: Jupyter Notebook containing all EDA, model training, feature importance, action plan logic, agent classification, and intervention strategies.
-   `dashboard/dashboard.py`: Streamlit script for the interactive dashboard.
-   `model/nill_prediction_model.pkl`: Saved trained XGBoost model for NILL prediction.
-   `submission_nill_prediction.csv`: Generated submission file for the Kaggle NILL prediction task (output to project root).
-   `data/train_storming_round.csv`: Provided training data (expected in this subfolder).
-   `data/test_storming_round.csv`: Provided test data (expected in this subfolder).
-   `requirements.txt`: Python package dependencies.
-   `README.md`: This file.

## I. EDA - Exploratory Data Analysis

Our Exploratory Data Analysis, detailed in `notebook/techspark.ipynb`, revealed several key insights:

-   **Key Metrics:** A significant skew in `new_policy_count` towards zero (approx. **[XX.XX]% NILL rate in training data - fill this from your notebook output]**) confirmed the prevalence of NILL months. Agent activity metrics like `unique_proposal` and `ANBP_value` also frequently showed low values, indicating periods of low sales engagement or outcome.
-   **Sales Patterns:** Time series analysis of `year_month` showed **[e.g., a slight upward trend in total policies until mid-2024, followed by a plateau. NILL rates fluctuated but showed a tendency to increase during Q4 potentially due to holiday season effects - customize with your findings]**.
-   **Multivariate Analysis:** Correlation heatmaps highlighted strong positive correlations between `new_policy_count` and activity metrics (`unique_proposal`, `unique_customers`, `ANBP_value`), and corresponding negative correlations for `is_NILL`. `Agent_age` showed a weak positive correlation with performance.
-   **Agent Trajectories:** Examining individual agent performance over time demonstrated high variability. Some agents showed consistent performance, while others were erratic, improved, or declined, underscoring the need for personalized interventions.
-   **Innovative EDA:**
    -   **Tenure Analysis:** Agents in their initial 0-3 months (`agent_tenure_at_record_months`) exhibited a higher NILL rate. Performance tended to improve and peak around **[X-Y months - fill this]** of tenure, after which it sometimes plateaued.
    -   **Time to First Sale:** Agents achieving their first policy sale within the first month (`time_to_first_sale_months_eda` = 0 or <1) generally demonstrated lower subsequent NILL rates, suggesting early success is a positive indicator.

_Refer to `notebook/techspark.ipynb` (Section I) for detailed EDA code, charts, and insights._

## II. Part 1 - Predict NILL Agents

### 1. Trained Prediction Model (Kaggle Task)

-   **Model:** An XGBoost Classifier (`xgb.XGBClassifier`) was trained to predict if an agent will sell any policies (target=1) or go NILL (target=0) in the subsequent month.
-   **Features:** The model utilized **[Number]** features, including derived tenure metrics (e.g., `agent_tenure_at_record_months`, `months_since_first_sale_at_record`), activity ratios (`proposal_per_customer`), raw activity counts from the past 7/15/21 days (e.g., `unique_proposals_last_7_days`), premium value (`ANBP_value`), agent demographics (`agent_age`), and date-based components. **[Mention 1-2 more specific important features if distinct, e.g., `tenure_x_proposals` if it was highly ranked]**.
-   **Performance:** **[e.g., On local validation (if performed), the model achieved an AUC of X.XX. Our submission to Kaggle, based on this model, achieved a score of Y.YY / rank ZZZ - fill this as applicable]**.
-   **Output:** Predictions for the test set are provided in `submission_nill_prediction.csv`. The trained model is saved as `model/nill_prediction_model.pkl`.

_Code for feature engineering and model training is in `notebook/techspark.ipynb` (Section II.A & II.B)._

### 2. List of Top Factors Affecting Early Performance

Based on the XGBoost model's feature importances (typically 'gain'), the top factors influencing NILL risk are:

1.  **`[Feature 1 name from your notebook output]`** (Importance: **[Score]**) - Interpretation: **[e.g., The number of unique proposals made in the last 7 days is the strongest predictor; lower values heavily increase NILL risk.]**
2.  **`[Feature 2 name]`** (Importance: **[Score]**) - Interpretation: **[e.g., Agent's tenure at the time of record; newer agents or those with very long tenure without recent success might be more at risk.]**
3.  **`[Feature 3 name]`** (Importance: **[Score]**) - Interpretation: **[e.g., The ANBP value from the current month; very low or zero ANBP is a strong indicator of NILL in the next month.]**
    _(List top 5-7 key factors with brief interpretations based on your actual model output)_

_Detailed feature importance plot and the full list are in `notebook/techspark.ipynb` (Section II.C)._

### 3. Personalized Action Plan Recommendation System for At-Risk Agents

-   **Identification:** Agents flagged with a '0' in `submission_nill_prediction.csv` (i.e., predicted as NILL for the next month) are considered "at-risk."
-   **System Logic:** A rule-based system generates personalized SMART (Specific, Measurable, Achievable, Relevant, Time-bound) action plans.
-   **Personalization:** Recommendations are triggered by the agent's specific data points corresponding to high-importance features from the NILL prediction model.
    -   Example Trigger: If `unique_proposals_last_7_days < X` (a defined threshold), the plan emphasizes increasing recent proposal activity with specific targets.
    -   Example Trigger: If `agent_tenure_at_record_months <= 3`, the plan suggests foundational training, mentorship, and structured onboarding support.
-   **Output:** The system provides a textual plan with actionable steps tailored to the likely reasons for the NILL risk.

_The logic and example outputs are demonstrated in `notebook/techspark.ipynb` (Section II.D) and interactively in `dashboard/dashboard.py`._

## III. Part 2 - Monitor and Improve Existing Agent Performance

### 1. Method to Classify Current Agent Performance (Low, Medium, High)

-   **Data Source:** Historical performance data from `data/train_storming_round.csv`.
-   **Metric:** `avg_policies_per_month_historical` (average `new_policy_count` per month for each agent based on their entire history in the training data).
-   **Classification:**
    -   Agents are categorized into 'Low', 'Medium', or 'High' performers.
    -   Thresholds are dynamically determined using the 33rd and 66th percentiles (quantiles) of the `avg_policies_per_month_historical` distribution across all agents.
    -   Low Threshold: `avg_policies_per_month_historical` <= **[Value of 33rd percentile from your notebook]**
    -   Medium Threshold: `avg_policies_per_month_historical` > **[33rd percentile]** AND <= **[Value of 66th percentile from notebook]**
    -   High Threshold: `avg_policies_per_month_historical` > **[66th percentile]**
    -   Fallback logic is implemented if quantiles are not distinct (e.g., due to a large number of agents with zero average sales).

_Detailed classification logic and the resulting distribution of agents are in `notebook/techspark.ipynb` (Section III.A)._

### 2. Intervention Strategy Based on Performance Category

Custom intervention strategies are proposed for each historical performance category:

-   **Low Performers:**
    -   Focus: Foundational skill building, activity increase, and close monitoring.
    -   Interventions: Diagnostic review, mandatory targeted retraining, structured mentorship, clear activity goal setting, and regular performance reviews.
-   **Medium Performers:**
    -   Focus: Skill enhancement, consistency improvement, and unlocking potential.
    -   Interventions: Targeted skill development (e.g., closing, negotiation), peer learning, stretch goals with incentives, strategic account planning, and career pathing discussions.
-   **High Performers:**
    -   Focus: Retention, leadership development, and leveraging expertise.
    -   Interventions: Advanced recognition/rewards, leadership/mentoring opportunities, specialized skill development (e.g., executive coaching), strategic input, and customized growth paths.

_Detailed strategies are outlined in `notebook/techspark.ipynb` (Section III.B) and summarized for selection in `dashboard/dashboard.py`._

### 3. Optional: Progress Tracker to Measure Changes Over Time (Concept)

-   **Purpose:** To enable long-term monitoring of agent development and the effectiveness of applied interventions.
-   **Method:**
    1.  **Regular Re-assessment:** Periodically (e.g., quarterly) recalculate agent performance and re-classify them.
    2.  **Track Key Metrics:** Monitor category migration (e.g., Low to Medium), changes in average policy sales, and NILL rates post-intervention.
    3.  **Visualization:** Conceptualized dashboards to show trends in category distribution and KPI improvements over time.

_Conceptual details are described in `notebook/techspark.ipynb` (Section III.C)._

## IV. Bonus: Dashboard / Visualization

-   **Tool:** An interactive dashboard was developed using Streamlit (`dashboard/dashboard.py`).
-   **Functionality:**
    1.  **NILL Risk Prediction (Test Data):** Allows selection of an agent record from the test data. Displays:
        -   Key metrics for the selected agent's record.
        -   The predicted probability of the agent being NILL in the next month.
        -   A personalized SMART action plan if the agent is flagged as at-risk.
    2.  **Agent Performance Segments (Train Data):** Allows users to view:
        -   The distribution of agents across Low, Medium, and High historical performance categories.
        -   General intervention strategies recommended for each selected category.
-   **Access:** The dashboard can be run locally. See "How to Run" section.
-   **Live Demo:** Our deployed dashboard can be accessed at: [https://datastorm-dashboard.streamlit.app/](https://datastorm-dashboard.streamlit.app/)

## How to Run

1.  **Prerequisites:**
    *   Python 3.9+
    *   Clone this repository.
    *   Install required packages: `pip install -r requirements.txt`
2.  **Data Setup:**
    *   Ensure `train_storming_round.csv` and `test_storming_round.csv` are placed inside the `data/` subfolder within the project root.
3.  **Jupyter Notebook (Analysis & Model Training):**
    *   Navigate to the `notebook/` directory.
    *   Open and execute `techspark.ipynb` using Jupyter Lab or Jupyter Notebook. This will:
        *   Perform EDA.
        *   Train the NILL prediction model and save it as `model/nill_prediction_model.pkl` (relative to project root).
        *   Generate the Kaggle submission file as `submission_nill_prediction.csv` (in the project root).
4.  **Streamlit Dashboard (Interactive Visualization):**
    *   Ensure `model/nill_prediction_model.pkl` and data files in `data/` exist (as generated/placed above).
    *   From the **project root directory**, run: `streamlit run dashboard/dashboard.py`
    *   The dashboard should open in your web browser.

## Challenges and Future Work

-   **Feature Engineering:** While comprehensive, exploring more complex lagged features (e.g., rolling averages of performance over 3/6 months) or interaction terms could further enhance model accuracy.
-   **External Data Integration:** Incorporating external factors like regional economic indicators, marketing campaign details, or competitor activity could provide a richer context for performance analysis if such data were available.
-   **Dynamic Thresholds for Action Plans:** The triggers for personalized action plans are currently based on fixed thresholds. These could be made more dynamic, perhaps based on an agent's deviation from their own historical baseline or peer group averages.
-   **Intervention Effectiveness Measurement:** If this system were operational, implementing A/B testing for different intervention strategies would be crucial for empirically validating and refining their effectiveness.
-   **Advanced Clustering:** For agent segmentation, more advanced clustering techniques (beyond quantile-based) could be explored to identify more nuanced agent personas.

---