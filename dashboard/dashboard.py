import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

st.set_page_config(
    page_title="DataStorm 6.0 Agent Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model(model_filename='nill_prediction_model.pkl'):
    model_path = PROJECT_ROOT / "model" / model_filename
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure it's in the correct location.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data(train_filename='train_storming_round.csv', test_filename='test_storming_round.csv'):
    train_path = PROJECT_ROOT / "data" / train_filename
    test_path = PROJECT_ROOT / "data" / test_filename
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df
    except FileNotFoundError:
        st.error("Train or Test CSV file not found. Please place them in the same directory as the dashboard script.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

model_nill = load_model()
train_df, test_df_orig = load_data()

def create_features_dashboard(df, ref_date_str="2025-01-01"):
    X = df.copy()
    for col in ['agent_join_month', 'first_policy_sold_month', 'year_month']:
        X[col] = pd.to_datetime(X[col], format='%m/%d/%Y', errors='coerce')

    fixed_ref_date = pd.to_datetime(ref_date_str)
    X['agent_tenure_at_record_months'] = ((X['year_month'] - X['agent_join_month']).dt.days // 30).fillna(0)
    X.loc[X['agent_tenure_at_record_months'] < 0, 'agent_tenure_at_record_months'] = 0
    X['months_since_first_sale_at_record'] = ((X['year_month'] - X['first_policy_sold_month']).dt.days // 30).fillna(-1)
    X.loc[X['months_since_first_sale_at_record'] < 0, 'months_since_first_sale_at_record'] = -1
    X['months_since_joined_fixed_ref'] = (fixed_ref_date - X['agent_join_month']).dt.days // 30
    X['months_since_first_sale_fixed_ref'] = (fixed_ref_date - X['first_policy_sold_month']).dt.days // 30
    X['months_since_first_sale_fixed_ref'] = X['months_since_first_sale_fixed_ref'].fillna(-1)
    X['proposal_per_customer'] = X['unique_proposal'] / (X['unique_customers'] + 1e-6)
    X['quotation_per_proposal'] = X['unique_quotations'] / (X['unique_proposal'] + 1e-6)

    for col_prefix, date_col in [('join', 'agent_join_month'),
                                  ('first_sale', 'first_policy_sold_month'),
                                  ('record', 'year_month')]:
        if date_col in X.columns and pd.api.types.is_datetime64_any_dtype(X[date_col]):
            X[f'{col_prefix}_year'] = X[date_col].dt.year
            X[f'{col_prefix}_month_of_year'] = X[date_col].dt.month
            X[f'{col_prefix}_day_of_year'] = X[date_col].dt.dayofyear
            X[f'{col_prefix}_week_of_year'] = X[date_col].dt.isocalendar().week.astype(int)
        else:
            X[f'{col_prefix}_year'] = 0 
            X[f'{col_prefix}_month_of_year'] = 0
            X[f'{col_prefix}_day_of_year'] = 0
            X[f'{col_prefix}_week_of_year'] = 0

    X['tenure_x_proposals'] = X.get('agent_tenure_at_record_months', 0) * X.get('unique_proposal', 0)
    X = X.drop(columns=['agent_join_month', 'first_policy_sold_month', 'year_month'], errors='ignore')
    
    expected_features = ['number_of_policy_holders', 'months_since_first_sale_fixed_ref', 'unique_quotations_last_21_days', 'join_week_of_year', 'join_month_of_year', 'quotation_per_proposal', 'record_day_of_year', 'ANBP_value', 'unique_customers_last_15_days', 'unique_customers_last_21_days', 'months_since_joined_fixed_ref', 'agent_tenure_at_record_months', 'unique_proposals_last_21_days', 'net_income', 'months_since_first_sale_at_record', 'number_of_cash_payment_policies', 'unique_proposals_last_7_days', 'record_month_of_year', 'unique_proposal', 'unique_quotations', 'record_week_of_year', 'agent_age', 'proposal_per_customer', 'unique_customers_last_7_days', 'join_year', 'unique_customers', 'unique_proposals_last_15_days', 'record_year', 'first_sale_year', 'first_sale_month_of_year', 'first_sale_day_of_year', 'unique_quotations_last_15_days', 'tenure_x_proposals', 'first_sale_week_of_year', 'join_day_of_year', 'unique_quotations_last_7_days']
    
    for col in expected_features:
        if col not in X.columns:
            X[col] = -1

    X_final = X[expected_features].copy()
    X_final = X_final.fillna(-1)
    return X_final

def recommend_smart_action_plan_dashboard(agent_data_row_dict, for_month_str="the next month"):
    actions = []
    agent_id = agent_data_row_dict.get('agent_code', 'Unknown Agent')

    plan_header = f"SMART Action Plan for Agent {agent_id} (for {for_month_str}):\n"
    actions.append(plan_header)
    actions.append("Overall Goal: Achieve at least 1 policy sale.\n")

    if 'unique_proposal' in agent_data_row_dict and agent_data_row_dict['unique_proposal'] < 5:
        actions.append(
            "**Focus: Increase Proposal Volume**\n"
            "  - S: Increase unique proposals.\n"
            "  - M: Target >=10 proposals.\n"
            "  - A: Extra hour daily for leads/proposals.\n"
            "  - R: More proposals = more sales opportunities.\n"
            "  - T: By end of {for_month_str}.\n".format(for_month_str=for_month_str)
        )
    tenure_feat = 'agent_tenure_at_record_months' if 'agent_tenure_at_record_months' in agent_data_row_dict else 'months_since_joined_fixed_ref'
    if tenure_feat in agent_data_row_dict and agent_data_row_dict[tenure_feat] <=3:
        actions.append(
            "**Focus: Foundational Skills & Mentorship (New Agent)**\n"
            "  - S: Strengthen product/sales knowledge.\n"
            "  - M: Complete 2 trainings, shadow 3 calls.\n"
            "  - A: Schedule time with mentor.\n"
            "  - R: Crucial for early success.\n"
            "  - T: Within two weeks of {for_month_str}.\n".format(for_month_str=for_month_str)
        )

    if len(actions) == 2:
        actions.append("**Focus: General Sales Activity Review & Mentorship.**\n")
    
    actions.append("\n**Generic Support:** Attend trainings, seek feedback, review best practices.\n")
    return "\n".join(actions)

def recommend_interventions_dashboard(category, agent_code="[Agent]"):
    interventions = f"Intervention Strategy for {agent_code} (Performance Category: {category}):\n"
    if category == 'Low':
        interventions += "- **Focus:** Foundational Improvement.\n- **Actions:** Mandatory Retraining, Mentorship, Activity Goals."
    elif category == 'Medium':
        interventions += "- **Focus:** Skill Enhancement.\n- **Actions:** Advanced Training, Peer Learning, Stretch Goals."
    elif category == 'High':
        interventions += "- **Focus:** Leadership & Retention.\n- **Actions:** Mentoring others, Advanced Skills, Recognition."
    return interventions

if not train_df.empty:
    agent_performance = train_df.groupby('agent_code').agg(
        avg_policies_per_month = ('new_policy_count', 'mean')
    ).reset_index()

    if len(agent_performance) > 2 and agent_performance['avg_policies_per_month'].nunique() > 2:
        quantiles = agent_performance['avg_policies_per_month'].quantile([0.33, 0.66]).tolist()
        low_threshold = quantiles[0]
        high_threshold = quantiles[1]
    elif len(agent_performance) > 0:
        low_threshold = agent_performance['avg_policies_per_month'].median() * 0.5
        high_threshold = agent_performance['avg_policies_per_month'].median() * 1.5
        if low_threshold == high_threshold and low_threshold == 0:
             low_threshold = 0.1
             high_threshold = 0.2
        elif low_threshold == high_threshold:
            low_threshold *= 0.9
            high_threshold *= 1.1
    else:
        low_threshold, high_threshold = 0.5, 1.5

    def classify_performance_dashboard(avg_policies):
        if avg_policies <= low_threshold:
            return 'Low'
        elif avg_policies <= high_threshold:
            return 'Medium'
        else:
            return 'High'
    agent_performance['performance_category'] = agent_performance['avg_policies_per_month'].apply(classify_performance_dashboard)
else:
    agent_performance = pd.DataFrame(columns=['agent_code', 'avg_policies_per_month', 'performance_category'])

st.title("📊 Agent Performance & NILL Prediction Dashboard")

if model_nill is None or train_df.empty or test_df_orig.empty:
    st.warning("Dashboard cannot fully operate due to missing model or data. Please check error messages above.")
else:
    tab1, tab2 = st.tabs(["🔮 NILL Risk Prediction (Test Data)", "📈 Agent Performance Segments (Train Data)"])

    with tab1:
        st.header("Predict NILL Risk for Next Month")
        st.markdown("Select an agent record from the test data to see their NILL risk prediction and recommended actions if at risk.")

        if not test_df_orig.empty:
            test_df_display = test_df_orig[['row_id', 'agent_code', 'year_month']].copy()
            test_df_display['display_label'] = "RowID: " + test_df_display['row_id'].astype(str) + \
                                           " | Agent: " + test_df_display['agent_code'] + \
                                           " | Month: " + test_df_display['year_month'].astype(str)
            
            selected_display_label = st.selectbox(
                "Select Agent Record (from Test Set):",
                options=test_df_display['display_label'].tolist(),
                index=0
            )
            
            selected_row_id = int(selected_display_label.split(" | ")[0].split(": ")[1])
            selected_agent_record_orig = test_df_orig[test_df_orig['row_id'] == selected_row_id]

            if not selected_agent_record_orig.empty:
                st.write("#### Selected Agent Record Details:")
                st.dataframe(selected_agent_record_orig[['agent_code', 'year_month', 'unique_proposal', 'unique_customers', 'ANBP_value']])

                agent_features = create_features_dashboard(selected_agent_record_orig.copy())

                if not agent_features.empty:
                    prob_sells = model_nill.predict_proba(agent_features)[0, 1]
                    prob_nill = 1 - prob_sells
                    is_at_risk = prob_nill > 0.5

                    st.metric(label="Predicted Probability of NILL Next Month", value=f"{prob_nill:.2%}")

                    if is_at_risk:
                        st.error("🔴 Agent is AT RISK of NILL performance next month.")
                        agent_data_for_plan = selected_agent_record_orig.iloc[0].to_dict()
                        for col in agent_features.columns:
                            agent_data_for_plan[col] = agent_features[col].iloc[0]

                        record_month_dt = pd.to_datetime(agent_data_for_plan.get('year_month'), errors='coerce')
                        next_month_str = (record_month_dt + pd.DateOffset(months=1)).strftime('%Y-%B') if pd.notnull(record_month_dt) else "the next month"

                        action_plan = recommend_smart_action_plan_dashboard(agent_data_for_plan, next_month_str)
                        st.subheader("Recommended SMART Action Plan:")
                        st.markdown(action_plan)
                    else:
                        st.success("🟢 Agent is NOT predicted to be NILL next month.")
                else:
                    st.warning("Could not generate features for the selected agent record.")
            else:
                st.warning("Selected agent record not found.")
        else:
            st.info("Test data is not loaded. Cannot display NILL risk predictions.")

    with tab2:
        st.header("Historical Agent Performance Segmentation")
        st.markdown("Agents from the training data are segmented into Low, Medium, or High performers based on their historical average monthly policy sales. Select a performance category to see suggested interventions.")

        if not agent_performance.empty:
            col1, col2 = st.columns([1,2])
            with col1:
                st.metric("Total Agents Analyzed (Train)", len(agent_performance))
                st.write("Performance Category Counts:")
                st.dataframe(agent_performance['performance_category'].value_counts())

                selected_category_for_intervention = st.selectbox(
                    "Select Performance Category for Intervention Ideas:",
                    options=['Low', 'Medium', 'High'],
                    index=0
                )
                intervention_text = recommend_interventions_dashboard(selected_category_for_intervention, agent_code="[Selected Category Agents]")
                st.subheader(f"General Intervention Strategy for {selected_category_for_intervention} Performers:")
                st.markdown(intervention_text)

            with col2:
                st.write("#### Agent Performance Distribution (Avg Policies per Month)")
                fig, ax = plt.subplots()
                sns.histplot(data=agent_performance, x='avg_policies_per_month', hue='performance_category',
                             kde=True, ax=ax, multiple="stack", hue_order=['Low', 'Medium', 'High'])
                ax.set_title("Distribution of Avg. Policies by Performance Category")
                ax.set_xlabel("Average Policies per Month (Historical)")
                ax.set_ylabel("Number of Agents")
                st.pyplot(fig)

                st.write("#### Sample Agents by Performance Category:")
                st.dataframe(agent_performance.groupby('performance_category').head(3)[['agent_code', 'avg_policies_per_month', 'performance_category']])
        else:
            st.info("Training data is not loaded or no agent performance data available. Cannot display segmentation.")

    st.sidebar.info(
        """
        **DataStorm v6.0 Dashboard**
        - **NILL Risk Prediction:** Uses the trained XGBoost model to predict if an agent from the *test set* will sell 0 policies next month.
        - **Agent Performance Segments:** Classifies agents from the *training set* based on their historical sales performance and suggests general interventions.
        """
    )
