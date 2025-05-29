import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Recommendations Analytics Dashboard",
    page_icon="❄️",
    layout="wide"
)

import snowflake.connector
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import sys
from datetime import datetime, timedelta, date # Added date
import duckdb
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# --- Global variable for warehouse (ensure it's loaded) ---
warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
user = os.getenv('SNOWFLAKE_USER')
account = os.getenv('SNOWFLAKE_ACCOUNT')


# --- Session State Initialization ---
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'conn': None,
        'data': None,
        'metrics_data': None,
        'data_loaded': False,
        'filtered_data': None,
        'date_range': None, # Tuple of (date_object, date_object) or (None, None)
        'lane_type_filter': [],
        'is_registered_filter': [],
        'device_platform_filter': [],
        'distribution_tenant_filter': [],
        'date_preset': "Full range", # Default preset
        'current_query': None,
        'precision_metrics_date_range': None, # Tuple of (date_object, date_object)
        'precision_metrics_registration': [],
        'precision_metrics_watched': [], # Should store boolean True/False
        'precision_metrics_distribution_tenant': [],
        'update_precision_metrics_display': False,
        'just_loaded_metrics': False,
        'trend_time_period': "Daily",
        'main_date_picker_key': None, # For main date input
        'main_date_preset_selector_key': "Full range", # For main preset selector
        'precision_metrics_date_picker_key': None, # For precision metrics date input
        # Keys for multiselects to store their current values
        'is_registered_multiselect_key': [],
        'device_platform_multiselect_key': [],
        'lane_type_multiselect_key': [],
        'distribution_tenant_multiselect_key': [],
        'pm_registration_filter_key': [],
        'pm_distribution_tenant_filter_key': [],
        'pm_watched_filter_key': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state() # Call it once at the beginning

# --- Callback functions for Precision Metrics Filters ---
def on_pm_reg_change():
    """Callback for precision metrics registration status filter."""
    new_value = st.session_state.get("pm_registration_filter_key")
    if new_value is None: new_value = [] 
    if st.session_state.get('precision_metrics_registration') != new_value:
        st.session_state.precision_metrics_registration = new_value

def on_pm_watch_change():
    """Callback for precision metrics watched status filter."""
    new_value = st.session_state.get("pm_watched_filter_key")
    if new_value is None: new_value = []
    if st.session_state.get('precision_metrics_watched') != new_value:
        st.session_state.precision_metrics_watched = new_value

def on_pm_dist_change():
    """Callback for precision metrics distribution tenant filter."""
    new_value = st.session_state.get("pm_distribution_tenant_filter_key")
    if new_value is None: new_value = []
    if st.session_state.get('precision_metrics_distribution_tenant') != new_value:
        st.session_state.precision_metrics_distribution_tenant = new_value

# --- Helper Functions ---
def get_date_range(data_key='data'):
    """Safely get the min/max date range from the specified data in session state.
    Returns (min_date_obj, max_date_obj) or (None, None)."""
    data_df = st.session_state.get(data_key)
    if data_df is not None and 'BASE_DATE' in data_df.columns:
        # Ensure BASE_DATE is datetime before min/max
        if not pd.api.types.is_datetime64_any_dtype(data_df['BASE_DATE']):
            # Attempt conversion if not already datetime, handling potential errors
            try:
                data_df['BASE_DATE'] = pd.to_datetime(data_df['BASE_DATE'], errors='coerce')
            except Exception: # Broad exception if conversion fails
                return None, None # Cannot determine range if conversion fails

        min_date_ts = data_df['BASE_DATE'].min()
        max_date_ts = data_df['BASE_DATE'].max()
        min_date_obj = min_date_ts.date() if pd.notna(min_date_ts) else None
        max_date_obj = max_date_ts.date() if pd.notna(max_date_ts) else None
        return min_date_obj, max_date_obj
    return None, None

def apply_filters():
    """Apply all selected filters to the main analytics data"""
    if st.session_state.data is not None:
        filtered_df = st.session_state.data.copy()
        
        # Date range filter (expects date objects in st.session_state.date_range)
        current_date_range = st.session_state.get('date_range')
        if isinstance(current_date_range, tuple) and len(current_date_range) == 2:
            start_date, end_date = current_date_range
            if start_date is not None: # Must be a date object
                filtered_df = filtered_df[filtered_df['BASE_DATE'].dt.date >= start_date]
            if end_date is not None: # Must be a date object
                filtered_df = filtered_df[filtered_df['BASE_DATE'].dt.date <= end_date]

        for filter_key, column_name in [
            ('lane_type_filter', 'LANE_TYPE'),
            ('is_registered_filter', 'IS_REGISTERED'),
            ('device_platform_filter', 'DEVICE_PLATFORM'),
            ('distribution_tenant_filter', 'DISTRIBUTION_TENANT')
        ]:
            filter_values = st.session_state.get(filter_key)
            if filter_values: # If list is not empty
                # Handle potential None in filter_values if 'None' string was selected
                processed_values = [None if v == 'None' else v for v in filter_values]
                if None in processed_values:
                    # Filter for values in list OR where column is NaN
                     filtered_df = filtered_df[filtered_df[column_name].isin([v for v in processed_values if v is not None]) | filtered_df[column_name].isnull()]
                else:
                     filtered_df = filtered_df[filtered_df[column_name].isin(processed_values)]
            
        st.session_state.filtered_data = filtered_df
    else:
        st.session_state.filtered_data = pd.DataFrame()

def _apply_date_preset_logic(preset_value, data_key='data', session_range_key='date_range', session_preset_key='date_preset'):
    """Logic to apply a date preset. Modifies session state for range and preset."""
    min_data_date, max_data_date = get_date_range(data_key=data_key) # date objects or None

    start_date, end_date = None, None
    today = datetime.now().date()
    
    effective_max_date_for_preset = max_data_date if max_data_date else today

    if preset_value == "Last 7 Days":
        end_date = effective_max_date_for_preset
        start_date = end_date - timedelta(days=6)
    elif preset_value == "Last 30 Days":
        end_date = effective_max_date_for_preset
        start_date = end_date - timedelta(days=29)
    elif preset_value == "Last 90 Days":
        end_date = effective_max_date_for_preset
        start_date = end_date - timedelta(days=89)
    elif preset_value == "Full range":
        start_date = min_data_date
        end_date = max_data_date
    elif preset_value == "Custom":
        st.session_state[session_preset_key] = "Custom"
        # For "Custom", date_range is set by the date_input, so no change here.
        # apply_filters() might be called by date_input's on_change.
        return 

    # Clamp to actual data bounds if they exist
    if min_data_date is not None and start_date is not None:
        start_date = max(min_data_date, start_date)
    elif start_date is None and preset_value != "Full range":
        start_date = min_data_date 

    if max_data_date is not None and end_date is not None:
        end_date = min(max_data_date, end_date)
    elif end_date is None and preset_value != "Full range":
        end_date = max_data_date

    if start_date and end_date and start_date > end_date:
        start_date = end_date 

    st.session_state[session_range_key] = (start_date, end_date)
    st.session_state[session_preset_key] = preset_value
    
    if data_key == 'data': # Only apply main filters if it's for main data
        apply_filters()


def main_date_preset_callback():
    """Callback for the main date preset selectbox."""
    preset_value = st.session_state.main_date_preset_selector_key
    _apply_date_preset_logic(preset_value, data_key='data', session_range_key='date_range', session_preset_key='date_preset')
    # apply_filters() is called within _apply_date_preset_logic

def main_date_input_callback():
    """Callback for the main date input."""
    new_dates = st.session_state.main_date_picker_key # tuple of date objects
    if new_dates and len(new_dates) == 2:
        # Ensure they are date objects, not None, before comparing
        if new_dates[0] is not None and new_dates[1] is not None:
            current_range = st.session_state.get('date_range')
            # Check if date actually changed
            if current_range is None or (new_dates[0] != current_range[0] or new_dates[1] != current_range[1]):
                st.session_state.date_range = new_dates
                st.session_state.date_preset = "Custom"
                apply_filters()

def apply_precision_metrics_filters(df_to_filter):
    """Apply filters to the precision metrics data. Returns a filtered DataFrame."""
    if df_to_filter is None or df_to_filter.empty:
        return pd.DataFrame() # Return empty if no data
        
    filtered_df = df_to_filter.copy()
    
    # Date range filter (expects date objects in st.session_state.precision_metrics_date_range)
    pm_date_range = st.session_state.get('precision_metrics_date_range')
    if isinstance(pm_date_range, tuple) and len(pm_date_range) == 2:
        start_date, end_date = pm_date_range
        if start_date is not None: # Must be a date object
            filtered_df = filtered_df[filtered_df['BASE_DATE'].dt.date >= start_date]
        if end_date is not None: # Must be a date object
            filtered_df = filtered_df[filtered_df['BASE_DATE'].dt.date <= end_date]

    # Registration filter
    pm_reg_filter = st.session_state.get('precision_metrics_registration')
    if pm_reg_filter:
        filtered_df = filtered_df[filtered_df['IS_REGISTERED'].isin(pm_reg_filter)]
    
    # Watched filter (expects boolean values in the list)
    pm_watched_filter = st.session_state.get('precision_metrics_watched')
    if pm_watched_filter: # Only filter if the list is not empty
        # Ensure comparison is with actual boolean values if WATCHED_ANY_RECOMMENDED is boolean
        # If it can contain strings 'True'/'False', conversion might be needed earlier.
        # Assuming WATCHED_ANY_RECOMMENDED is already boolean in the DataFrame.
        filtered_df = filtered_df[filtered_df['WATCHED_ANY_RECOMMENDED'].isin(pm_watched_filter)]
        
    # Distribution tenant filter
    pm_dist_filter = st.session_state.get('precision_metrics_distribution_tenant')
    if pm_dist_filter:
        filtered_df = filtered_df[filtered_df['DISTRIBUTION_TENANT'].isin(pm_dist_filter)]
            
    return filtered_df

def precision_metrics_date_input_callback():
    """Callback for the precision metrics date input."""
    new_pm_dates = st.session_state.precision_metrics_date_picker_key # tuple of date objects
    if new_pm_dates and len(new_pm_dates) == 2:
        if new_pm_dates[0] is not None and new_pm_dates[1] is not None:
            current_pm_range = st.session_state.get('precision_metrics_date_range')
            if current_pm_range is None or (new_pm_dates[0] != current_pm_range[0] or new_pm_dates[1] != current_pm_range[1]):
                st.session_state.precision_metrics_date_range = new_pm_dates
                # No preset for precision metrics, so no need to set it to "Custom"
                st.session_state.update_precision_metrics_display = True # Keep this to signal display refresh

# --- Snowflake Connection and Queries ---
def test_connection():
    """Test connection to Snowflake and display detailed information"""
    global user, account, warehouse # Ensure global vars are used
    try:
        if not user or not account or not warehouse:
            st.error("❌ Missing required Snowflake environment variables (USER, ACCOUNT, WAREHOUSE).")
            return False # Explicitly return False
            
        st.info(f"🔄 Attempting to connect to Snowflake: {account} (User: {user}, Warehouse: {warehouse})")
        conn_params = {
            'user': user,
            'account': account,
            'authenticator': 'externalbrowser',
            'client_session_keep_alive': True,
            'warehouse': warehouse,
            'insecure_mode': True 
        }
        conn = snowflake.connector.connect(**conn_params)
        cursor = conn.cursor()
        cursor.execute("SELECT current_version(), current_account(), current_region(), current_warehouse(), current_database(), current_schema()")
        details = cursor.fetchone()
        st.success("✅ Successfully connected to Snowflake!")
        st.session_state.conn = conn
        # Optionally display details
        # st.write(f"Snowflake Version: {details[0]}, Account: {details[1]}, Region: {details[2]}, Warehouse: {details[3]}, DB: {details[4]}, Schema: {details[5]}")
        return True # Explicitly return True
    except Exception as e:
        st.error(f"❌ Error connecting to Snowflake: {str(e)}")
        st.session_state.conn = None
        return False # Explicitly return False

def run_analytics_query():
    """Execute the analytics query and return the results as a DataFrame"""
    global warehouse # Ensure global warehouse is used
    if not st.session_state.conn:
        st.error("❌ No Snowflake connection. Please connect first.")
        return None
    try:
        st.info("🔄 Running analytics query...")
        cursor = st.session_state.conn.cursor()
        if warehouse: # Ensure warehouse is set before using it
             cursor.execute(f"USE WAREHOUSE {warehouse}")
        else:
            st.error("❌ SNOWFLAKE_WAREHOUSE is not set. Cannot select warehouse.")
            return None

        query = """
        WITH correct_lane_views_f AS (
            SELECT *,
                playground.dani.standardize_lane_type(list_type, list_name,screen_name) AS lane_type,
                case when user_id not like 'JNAA%' then 'no' else 'yes' end as is_registered
            FROM joyn_snow.im_main.lane_views_f
            WHERE base_date > dateadd(DAY, -365, CURRENT_DATE)
            and base_date < dateadd(DAY, -7, CURRENT_DATE)
        ),
        correct_as_f AS (
            SELECT user_id,lane_type,base_date,lane_label,event_type,
                playground.dani.standardize_lane_type(lane_type, lane_label,screen_name) AS rlane_type,distribution_tenant, 'as' as event_src
            FROM joyn_snow.im_main.asset_select_f
            WHERE base_date > dateadd(DAY, -365, CURRENT_DATE)
            union all
            SELECT user_id,lane_type,base_date,lane_label,event_type,
                playground.dani.standardize_lane_type(lane_type, lane_label,screen_name) AS rlane_type,distribution_tenant, 'vpr' as event_src
            FROM joyn_snow.im_main.video_playback_request_f
            WHERE base_date > dateadd(DAY, -365, CURRENT_DATE)
        )
        SELECT 
            a.base_date,a.lane_type,a.is_registered,a.device_platform,a.distribution_tenant,
            HLL(DISTINCT a.user_id) as distinct_user_impressions,
            HLL(DISTINCT CASE WHEN b.user_id IS NOT NULL THEN b.user_id END) AS distinct_user_clicks,
            ROUND((HLL(CASE WHEN b.user_id IS NOT NULL THEN b.user_id END) / NULLIF(COUNT(DISTINCT a.user_id), 0)) * 100, 2) AS conversion_rate_pct 
        FROM correct_lane_views_f a 
        LEFT JOIN correct_as_f b ON 
        (a.user_id = b.user_id  and a.lane_type = b.rlane_type AND datediff(day, a.base_date, b.base_date) < 8 and b.base_date >= a.base_date and a.distribution_tenant = b.distribution_tenant)
        GROUP BY all 
        order by 1 asc;
        """
        st.session_state.current_query = query
        cursor.execute(query)
        df = cursor.fetch_pandas_all() # More efficient for pandas
        
        if 'BASE_DATE' in df.columns:
            df['BASE_DATE'] = pd.to_datetime(df['BASE_DATE'], errors='coerce').dt.tz_localize(None).astype('datetime64[ns]')
        
        original_count = len(df)
        df = df[df['LANE_TYPE'].str.match(r'^[a-zA-Z_]+$', na=False)] # Allow underscores
        filtered_count = len(df)
        if original_count > filtered_count:
            st.info(f"⚠️ Filtered out {original_count - filtered_count} rows where lane_type contained non-alpha/underscore characters.")
        
        st.session_state.data = df
        st.session_state.data_loaded = True
        
        # Initialize filters and date range after loading new data
        min_d, max_d = get_date_range(data_key='data') # These are date objects
        st.session_state.date_range = (min_d, max_d)
        st.session_state.date_preset = "Full range" # Reset to full range
        # Reset other filters to default (empty lists)
        for fk in ['lane_type_filter', 'is_registered_filter', 'device_platform_filter', 'distribution_tenant_filter']:
            st.session_state[fk] = []
        apply_filters() # Apply the initial full range filter

        st.success(f"✅ Analytics query executed! Retrieved {len(df)} rows.")
        return df
    except Exception as e:
        st.error(f"❌ Error executing analytics query: {str(e)}")
        return None

def run_precision_metrics_query():
    """Execute the precision metrics query."""
    global warehouse
    if not st.session_state.conn:
        st.error("❌ No Snowflake connection.")
        return None
    try:
        st.info("🔄 Running precision metrics query...")
        cursor = st.session_state.conn.cursor()
        if warehouse:
            cursor.execute(f"USE WAREHOUSE {warehouse}")
        else:
            st.error("❌ SNOWFLAKE_WAREHOUSE is not set.")
            return None
        
        query = """
        WITH recent_lane_views AS (
            SELECT
                lvf.user_id,
                lvf.base_date,
                lvf.device_platform,
                playground.dani.standardize_lane_type(lvf.list_type, lvf.list_name) AS lane_type, -- Removed screen_name
                CAST(GET_PATH(flattened.value, 'asset_id') AS TEXT) AS asset_id, 
                lvf.distribution_tenant
            FROM joyn_snow.im_main.lane_views_f AS lvf,
                 LATERAL FLATTEN(INPUT => lvf.asset_list) AS flattened
            WHERE lvf.base_date > DATEADD(DAY, -180, CURRENT_DATE)
              AND playground.dani.standardize_lane_type(lvf.list_type, lvf.list_name) IN ( -- Removed screen_name
                  'recoforyoulane',
                  'becauseyouwatchedlane',
                  'becauseyouwatchedlanediscovery'
              )
              and lvf.base_date < DATEADD(DAY, -7, CURRENT_DATE)
        ),
        watched_videos as (
            SELECT
                r.base_date,
                r.user_id,
                r.device_platform,
                r.distribution_tenant,
                COUNT(DISTINCT r.asset_id) AS distinct_recommended,
                COUNT(DISTINCT CASE 
                    WHEN v.user_id IS NOT NULL THEN COALESCE(v.tvshow_asset_id, v.asset_id)
                END) AS distinct_vvs_from_recommendations,
                IFF(COUNT_IF(v.user_id IS NOT NULL) > 0, TRUE, FALSE) AS watched_any_recommended,
                CASE 
                    WHEN r.user_id LIKE 'JNAA%' THEN 'no' -- Assuming JNAA is non-registered, adjust if needed
                    ELSE 'yes' -- Assuming others are registered
                END AS is_registered, -- Corrected logic based on typical Snowflake usage
                round(zeroifnull(distinct_vvs_from_recommendations)/nullifzero(distinct_recommended),2) as pct_watched
            FROM recent_lane_views r
            LEFT JOIN joyn_snow.im_main.video_views_epg_extended v
                ON r.user_id = v.user_id
                AND (v.tvshow_asset_id = r.asset_id OR v.asset_id = r.asset_id)
                AND v.base_date > r.base_date -- Watched after recommendation
                AND DATEDIFF(DAY, r.base_date, v.base_date) < 8 -- Within 7 days
                and v.base_date > dateadd(day,-180,current_date) 
                and v.content_type = 'VOD'
                and r.distribution_tenant = v.distribution_tenant
            GROUP BY r.user_id, r.base_date, r.device_platform, r.distribution_tenant -- Added missing group bys
        )
        SELECT
            base_date,
            is_registered,
            watched_any_recommended, -- This is boolean
            count(distinct user_id) as total_users,
            median(pct_watched) AS median_recommendation_watch_ratio,
            distribution_tenant
        FROM watched_videos
        group by all
        order by 1 asc
        """
        cursor.execute(query)
        df = cursor.fetch_pandas_all()
        
        if 'BASE_DATE' in df.columns:
            df['BASE_DATE'] = pd.to_datetime(df['BASE_DATE'], errors='coerce').dt.tz_localize(None).astype('datetime64[ns]')
        if 'WATCHED_ANY_RECOMMENDED' in df.columns:
             df['WATCHED_ANY_RECOMMENDED'] = df['WATCHED_ANY_RECOMMENDED'].astype(bool)
        if 'MEDIAN_RECOMMENDATION_WATCH_RATIO' in df.columns:
            df['MEDIAN_RECOMMENDATION_WATCH_RATIO'] = pd.to_numeric(df['MEDIAN_RECOMMENDATION_WATCH_RATIO'], errors='coerce').fillna(0) # Added fillna(0)
        if 'TOTAL_USERS' in df.columns:
            df['TOTAL_USERS'] = pd.to_numeric(df['TOTAL_USERS'], errors='coerce').fillna(0) # Added fillna(0)

        st.session_state.metrics_data = df
        st.session_state.just_loaded_metrics = True 
        
        pm_min_d, pm_max_d = get_date_range(data_key='metrics_data')
        st.session_state.precision_metrics_date_range = (pm_min_d, pm_max_d)
        # Reset precision metrics filters
        for fk in ['precision_metrics_registration', 'precision_metrics_watched', 'precision_metrics_distribution_tenant']:
            st.session_state[fk] = []

        st.success(f"✅ Precision metrics query executed! Retrieved {len(df)} rows.")
        return df
    except Exception as e:
        st.error(f"❌ Error executing precision metrics query: {str(e)}")
        return None

# --- UI Display Functions ---
def display_data_summary(df_display):
    """Display summary statistics and metrics for the filtered data"""
    if df_display is None or df_display.empty:
        st.warning("No data available for summary based on current filters.")
        return

    total_impressions = df_display['DISTINCT_USER_IMPRESSIONS'].sum()
    total_clicks = df_display['DISTINCT_USER_CLICKS'].sum()
    if 'DISTINCT_USER_IMPRESSIONS' in df_display.columns and total_impressions > 0 :
        weighted_avg_conversion_rate = (df_display['DISTINCT_USER_CLICKS'].sum() / total_impressions * 100) if total_impressions > 0 else 0
    else:
        weighted_avg_conversion_rate = df_display['CONVERSION_RATE_PCT'].mean()


    col1, col2, col3 = st.columns(3)
    col1.metric("Total Impressions", f"{total_impressions:,.0f}")
    col2.metric("Total Clicks", f"{total_clicks:,.0f}")
    col3.metric("Avg. Conversion Rate", f"{weighted_avg_conversion_rate:.2f}%")

    with st.expander("Data Overview (Filtered)", expanded=False):
        st.dataframe(df_display.describe(include='all'), use_container_width=True) # Corrected: added closing parenthesis
    
    with st.expander("Metrics Definitions & Query", expanded=False):
        query_to_display = st.session_state.get('current_query', "Analytics query not run yet.")
        st.markdown(f"""
        ### Key Metrics Definitions
        - **Total Impressions**: Sum of HLL distinct user impressions.
        - **Total Clicks**: Sum of HLL distinct user clicks.
        - **Avg. Conversion Rate**: (Total Clicks / Total Impressions) * 100.
        
        ### Analytics Query
        ```sql
{query_to_display}
        ```
        """)

def display_data_visualization(df_display):
    """Display visualizations for the filtered data"""
    if df_display is None or df_display.empty:
        st.warning("No data for visualization based on current filters.")
        return

    # Ensure CONVERSION_RATE_PCT is numeric if it exists, do this on a copy
    df_display_copy = df_display.copy()
    if 'CONVERSION_RATE_PCT' in df_display_copy.columns:
        df_display_copy['CONVERSION_RATE_PCT'] = pd.to_numeric(df_display_copy['CONVERSION_RATE_PCT'], errors='coerce')

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Conv. Rate Trend", "Lane Analysis", "Lane Trend", "Platform Dist.", "Raw Data"])
    
    # Ensure BASE_DATE is datetime for temporal operations
    if 'BASE_DATE' in df_display_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_display_copy['BASE_DATE']):
        df_display_copy['BASE_DATE'] = pd.to_datetime(df_display_copy['BASE_DATE'], errors='coerce')

    with tab1: # Conversion Rate Trend
        time_period = st.selectbox("Select Time Period", ["Daily", "Weekly", "Monthly"], key="main_trend_period_tab1", index=0) # Unique key
        
        # Ensure BASE_DATE is present and not all NaT
        if 'BASE_DATE' not in df_display_copy.columns or df_display_copy['BASE_DATE'].isnull().all():
            st.warning("BASE_DATE column is missing or all null, cannot plot trend.")
            # return # Keep return or allow to proceed to other tabs? For now, let it proceed.
        else:
            trend_df = df_display_copy.copy()
            trend_df = trend_df.dropna(subset=['BASE_DATE', 'CONVERSION_RATE_PCT']) # Drop NaNs in relevant columns

            if trend_df.empty:
                st.warning("No valid date data for trend plot after NaT removal.")
            else:
                if time_period == "Daily":
                    daily_conv = trend_df.groupby(trend_df['BASE_DATE'].dt.date)['CONVERSION_RATE_PCT'].mean().reset_index()
                    fig = px.line(daily_conv, x='BASE_DATE', y='CONVERSION_RATE_PCT', title='Daily Avg. Conversion Rate Trend')
                elif time_period == "Weekly":
                    weekly_conv = trend_df.set_index('BASE_DATE').resample('W-MON')['CONVERSION_RATE_PCT'].mean().reset_index()
                    fig = px.line(weekly_conv, x='BASE_DATE', y='CONVERSION_RATE_PCT', title='Weekly Avg. Conversion Rate Trend')
                else: # Monthly
                    monthly_conv = trend_df.set_index('BASE_DATE').resample('M')['CONVERSION_RATE_PCT'].mean().reset_index()
                    fig = px.line(monthly_conv, x='BASE_DATE', y='CONVERSION_RATE_PCT', title='Monthly Avg. Conversion Rate Trend')
                st.plotly_chart(fig, use_container_width=True)

    with tab2: # Lane Type Analysis (Top 20 by Conversion Rate)
        if 'LANE_TYPE' in df_display_copy.columns and 'CONVERSION_RATE_PCT' in df_display_copy.columns and 'DISTINCT_USER_IMPRESSIONS' in df_display_copy.columns and 'DISTINCT_USER_CLICKS' in df_display_copy.columns:
            lane_metrics = df_display_copy.groupby('LANE_TYPE').agg(
                total_impressions=('DISTINCT_USER_IMPRESSIONS', 'sum'),
                total_clicks=('DISTINCT_USER_CLICKS', 'sum')
            ).reset_index()
            # Ensure total_impressions is not zero before division
            lane_metrics['conversion_rate'] = lane_metrics.apply(
                lambda row: (row['total_clicks'] / row['total_impressions'] * 100) if row['total_impressions'] > 0 else 0,
                axis=1
            ).fillna(0)
            
            # Ensure conversion_rate is numeric before sorting
            lane_metrics['conversion_rate'] = pd.to_numeric(lane_metrics['conversion_rate'], errors='coerce').fillna(0)
            top_lanes = lane_metrics.sort_values('conversion_rate', ascending=False).head(20)
            
            median_overall_conversion = df_display_copy['CONVERSION_RATE_PCT'].median()

            fig = go.Figure()
            fig.add_trace(go.Bar(x=top_lanes['LANE_TYPE'], y=top_lanes['conversion_rate'], name='Conversion Rate'))
            if pd.notna(median_overall_conversion):
                 fig.add_hline(y=median_overall_conversion, line_dash="dash", line_color="red", annotation_text=f"Median: {median_overall_conversion:.2f}%")
            fig.update_layout(title='Top 20 Lane Types by Avg. Conversion Rate', xaxis_title='Lane Type', yaxis_title='Avg. Conversion Rate (%)')
            st.plotly_chart(fig, use_container_width=True)
            if pd.notna(median_overall_conversion):
                st.info(f"Overall Median Conversion Rate: {median_overall_conversion:.2f}%")
            else:
                st.info("Overall Median Conversion Rate could not be calculated (likely due to no valid data).")

        else:
            st.warning("LANE_TYPE, CONVERSION_RATE_PCT, DISTINCT_USER_IMPRESSIONS, or DISTINCT_USER_CLICKS column missing for this analysis.")

    with tab3: # Lane Type Trend (Top 5)
        if 'LANE_TYPE' in df_display_copy.columns and 'CONVERSION_RATE_PCT' in df_display_copy.columns and 'BASE_DATE' in df_display_copy.columns:
            temp_df_for_lane_trend = df_display_copy[['LANE_TYPE', 'CONVERSION_RATE_PCT', 'BASE_DATE']].copy()
            temp_df_for_lane_trend.dropna(subset=['LANE_TYPE', 'CONVERSION_RATE_PCT', 'BASE_DATE'], inplace=True)

            if temp_df_for_lane_trend.empty:
                st.warning("Not enough valid data (LANE_TYPE, CONVERSION_RATE_PCT, BASE_DATE) for lane trend analysis.")
            else:
                lane_avg_conv_series = temp_df_for_lane_trend.groupby('LANE_TYPE')['CONVERSION_RATE_PCT'].mean()
                lane_avg_conv_series = lane_avg_conv_series.dropna() 

                if lane_avg_conv_series.empty or not pd.api.types.is_numeric_dtype(lane_avg_conv_series):
                    st.warning("No lane types with valid numeric average conversion rates to display trend.")
                else:
                    lane_avg_conv = lane_avg_conv_series.nlargest(5).index.tolist()
                    if not lane_avg_conv:
                        st.warning("Not enough lane types with data to display trend after attempting to find top 5.")
                    else:
                        trend_lanes_df = df_display_copy[df_display_copy['LANE_TYPE'].isin(lane_avg_conv)].copy()
                        trend_lanes_df.dropna(subset=['BASE_DATE', 'CONVERSION_RATE_PCT'], inplace=True) # Ensure relevant columns are clean for plotting

                        time_period_lane = st.selectbox("Select Time Period", ["Daily", "Weekly", "Monthly"], key="lane_trend_period_tab3", index=0) # Unique key
                        
                        fig = go.Figure()
                        for lane in lane_avg_conv:
                            lane_data = trend_lanes_df[trend_lanes_df['LANE_TYPE'] == lane]
                            if lane_data.empty or lane_data['BASE_DATE'].isnull().all() or lane_data['CONVERSION_RATE_PCT'].isnull().all(): continue

                            if time_period_lane == "Daily":
                                plot_data = lane_data.groupby(lane_data['BASE_DATE'].dt.date)['CONVERSION_RATE_PCT'].mean().reset_index()
                            elif time_period_lane == "Weekly":
                                plot_data = lane_data.set_index('BASE_DATE').resample('W-MON')['CONVERSION_RATE_PCT'].mean().reset_index()
                            else: # Monthly
                                plot_data = lane_data.set_index('BASE_DATE').resample('M')['CONVERSION_RATE_PCT'].mean().reset_index()
                            
                            if not plot_data.empty:
                                 fig.add_trace(go.Scatter(x=plot_data['BASE_DATE'], y=plot_data['CONVERSION_RATE_PCT'], mode='lines+markers', name=lane))
                        
                        fig.update_layout(title=f'Top 5 Lane Types - {time_period_lane} Avg. Conversion Rate Trend', xaxis_title='Date', yaxis_title='Avg. Conversion Rate (%)')
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required columns (LANE_TYPE, CONVERSION_RATE_PCT, BASE_DATE) missing for lane trend.")

    with tab4: # Platform Distribution
        if 'DEVICE_PLATFORM' in df_display_copy.columns and 'DISTINCT_USER_IMPRESSIONS' in df_display_copy.columns and 'DISTINCT_USER_CLICKS' in df_display_copy.columns:
            platform_metrics = df_display_copy.groupby('DEVICE_PLATFORM').agg(
                total_impressions=('DISTINCT_USER_IMPRESSIONS', 'sum'),
                total_clicks=('DISTINCT_USER_CLICKS', 'sum')
            ).reset_index()
            platform_metrics['ctr'] = (platform_metrics['total_clicks'] / platform_metrics['total_impressions'] * 100).fillna(0)

            col_pie1, col_pie2 = st.columns(2)
            with col_pie1:
                fig_imp = px.pie(platform_metrics, values='total_impressions', names='DEVICE_PLATFORM', title='Impressions by Platform')
                st.plotly_chart(fig_imp, use_container_width=True)
            with col_pie2:
                fig_ctr = px.pie(platform_metrics, values='ctr', names='DEVICE_PLATFORM', title='Avg. CTR by Platform (%)')
                st.plotly_chart(fig_ctr, use_container_width=True)
        else:
            st.warning("Required columns missing for platform distribution.")
            
    with tab5: # Raw Data
        st.subheader("Filtered Raw Data")
        st.dataframe(df_display, use_container_width=True)
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Data (CSV)", csv, "filtered_analytics_data.csv", "text/csv", key='download-csv-main')

def display_precision_metrics(metrics_df_input):
    """Display precision metrics, including filters and visualizations."""
    if metrics_df_input is None or metrics_df_input.empty:
        st.info("Precision metrics data is not loaded or is empty.")
        # Attempt to load if not loaded and connection exists
        if st.session_state.conn and not st.session_state.get('metrics_data'):
            if st.button("Load Precision Metrics Data", key="load_pm_data_button_display"):
                with st.spinner("Loading precision metrics..."):
                    run_precision_metrics_query()
                    st.rerun() # Rerun to update display with new data
        return

    # Work with a copy, ensure BASE_DATE is datetime and WATCHED_ANY_RECOMMENDED is bool
    metrics_df = metrics_df_input.copy()
    if 'BASE_DATE' in metrics_df.columns:
        metrics_df['BASE_DATE'] = pd.to_datetime(metrics_df['BASE_DATE'], errors='coerce').dt.tz_localize(None).astype('datetime64[ns]')
    if 'WATCHED_ANY_RECOMMENDED' in metrics_df.columns: # Ensure boolean
        metrics_df['WATCHED_ANY_RECOMMENDED'] = metrics_df['WATCHED_ANY_RECOMMENDED'].astype(bool)

    with st.expander("📊 Precision Metrics Definitions", expanded=False):
        st.markdown("""
        #### Median Precision Ratio
        - **Definition**: Median of (watched recommendations / total recommendations) per user.
        #### Total Users
        - **Definition**: Count of distinct users who received recommendations.
        #### Users Who Watched Recommendations
        - **Definition**: Percentage of users who watched at least one recommended item.
        """)

    # --- Precision Metrics Filters ---
    with st.expander("Filters (Precision Metrics)", expanded=True):
        pm_col1, pm_col2 = st.columns(2)
        with pm_col1:
            # Date range for precision metrics
            pm_min_data_date, pm_max_data_date = get_date_range(data_key='metrics_data') # date objects
            
            pm_default_min_widget_date = (datetime.now() - timedelta(days=180)).date()
            pm_default_max_widget_date = datetime.now().date()

            pm_widget_min_val = pm_min_data_date if pm_min_data_date else pm_default_min_widget_date
            pm_widget_max_val = pm_max_data_date if pm_max_data_date else pm_default_max_widget_date
            if pm_widget_min_val > pm_widget_max_val: pm_widget_min_val = pm_widget_max_val

            # Initialize session state for precision_metrics_date_range if needed
            current_pm_range_ss = st.session_state.get('precision_metrics_date_range')
            if not (isinstance(current_pm_range_ss, tuple) and len(current_pm_range_ss) == 2 and
                    isinstance(current_pm_range_ss[0], date) and isinstance(current_pm_range_ss[1], date)):
                st.session_state.precision_metrics_date_range = (pm_widget_min_val, pm_widget_max_val)
            
            current_pm_start_val, current_pm_end_val = st.session_state.precision_metrics_date_range
            
            # Clamp value for widget to be within widget's own min/max
            pm_val_start_widget = current_pm_start_val if current_pm_start_val is not None else pm_widget_min_val
            pm_val_start_widget = max(pm_widget_min_val, min(pm_val_start_widget, pm_widget_max_val))
            
            pm_val_end_widget = current_pm_end_val if current_pm_end_val is not None else pm_widget_max_val
            pm_val_end_widget = max(pm_widget_min_val, min(pm_val_end_widget, pm_widget_max_val))
            
            if pm_val_start_widget > pm_val_end_widget: pm_val_start_widget = pm_val_end_widget
            
            st.date_input(
                "Select Date Range (Precision)",
                value=(pm_val_start_widget, pm_val_end_widget),
                min_value=pm_widget_min_val,
                max_value=pm_widget_max_val,
                key="precision_metrics_date_picker_key",
                on_change=precision_metrics_date_input_callback
            )

            # Registration status filter
            pm_reg_options = sorted([opt for opt in metrics_df['IS_REGISTERED'].unique() if pd.notna(opt)])
            st.multiselect(
                "Filter by Registration Status (Precision)",
                options=pm_reg_options,
                default=[val for val in st.session_state.get('precision_metrics_registration',[]) if val in pm_reg_options],
                key="pm_registration_filter_key",
                on_change=on_pm_reg_change # Use new callback
            )
        
        with pm_col2:
            # Watched status filter
            pm_watch_options_available = []
            if 'WATCHED_ANY_RECOMMENDED' in metrics_df.columns and \
               pd.api.types.is_bool_dtype(metrics_df['WATCHED_ANY_RECOMMENDED']):
                unique_statuses = metrics_df['WATCHED_ANY_RECOMMENDED'].unique()
                if True in unique_statuses:
                    pm_watch_options_available.append(True)
                if False in unique_statuses:
                    pm_watch_options_available.append(False)
                # Sort to ensure "Watched" (True) appears before "Not Watched" (False)
                pm_watch_options = sorted(pm_watch_options_available, key=lambda x: not x) 
            else:
                # This case should ideally not be hit if data loading and prep is correct
                # st.warning("WATCHED_ANY_RECOMMENDED column is missing or not boolean for filter options.")
                pm_watch_options = []


            st.multiselect(
                "Filter by Watched Status (Precision)",
                options=pm_watch_options, 
                format_func=lambda x: "Watched" if x is True else "Not Watched",
                default=[val for val in st.session_state.get('precision_metrics_watched',[]) if val in pm_watch_options],
                key="pm_watched_filter_key",
                on_change=on_pm_watch_change # Use new callback
            )

            # Distribution tenant filter
            pm_dist_options = sorted([opt for opt in metrics_df['DISTRIBUTION_TENANT'].unique() if pd.notna(opt)])
            st.multiselect(
                "Filter by Distribution Tenant (Precision)",
                options=pm_dist_options,
                default=[val for val in st.session_state.get('precision_metrics_distribution_tenant',[]) if val in pm_dist_options],
                key="pm_distribution_tenant_filter_key",
                on_change=on_pm_dist_change # Use new callback
            )
    
    # Apply filters to get the DataFrame for display
    filtered_pm_df = apply_precision_metrics_filters(metrics_df)

    if filtered_pm_df.empty:
        st.warning("No precision metrics data available after applying filters.")
        return

    # --- Display Calculated Metrics ---
    pm_metrics_container = st.container()
    with pm_metrics_container:
        # Calculate overall metrics from the filtered_pm_df
        if not filtered_pm_df.empty and 'MEDIAN_RECOMMENDATION_WATCH_RATIO' in filtered_pm_df.columns and \
           pd.api.types.is_numeric_dtype(filtered_pm_df['MEDIAN_RECOMMENDATION_WATCH_RATIO']) and \
           'TOTAL_USERS' in filtered_pm_df.columns and pd.api.types.is_numeric_dtype(filtered_pm_df['TOTAL_USERS']):
            
            # Weighted average of medians by total_users for that group
            # Ensure no division by zero if total_users_sum is 0
            total_users_sum = filtered_pm_df['TOTAL_USERS'].sum()
            if total_users_sum > 0:
                weighted_avg_median_ratio = (filtered_pm_df['MEDIAN_RECOMMENDATION_WATCH_RATIO'] * filtered_pm_df['TOTAL_USERS']).sum() / total_users_sum
            else:
                weighted_avg_median_ratio = 0 # Fallback if no users
        else:
            weighted_avg_median_ratio = 0 # Fallback if columns missing, not numeric, or df is empty

        overall_total_users = filtered_pm_df['TOTAL_USERS'].sum() if not filtered_pm_df.empty and 'TOTAL_USERS' in filtered_pm_df.columns else 0
        
        users_watched_df = filtered_pm_df[filtered_pm_df['WATCHED_ANY_RECOMMENDED'] == True] if not filtered_pm_df.empty and 'WATCHED_ANY_RECOMMENDED' in filtered_pm_df.columns else pd.DataFrame(columns=['TOTAL_USERS'])
        overall_watched_users_count = users_watched_df['TOTAL_USERS'].sum() if not users_watched_df.empty else 0
        
        percent_users_watched = (overall_watched_users_count / overall_total_users * 100) if overall_total_users > 0 else 0

        pm_m_col1, pm_m_col2, pm_m_col3 = st.columns(3)
        pm_m_col1.metric("Avg. Median Precision Ratio", f"{weighted_avg_median_ratio:.2%}") # Changed from overall_median_ratio
        pm_m_col2.metric("Total Users (Filtered)", f"{overall_total_users:,.0f}")
        pm_m_col3.metric("% Users Watched Reco", f"{percent_users_watched:.2f}%", f"({overall_watched_users_count:,.0f} users)")

    # --- Trend Graphs ---
    st.markdown("#### Monthly Trends (Precision Metrics)")
    pm_trend_df = filtered_pm_df.copy()
    
    if pm_trend_df.empty or 'BASE_DATE' not in pm_trend_df.columns or pm_trend_df['BASE_DATE'].isnull().all():
        st.warning("Not enough data for precision trend graphs.")
        return

    pm_trend_df['MONTH_YEAR'] = pm_trend_df['BASE_DATE'].dt.to_period('M')

    # Trend 1: % Users Who Watched Recommendations
    monthly_watched_trend = pm_trend_df.groupby('MONTH_YEAR').agg(
        total_users_month=('TOTAL_USERS', 'sum'),
        watched_users_month=('TOTAL_USERS', lambda x: x[pm_trend_df.loc[x.index, 'WATCHED_ANY_RECOMMENDED'] == True].sum()) # Sum users where watched is true
    ).reset_index()
    monthly_watched_trend['percent_watched'] = (monthly_watched_trend['watched_users_month'] / monthly_watched_trend['total_users_month'] * 100).fillna(0)
    monthly_watched_trend['MONTH_YEAR_DT'] = monthly_watched_trend['MONTH_YEAR'].dt.to_timestamp() # For plotting

    fig_watched_trend = px.line(monthly_watched_trend, x='MONTH_YEAR_DT', y='percent_watched', title='% Users Who Watched Recommendations (Monthly)')
    fig_watched_trend.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_watched_trend, use_container_width=True)

    # Trend 2: Median Precision Ratio
    monthly_precision_trend = pm_trend_df.groupby('MONTH_YEAR')['MEDIAN_RECOMMENDATION_WATCH_RATIO'].mean().reset_index() # Mean of medians
    monthly_precision_trend['MONTH_YEAR_DT'] = monthly_precision_trend['MONTH_YEAR'].dt.to_timestamp()

    fig_precision_trend = px.line(monthly_precision_trend, x='MONTH_YEAR_DT', y='MEDIAN_RECOMMENDATION_WATCH_RATIO', title='Avg. Median Precision Ratio (Monthly)')
    fig_precision_trend.update_yaxes(tickformat=".2%")
    st.plotly_chart(fig_precision_trend, use_container_width=True)
    
    # Raw Data for Precision Metrics
    with st.expander("Filtered Raw Precision Metrics Data", expanded=False):
        st.dataframe(filtered_pm_df, use_container_width=True)
        csv_pm = filtered_pm_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered PM Data (CSV)", csv_pm, "filtered_precision_metrics.csv", "text/csv", key='download-csv-pm')


# --- DuckDB Notes Functions (assumed to be largely correct from summary) ---
@st.cache_resource
def init_duckdb():
    conn = duckdb.connect('notes.db', read_only=False) # Ensure not read_only for creation
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS notes_id_seq;
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY DEFAULT nextval('notes_id_seq'),
            note_date DATE DEFAULT CURRENT_DATE,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'General'
        );
    """)
    return conn

def add_note(content, note_date_obj, category='General'):
    db = init_duckdb()
    db.execute("INSERT INTO notes (content, note_date, category) VALUES (?, ?, ?)", [content, note_date_obj, category])

def get_notes_by_date(date_obj):
    db = init_duckdb()
    return db.execute("SELECT id, strftime(timestamp, '%Y-%m-%d %H:%M') as timestamp_str, content, category FROM notes WHERE note_date = ? ORDER BY timestamp DESC", [date_obj]).fetchdf()

def get_note_dates(): # Returns unique dates with notes
    db = init_duckdb()
    return db.execute("SELECT DISTINCT note_date FROM notes ORDER BY note_date DESC").fetchdf()

# Initialize DuckDB
db_conn = init_duckdb()


# --- Main App Layout ---
st.title("❄️ Recommendations Analytics Dashboard")

# --- Sidebar ---
st.sidebar.header("Connection & Debug")
if st.session_state.conn:
    st.sidebar.success("✅ Connected to Snowflake")
else:
    st.sidebar.warning("⚠️ Not Connected")
if st.sidebar.button("Connect/Reconnect to Snowflake", key="sidebar_connect_button"):
    with st.spinner("Connecting..."):
        test_connection() # This will update st.session_state.conn

st.sidebar.markdown("---")
st.sidebar.header("Load Data")
if st.sidebar.button("Load Analytics Data", key="sidebar_load_analytics"):
    if st.session_state.conn:
        with st.spinner("Loading analytics data..."):
            run_analytics_query() # This updates session state and applies initial filters
            # No explicit rerun needed here as run_analytics_query handles it by apply_filters
    else:
        st.sidebar.error("Connect to Snowflake first.")

if st.sidebar.button("Load Precision Metrics Data", key="sidebar_load_precision"):
    if st.session_state.conn:
        with st.spinner("Loading precision metrics..."):
            run_precision_metrics_query() # This updates session state
            st.session_state.update_precision_metrics_display = True # Flag to refresh display part
            # Rerun might be good if display logic depends on immediate data
    else:
        st.sidebar.error("Connect to Snowflake first.")

st.sidebar.markdown("---")
st.sidebar.header("Main Filters (Analytics)")
if st.session_state.data_loaded:
    # Date Range Filter for Main Analytics
    min_data_date_main, max_data_date_main = get_date_range(data_key='data')
    
    default_min_widget_main = (datetime.now() - timedelta(days=30)).date()
    default_max_widget_main = datetime.now().date()

    widget_min_main = min_data_date_main if min_data_date_main else default_min_widget_main
    widget_max_main = max_data_date_main if max_data_date_main else default_max_widget_main
    if widget_min_main > widget_max_main: widget_min_main = widget_max_main
    
    # Initialize date_range in session state if not correctly set
    current_main_range_ss = st.session_state.get('date_range')
    if not (isinstance(current_main_range_ss, tuple) and len(current_main_range_ss) == 2 and
            (isinstance(current_main_range_ss[0], date) or current_main_range_ss[0] is None) and
            (isinstance(current_main_range_ss[1], date) or current_main_range_ss[1] is None)):
        st.session_state.date_range = (widget_min_main, widget_max_main) # Use widget bounds for init

    current_start_main_val, current_end_main_val = st.session_state.date_range
    
    # Value for widget (must be concrete dates, not None)
    val_start_main_widget = current_start_main_val if current_start_main_val else widget_min_main
    val_end_main_widget = current_end_main_val if current_end_main_val else widget_max_main

    # Clamp to widget's own min/max
    val_start_main_widget = max(widget_min_main, min(val_start_main_widget, widget_max_main))
    val_end_main_widget = max(widget_min_main, min(val_end_main_widget, widget_max_main))
    if val_start_main_widget > val_end_main_widget: val_start_main_widget = val_end_main_widget

    st.sidebar.date_input(
        "Date Range (Analytics)",
        value=(val_start_main_widget, val_end_main_widget),
        min_value=widget_min_main,
        max_value=widget_max_main,
        key="main_date_picker_key",
        on_change=main_date_input_callback
    )

    # Date Presets for Main Analytics
    date_presets_options = ["Custom", "Full range", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
    current_preset_main = st.session_state.get("date_preset", "Full range")
    if current_preset_main not in date_presets_options: current_preset_main = "Full range"
    
    st.sidebar.selectbox(
        "Date Presets (Analytics)",
        options=date_presets_options,
        index=date_presets_options.index(current_preset_main),
        key="main_date_preset_selector_key",
        on_change=main_date_preset_callback
    )

    # Other filters for Main Analytics
    filter_configs = [
        ("Lane Type", 'lane_type_filter', 'LANE_TYPE', 'lane_type_multiselect_key'),
        ("Registration Status", 'is_registered_filter', 'IS_REGISTERED', 'is_registered_multiselect_key'),
        ("Device Platform", 'device_platform_filter', 'DEVICE_PLATFORM', 'device_platform_multiselect_key'),
        ("Distribution Tenant", 'distribution_tenant_filter', 'DISTRIBUTION_TENANT', 'distribution_tenant_multiselect_key')
    ]

    for label, session_key, col_name, widget_key in filter_configs:
        if st.session_state.data is not None and col_name in st.session_state.data.columns:
            options = st.session_state.data[col_name].unique().tolist()
            options_display = sorted([str(opt) if opt is not None else 'None' for opt in options])
            
            current_filter_values_str = [str(v) if v is not None else 'None' for v in st.session_state.get(session_key, [])]
            valid_defaults = [v for v in current_filter_values_str if v in options_display]

            selected_display_values = st.sidebar.multiselect(
                label,
                options=options_display,
                default=valid_defaults,
                key=widget_key
            )
            selected_actual_values = [None if val_str == 'None' else val_str for val_str in selected_display_values]
            
            if selected_actual_values != st.session_state.get(session_key):
                st.session_state[session_key] = selected_actual_values
                apply_filters()
        else:
            st.sidebar.markdown(f"_{label} filter unavailable (column missing)._")

else:
    st.sidebar.info("Load analytics data to enable filters.")


# --- Main Content Area using Tabs ---
tab_titles = ["📊 Analytics Dashboard", "🎯 Precision Metrics", "📝 Notes"]
tabs = st.tabs(tab_titles)

with tabs[0]: # Analytics Dashboard
    st.header("Lane Performance Analytics")
    if st.session_state.data_loaded and st.session_state.filtered_data is not None:
        display_data_summary(st.session_state.filtered_data)
        st.markdown("---")
        display_data_visualization(st.session_state.filtered_data)
    elif st.session_state.data_loaded and st.session_state.filtered_data is None:
        st.warning("Data loaded, but filtered data is not available. Try adjusting filters or reloading.")
        apply_filters() # Attempt to apply filters again
    else:
        st.info("Load analytics data using the sidebar button to view the dashboard.")

with tabs[1]: # Precision Metrics
    st.header("Recommendation Precision Metrics")
    if st.session_state.get('metrics_data') is not None:
        display_precision_metrics(st.session_state.metrics_data)
        if st.session_state.get('update_precision_metrics_display'): 
            st.session_state.update_precision_metrics_display = False 
            # display_precision_metrics already called, natural rerun will occur if state changed
    else:
        st.info("Load precision metrics data using the sidebar button to view this section.")
        # Offer a button to load here as well for convenience
        if st.session_state.conn:
            if st.button("Load Precision Metrics Data Here", key="load_pm_data_tab_button"):
                with st.spinner("Loading precision metrics..."):
                    run_precision_metrics_query()

with tabs[2]: # Notes
    st.header("Notes & Observations")
    note_date_input = st.date_input("Note Date", value=datetime.now().date(), key="note_main_date")
    note_content = st.text_area("Note Content", key="note_main_content")
    note_category = st.selectbox("Category", ["General", "Issue", "Observation", "Action Item"], key="note_main_category")

    if st.button("Add Note", key="add_note_main_button"):
        if note_content and note_date_input:
            add_note(note_content, note_date_input, note_category)
            st.success(f"Note added for {note_date_input.strftime('%Y-%m-%d')}!")
            # Clear inputs after adding
            st.session_state.note_main_content = "" 
        else:
            st.error("Please provide content and a date for the note.")

    st.markdown("---")
    st.subheader("Recent Notes by Date")
    
    # Display unique dates with notes for selection
    unique_dates_df = get_note_dates() # df with 'note_date' column
    if not unique_dates_df.empty:
        # Convert to list of date objects for selectbox, ensure they are strings for display
        # And handle potential non-date objects if any slip through (though unlikely from DB)
        date_options = sorted([d.strftime('%Y-%m-%d') for d in unique_dates_df['note_date'] if isinstance(d, date)], reverse=True)

        if date_options:
            selected_date_str = st.selectbox("View notes for date:", options=date_options, key="view_notes_date_select")
            if selected_date_str:
                selected_date_obj = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
                notes_for_date_df = get_notes_by_date(selected_date_obj)
                if not notes_for_date_df.empty:
                    st.dataframe(notes_for_date_df[['timestamp_str', 'content', 'category', 'id']], use_container_width=True)
                else:
                    st.info(f"No notes found for {selected_date_str}.")
        else:
            st.info("No dates with notes found.")
    else:
        st.info("No notes added yet.")

# --- Footer (Optional) ---
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: grey;'>Dashboard v1.0 | Python: {sys.version.split(' ')[0]} | Streamlit: {st.__version__} | Snowflake Connector: {snowflake.connector.__version__}</p>", unsafe_allow_html=True)

# Final check for reruns if flags are set (e.g., after data loading)