import streamlit as st
import snowflake.connector
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import sys
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Snowflake Analytics Dashboard",
    page_icon="❄️",
    layout="wide"
)

# Debug information in sidebar
st.sidebar.header("Debug Information")
st.sidebar.write("Python Version:", sys.version)
st.sidebar.write("Snowflake Connector Version:", snowflake.connector.__version__)

# Environment variables status
st.sidebar.header("Environment Variables")
user = os.getenv('SNOWFLAKE_USER')
account = os.getenv('SNOWFLAKE_ACCOUNT')
warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
st.sidebar.write(f"SNOWFLAKE_USER: {'✅ Set' if user else '❌ Not Set'}")
st.sidebar.write(f"SNOWFLAKE_ACCOUNT: {'✅ Set' if account else '❌ Not Set'}")
st.sidebar.write(f"SNOWFLAKE_WAREHOUSE: {'✅ Set' if warehouse else '❌ Not Set'}")

# Initialize session state for connection and data
if 'conn' not in st.session_state:
    st.session_state.conn = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'date_range' not in st.session_state:
    st.session_state.date_range = None
if 'lane_type_filter' not in st.session_state:
    st.session_state.lane_type_filter = []
if 'is_registered_filter' not in st.session_state:
    st.session_state.is_registered_filter = []
if 'date_preset' not in st.session_state:
    st.session_state.date_preset = None

# Main app
st.title("❄️ Snowflake Analytics Dashboard")

def test_connection():
    """Test connection to Snowflake and display detailed information"""
    try:
        # Validate environment variables
        if not user or not account:
            st.error("❌ Missing required environment variables")
            st.error("Please check your .env file and ensure SNOWFLAKE_USER and SNOWFLAKE_ACCOUNT are set")
            return None
            
        if not warehouse:
            st.error("❌ Missing required warehouse")
            st.error("Please check your .env file and ensure SNOWFLAKE_WAREHOUSE is set")
            return None
        
        # Display connection attempt
        st.info(f"🔄 Attempting to connect to Snowflake account: {account}")
        
        # Create connection parameters
        conn_params = {
            'user': user,
            'account': account,
            'authenticator': 'externalbrowser',
            'client_session_keep_alive': True,
            'warehouse': warehouse  # Always include warehouse
        }
        
        # Add optional parameters if they exist
        for param in ['SNOWFLAKE_ROLE', 'SNOWFLAKE_DATABASE', 'SNOWFLAKE_SCHEMA']:
            value = os.getenv(param)
            if value:
                param_name = param.replace('SNOWFLAKE_', '').lower()
                conn_params[param_name] = value
                st.write(f"Using {param_name}: {value}")
        
        # Establish connection
        st.write("Establishing connection...")
        conn = snowflake.connector.connect(**conn_params)
        
        # Test connection with a simple query
        st.write("Testing connection with a simple query...")
        cursor = conn.cursor()
        cursor.execute("SELECT current_version()")
        version = cursor.fetchone()[0]
        
        # Get account information
        cursor.execute("SELECT current_account(), current_region(), current_warehouse(), current_database(), current_schema()")
        account_info = cursor.fetchone()
        
        # Display success information
        st.success("✅ Successfully connected to Snowflake!")
        
        # Display connection details
        st.subheader("Connection Details")
        st.write(f"**Snowflake Version:** {version}")
        st.write(f"**Account:** {account_info[0]}")
        st.write(f"**Region:** {account_info[1]}")
        st.write(f"**Warehouse:** {account_info[2]}")
        st.write(f"**Database:** {account_info[3]}")
        st.write(f"**Schema:** {account_info[4]}")
        
        # Store connection in session state
        st.session_state.conn = conn
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error connecting to Snowflake: {str(e)}")
        st.error("Please check your credentials and network connection")
        return None

def run_analytics_query():
    """Execute the analytics query and return the results as a DataFrame"""
    try:
        if not st.session_state.conn:
            st.error("❌ No connection to Snowflake. Please connect first.")
            return None
            
        st.info("🔄 Running analytics query...")
        
        # Ensure warehouse is selected
        cursor = st.session_state.conn.cursor()
        cursor.execute(f"USE WAREHOUSE {warehouse}")
        st.write(f"✅ Using warehouse: {warehouse}")
        
        query = """
        WITH correct_lane_views_f AS (
            SELECT *,
                playground.dani.standardize_lane_type(list_type, list_name) AS lane_type,
                case when user_id like 'JNDE%' then 'yes' else 'no' end as is_registered,
            FROM joyn_snow.im.lane_views_f
            WHERE base_date > dateadd(DAY, -90, CURRENT_DATE)
        ),
        correct_as_f AS (
            SELECT *,
                playground.dani.standardize_lane_type(lane_type, lane_label) AS rlane_type
            FROM joyn_snow.im.asset_select_f
            WHERE base_date > dateadd(DAY, -90, CURRENT_DATE)
        )
        SELECT 
            a.base_date,a.lane_type,a.is_registered,
            HLL(DISTINCT a.user_id) as distinct_user_impressions,
            HLL(DISTINCT CASE WHEN b.user_id IS NOT NULL THEN b.user_id END) AS distinct_user_clicks,
            ROUND((HLL(CASE WHEN b.user_id IS NOT NULL THEN b.user_id END) / NULLIF(COUNT(DISTINCT a.user_id), 0)) * 100, 2) AS conversion_rate_pct 
        FROM correct_lane_views_f a 
        LEFT JOIN correct_as_f b ON 
        (a.user_id = b.user_id and a.user_id like 'JNDE%' and a.lane_type = b.rlane_type AND datediff(day, a.base_date, b.base_date) < 8 and b.base_date >= a.base_date)
        GROUP BY all 
        order by 1 asc;
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        
        # Convert base_date to datetime
        df['BASE_DATE'] = pd.to_datetime(df['BASE_DATE'])
        
        # Store data in session state
        st.session_state.data = df
        st.session_state.data_loaded = True
        st.session_state.filtered_data = df.copy()
        
        # Initialize date range to full range
        if st.session_state.date_range is None:
            st.session_state.date_range = (df['BASE_DATE'].min(), df['BASE_DATE'].max())
        
        st.success(f"✅ Query executed successfully! Retrieved {len(df)} rows.")
        return df
        
    except Exception as e:
        st.error(f"❌ Error executing query: {str(e)}")
        if "No active warehouse selected" in str(e):
            st.error("Please check your .env file and ensure SNOWFLAKE_WAREHOUSE is set correctly")
        return None

def apply_filters():
    """Apply selected filters to the data"""
    if st.session_state.data is None:
        return None
        
    filtered_df = st.session_state.data.copy()
    
    # Apply date range filter
    if st.session_state.date_range:
        start_date, end_date = st.session_state.date_range
        filtered_df = filtered_df[(filtered_df['BASE_DATE'] >= start_date) & (filtered_df['BASE_DATE'] <= end_date)]
    
    # Apply lane_type filter
    if st.session_state.lane_type_filter:
        filtered_df = filtered_df[filtered_df['LANE_TYPE'].isin(st.session_state.lane_type_filter)]
    
    # Apply is_registered filter
    if st.session_state.is_registered_filter:
        filtered_df = filtered_df[filtered_df['IS_REGISTERED'].isin(st.session_state.is_registered_filter)]
    
    st.session_state.filtered_data = filtered_df
    return filtered_df

def apply_date_preset(preset):
    """Apply a date preset to the date range filter"""
    if st.session_state.data is None:
        return
    
    today = datetime.now()
    max_date = st.session_state.data['BASE_DATE'].max()
    
    if preset == "Last 7 days":
        start_date = max_date - timedelta(days=7)
        end_date = max_date
    elif preset == "Last 14 days":
        start_date = max_date - timedelta(days=14)
        end_date = max_date
    elif preset == "Last 30 days":
        start_date = max_date - timedelta(days=30)
        end_date = max_date
    elif preset == "Last 90 days":
        start_date = max_date - timedelta(days=90)
        end_date = max_date
    elif preset == "Last 180 days":
        start_date = max_date - timedelta(days=180)
        end_date = max_date
    elif preset == "Last 365 days":
        start_date = max_date - timedelta(days=365)
        end_date = max_date
    elif preset == "This month":
        start_date = datetime(today.year, today.month, 1)
        end_date = max_date
    elif preset == "Last month":
        if today.month == 1:
            start_date = datetime(today.year - 1, 12, 1)
        else:
            start_date = datetime(today.year, today.month - 1, 1)
        end_date = datetime(today.year, today.month, 1) - timedelta(days=1)
    elif preset == "This year":
        start_date = datetime(today.year, 1, 1)
        end_date = max_date
    elif preset == "Last year":
        start_date = datetime(today.year - 1, 1, 1)
        end_date = datetime(today.year, 1, 1) - timedelta(days=1)
    elif preset == "All time":
        start_date = st.session_state.data['BASE_DATE'].min()
        end_date = max_date
    else:
        return
    
    st.session_state.date_range = (start_date, end_date)
    st.session_state.date_preset = preset
    apply_filters()
    st.rerun()

# Connection section
st.sidebar.header("Snowflake Connection")
if st.sidebar.button("Connect to Snowflake"):
    with st.spinner("Connecting to Snowflake..."):
        test_connection()

# Data loading section
st.sidebar.header("Data Loading")
if st.sidebar.button("Load Analytics Data"):
    with st.spinner("Loading data..."):
        run_analytics_query()

# Filters section
st.sidebar.header("Filters")
if st.session_state.data_loaded:
    # Date presets filter
    st.sidebar.subheader("Date Presets")
    date_presets = [
        "Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days", 
        "Last 180 days", "Last 365 days", "This month", "Last month", 
        "This year", "Last year", "All time"
    ]
    
    selected_preset = st.sidebar.selectbox(
        "Select Date Preset",
        options=["Custom"] + date_presets,
        index=0 if st.session_state.date_preset is None else date_presets.index(st.session_state.date_preset) + 1,
        key="date_preset_select"
    )
    
    if selected_preset != "Custom" and selected_preset != st.session_state.date_preset:
        apply_date_preset(selected_preset)
    
    # Date range filter
    st.sidebar.subheader("Custom Date Range")
    min_date = st.session_state.data['BASE_DATE'].min()
    max_date = st.session_state.data['BASE_DATE'].max()
    
    # Initialize date range if not set
    if st.session_state.date_range is None:
        st.session_state.date_range = (min_date, max_date)
    
    # Date range picker
    selected_dates = st.sidebar.date_input(
        "Select Date Range",
        value=(st.session_state.date_range[0], st.session_state.date_range[1]),
        min_value=min_date.date(),
        max_value=max_date.date(),
        key="date_range_picker"
    )
    
    # Update date range in session state
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
        # Convert to datetime for comparison
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        if (start_datetime, end_datetime) != st.session_state.date_range:
            st.session_state.date_range = (start_datetime, end_datetime)
            st.session_state.date_preset = None  # Reset preset when custom range is selected
            apply_filters()
            st.rerun()
    
    # Lane type filter
    st.sidebar.subheader("Lane Type")
    # Get unique lane types and handle None values
    unique_lane_types = st.session_state.data['LANE_TYPE'].unique().tolist()
    # Replace None with 'None' for display purposes
    unique_lane_types = ['None' if x is None else x for x in unique_lane_types]
    # Sort the list, handling the 'None' string appropriately
    unique_lane_types = sorted([x for x in unique_lane_types if x != 'None'] + ['None'])
    
    # Create multiselect for lane_type
    selected_lane_types = st.sidebar.multiselect(
        "Filter by Lane Type",
        options=unique_lane_types,
        default=st.session_state.lane_type_filter,
        key="lane_type_multiselect"
    )
    
    # Convert 'None' string back to None for filtering
    selected_lane_types = [None if x == 'None' else x for x in selected_lane_types]
    
    # Update lane type filter in session state
    if selected_lane_types != st.session_state.lane_type_filter:
        st.session_state.lane_type_filter = selected_lane_types
        apply_filters()
        st.rerun()
    
    # Registration status filter
    st.sidebar.subheader("Registration Status")
    unique_registered = sorted(st.session_state.data['IS_REGISTERED'].unique().tolist())
    
    # Create multiselect for is_registered
    selected_registered = st.sidebar.multiselect(
        "Filter by Registration Status",
        options=unique_registered,
        default=st.session_state.is_registered_filter,
        key="is_registered_multiselect"
    )
    
    # Update registration filter in session state
    if selected_registered != st.session_state.is_registered_filter:
        st.session_state.is_registered_filter = selected_registered
        apply_filters()
        st.rerun()

# Main content
if st.session_state.conn:
    st.sidebar.success("✅ Connected to Snowflake")
    
    if st.session_state.data_loaded:
        st.sidebar.success("✅ Data loaded successfully")
        
        # Display data summary
        st.subheader("Data Summary")
        st.write(f"Total Records: {len(st.session_state.data)}")
        st.write(f"Filtered Records: {len(st.session_state.filtered_data)}")
        
        # Display filter summary
        with st.expander("Filter Summary"):
            if st.session_state.date_range:
                start_date, end_date = st.session_state.date_range
                st.write(f"**Date Range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                if st.session_state.date_preset:
                    st.write(f"**Date Preset:** {st.session_state.date_preset}")
            
            if st.session_state.lane_type_filter:
                st.write(f"**Lane Types:** {', '.join(st.session_state.lane_type_filter)}")
            else:
                st.write("**Lane Types:** All")
                
            if st.session_state.is_registered_filter:
                st.write(f"**Registration Status:** {', '.join(st.session_state.is_registered_filter)}")
            else:
                st.write("**Registration Status:** All")
        
        # Display raw data
        with st.expander("View Raw Data"):
            st.dataframe(st.session_state.filtered_data, use_container_width=True)
        
        # Display trendline graphs
        st.subheader("Trend Analysis")
        
        # Prepare data for trendlines
        if len(st.session_state.filtered_data) > 0:
            # Group by date to get daily metrics
            daily_metrics = st.session_state.filtered_data.groupby('BASE_DATE').agg({
                'DISTINCT_USER_IMPRESSIONS': 'sum',
                'DISTINCT_USER_CLICKS': 'sum',
                'CONVERSION_RATE_PCT': 'mean'
            }).reset_index()
            
            # Calculate clickthrough rate
            daily_metrics['CLICKTHROUGH_RATE'] = (daily_metrics['DISTINCT_USER_CLICKS'] / daily_metrics['DISTINCT_USER_IMPRESSIONS'] * 100).round(2)
            
            # Create tabs for different metrics
            tab1, tab2, tab3, tab4 = st.tabs(["Impressions & Clicks", "Conversion Rate", "Clickthrough Rate", "Combined View"])
            
            with tab1:
                # Plot impressions and clicks
                fig_imp_clicks = go.Figure()
                fig_imp_clicks.add_trace(go.Scatter(
                    x=daily_metrics['BASE_DATE'], 
                    y=daily_metrics['DISTINCT_USER_IMPRESSIONS'],
                    name='Impressions',
                    line=dict(color='blue')
                ))
                fig_imp_clicks.add_trace(go.Scatter(
                    x=daily_metrics['BASE_DATE'], 
                    y=daily_metrics['DISTINCT_USER_CLICKS'],
                    name='Clicks',
                    line=dict(color='red')
                ))
                fig_imp_clicks.update_layout(
                    title='Daily Impressions and Clicks Over Time',
                    xaxis_title='Date',
                    yaxis_title='Count',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_imp_clicks, use_container_width=True)
            
            with tab2:
                # Plot conversion rate
                fig_conv = px.line(
                    daily_metrics, 
                    x='BASE_DATE', 
                    y='CONVERSION_RATE_PCT',
                    title='Daily Conversion Rate Over Time'
                )
                fig_conv.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Conversion Rate (%)'
                )
                st.plotly_chart(fig_conv, use_container_width=True)
            
            with tab3:
                # Plot clickthrough rate
                fig_ctr = px.line(
                    daily_metrics, 
                    x='BASE_DATE', 
                    y='CLICKTHROUGH_RATE',
                    title='Daily Clickthrough Rate Over Time'
                )
                fig_ctr.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Clickthrough Rate (%)'
                )
                st.plotly_chart(fig_ctr, use_container_width=True)
            
            with tab4:
                # Combined view with secondary y-axis
                fig_combined = go.Figure()
                
                # Add impressions trace
                fig_combined.add_trace(go.Scatter(
                    x=daily_metrics['BASE_DATE'], 
                    y=daily_metrics['DISTINCT_USER_IMPRESSIONS'],
                    name='Impressions',
                    line=dict(color='blue')
                ))
                
                # Add clicks trace
                fig_combined.add_trace(go.Scatter(
                    x=daily_metrics['BASE_DATE'], 
                    y=daily_metrics['DISTINCT_USER_CLICKS'],
                    name='Clicks',
                    line=dict(color='red')
                ))
                
                # Add conversion rate trace with secondary y-axis
                fig_combined.add_trace(go.Scatter(
                    x=daily_metrics['BASE_DATE'], 
                    y=daily_metrics['CONVERSION_RATE_PCT'],
                    name='Conversion Rate (%)',
                    line=dict(color='green'),
                    yaxis='y2'
                ))
                
                # Add clickthrough rate trace with secondary y-axis
                fig_combined.add_trace(go.Scatter(
                    x=daily_metrics['BASE_DATE'], 
                    y=daily_metrics['CLICKTHROUGH_RATE'],
                    name='Clickthrough Rate (%)',
                    line=dict(color='purple'),
                    yaxis='y2'
                ))
                
                # Update layout with secondary y-axis
                fig_combined.update_layout(
                    title='Combined Metrics Over Time',
                    xaxis_title='Date',
                    yaxis_title='Count',
                    yaxis2=dict(
                        title='Rate (%)',
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_combined, use_container_width=True)
        else:
            st.warning("No data available for trend analysis with current filters.")
    else:
        st.info("👈 Click 'Load Analytics Data' in the sidebar to analyze the dataset.")
else:
    st.info("👈 Please connect to Snowflake using the sidebar button to begin.")

# Instructions
with st.expander("Connection Instructions"):
    st.markdown("""
    ### How to set up your .env file:
    
    1. Create a `.env` file in the same directory as this app
    2. Add the following variables:
    ```
    SNOWFLAKE_USER=your_username
    SNOWFLAKE_ACCOUNT=your_account.region.cloud
    ```
    
    3. Optional parameters:
    ```
    SNOWFLAKE_ROLE=your_role
    SNOWFLAKE_WAREHOUSE=your_warehouse
    SNOWFLAKE_DATABASE=your_database
    SNOWFLAKE_SCHEMA=your_schema
    ```
    
    ### Troubleshooting:
    
    - Make sure your account identifier is in the format: `account.region.cloud`
    - Do not include `https://` or `.snowflakecomputing.com` in the account
    - Ensure you have SSO access configured in Snowflake
    - Check that your network allows browser-based authentication
    """) 