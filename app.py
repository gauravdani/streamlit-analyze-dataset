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
from datetime import datetime, timedelta
import duckdb
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

def display_data_summary(df):
    """Display summary statistics and metrics for the filtered data"""
    if df is None or df.empty:
        st.warning("No data available for summary")
        return

    # Calculate key metrics
    total_impressions = df['DISTINCT_USER_IMPRESSIONS'].sum()
    total_clicks = df['DISTINCT_USER_CLICKS'].sum()
    avg_conversion_rate = df['CONVERSION_RATE_PCT'].mean()
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Impressions", f"{total_impressions:,.0f}")
    with col2:
        st.metric("Total Clicks", f"{total_clicks:,.0f}")
    with col3:
        st.metric("Average Conversion Rate", f"{avg_conversion_rate:.2f}%")

    # Display data summary in an expander
    with st.expander("Data Overview", expanded=False):
        st.dataframe(df.describe(), use_container_width=True)
    
    # Display metrics definitions
    with st.expander("Metrics Definitions", expanded=False):
        # Get the current query from session state, or a default message if not foundnt query from session state, or a default message if not foundy = st.session_state.get('current_query', "Analytics query has not been run yet or is not available.")
        query_to_display = st.session_state.get('current_query', "Analytics query has not been run yet or is not available.")
        st.markdown(f"""
        st.markdown(f"""
        ### Key Metrics Definitions
        pes
        - **Total Impressions**: The total number of unique user impressions across all lane types
        - **Total Clicks**: The total number of unique user clicks across all lane types- **Conversion Rate**: The percentage of unique users who clicked after being impressed
        - **Conversion Rate**: The percentage of unique users who clicked after being impressedRate**: The middle value of all conversion rates, separating the higher half from the lower half
        - **Median Conversion Rate**: The middle value of all conversion rates, separating the higher half from the lower half
        
        ### Additional Metrics
        rates
        - **Standard Deviation**: Measures the amount of variation or dispersion in the conversion rates        - **Minimum/Maximum**: The lowest and highest conversion rates observed
        - **Minimum/Maximum**: The lowest and highest conversion rates observede**: The values below which 25% and 75% of the conversion rates fall
        - **25th/75th Percentile**: The values below which 25% and 75% of the conversion rates fall

        ### Lane Analysis Query
        ```sql
{query_to_display}
        ```
        """)

def display_data_visualization(df):
    """Display visualizations for the filtered data"""
    if df is None or df.empty:
        st.warning("No data available for visualization")
        return

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Conversion Rate Trend", "Lane Type Analysis", "Lane Type Trend", "Platform Distribution", "Raw Data"])
    
    with tab1:
        # Add time period selector
        time_period = st.selectbox(
            "Select Time Period",
            options=["Daily", "Weekly", "Monthly"],
            index=0,
            key="conversion_trend_period"
        )
        
        # Daily conversion rate trend
        if time_period == "Daily":
            daily_conversion = df.groupby('BASE_DATE')['CONVERSION_RATE_PCT'].mean().reset_index()
            fig = px.line(daily_conversion, x='BASE_DATE', y='CONVERSION_RATE_PCT',
                         title='Daily Conversion Rate Trend')
        elif time_period == "Weekly":
            df['WEEK'] = df['BASE_DATE'].dt.isocalendar().week
            df['YEAR'] = df['BASE_DATE'].dt.isocalendar().year
            weekly_conversion = df.groupby(['YEAR', 'WEEK'])['CONVERSION_RATE_PCT'].mean().reset_index()
            weekly_conversion['DATE'] = pd.to_datetime(weekly_conversion['YEAR'].astype(str) + '-W' + 
                                                     weekly_conversion['WEEK'].astype(str) + '-1', format='%Y-W%W-%w')
            fig = px.line(weekly_conversion, x='DATE', y='CONVERSION_RATE_PCT',
                         title='Weekly Conversion Rate Trend')
        else:  # Monthly
            df['MONTH'] = df['BASE_DATE'].dt.to_period('M')
            monthly_conversion = df.groupby('MONTH')['CONVERSION_RATE_PCT'].mean().reset_index()
            monthly_conversion['DATE'] = pd.to_datetime(monthly_conversion['MONTH'].astype(str) + '-01')
            fig = px.line(monthly_conversion, x='DATE', y='CONVERSION_RATE_PCT',
                         title='Monthly Conversion Rate Trend')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Lane type analysis
        lane_type_metrics = df.groupby('LANE_TYPE').agg({
            'DISTINCT_USER_IMPRESSIONS': 'sum',
            'DISTINCT_USER_CLICKS': 'sum',
            'CONVERSION_RATE_PCT': 'mean'
        }).reset_index()
        
        # Sort by conversion rate in descending order and take top 20
        lane_type_metrics = lane_type_metrics.sort_values('CONVERSION_RATE_PCT', ascending=False).head(20)
        
        # Calculate median conversion rate
        median_conversion_rate = df['CONVERSION_RATE_PCT'].median()
        
        # Create figure using go.Figure instead of px.bar for more control
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=lane_type_metrics['LANE_TYPE'],
            y=lane_type_metrics['CONVERSION_RATE_PCT'],
            name='Conversion Rate',
            marker_color='lightblue'
        ))
        
        # Add median line
        fig.add_hline(
            y=median_conversion_rate,
            line_dash="dash",
            line_color="red",
            annotation_text="Median",
            annotation_position="right"
        )
        
        # Update layout
        fig.update_layout(
            title='Top 20 Lane Types by Conversion Rate',
            xaxis_title='Lane Type',
            yaxis_title='Conversion Rate (%)',
            yaxis=dict(tickformat='.2f'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display median value
        st.info(f"Median Conversion Rate: {median_conversion_rate:.2f}%")
    
    with tab3:
        # Add time period selector
        time_period = st.selectbox(
            "Select Time Period",
            options=["Daily", "Weekly", "Monthly"],
            index=0,
            key="lane_type_trend_period"
        )
        
        # Lane type trend comparison
        if len(df['LANE_TYPE'].unique()) > 0:
            # Get top 5 lane types by conversion rate
            lane_type_metrics = df.groupby('LANE_TYPE')['CONVERSION_RATE_PCT'].mean().reset_index()
            lane_type_metrics = lane_type_metrics.sort_values('CONVERSION_RATE_PCT', ascending=False)
            top_lane_types = lane_type_metrics.head(5)['LANE_TYPE'].tolist()
            
            # Calculate conversion rates by lane type and date
            if time_period == "Daily":
                lane_type_trend = df[df['LANE_TYPE'].isin(top_lane_types)].groupby(['BASE_DATE', 'LANE_TYPE'])['CONVERSION_RATE_PCT'].mean().reset_index()
                date_col = 'BASE_DATE'
            elif time_period == "Weekly":
                df['WEEK'] = df['BASE_DATE'].dt.isocalendar().week
                df['YEAR'] = df['BASE_DATE'].dt.isocalendar().year
                lane_type_trend = df[df['LANE_TYPE'].isin(top_lane_types)].groupby(['YEAR', 'WEEK', 'LANE_TYPE'])['CONVERSION_RATE_PCT'].mean().reset_index()
                lane_type_trend['DATE'] = pd.to_datetime(lane_type_trend['YEAR'].astype(str) + '-W' + 
                                                        lane_type_trend['WEEK'].astype(str) + '-1', format='%Y-W%W-%w')
                date_col = 'DATE'
            else:  # Monthly
                df['MONTH'] = df['BASE_DATE'].dt.to_period('M')
                lane_type_trend = df[df['LANE_TYPE'].isin(top_lane_types)].groupby(['MONTH', 'LANE_TYPE'])['CONVERSION_RATE_PCT'].mean().reset_index()
                lane_type_trend['DATE'] = pd.to_datetime(lane_type_trend['MONTH'].astype(str) + '-01')
                date_col = 'DATE'
            
            # Create figure for lane type comparison
            fig = go.Figure()
            
            # Add traces for each lane type
            for lane_type in sorted(top_lane_types):
                lane_data = lane_type_trend[lane_type_trend['LANE_TYPE'] == lane_type]
                fig.add_trace(go.Scatter(
                    x=lane_data[date_col],
                    y=lane_data['CONVERSION_RATE_PCT'],
                    name=lane_type,
                    mode='lines+markers'
                ))
            
            # Update layout
            fig.update_layout(
                title=f'Top 5 Lane Types - {time_period} Conversion Rate Trend',
                xaxis_title='Date',
                yaxis_title='Conversion Rate (%)',
                yaxis=dict(tickformat='.2f'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add lane type statistics
            st.subheader("Lane Type Statistics")
            lane_stats = df[df['LANE_TYPE'].isin(top_lane_types)].groupby('LANE_TYPE').agg({
                'CONVERSION_RATE_PCT': ['mean', 'median', 'std', 'min', 'max']
            }).round(2)
            lane_stats.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
            st.dataframe(lane_stats)
        else:
            st.warning("Please select at least one lane type to compare trends.")
    
    with tab4:
        # Platform distribution
        platform_metrics = df.groupby('DEVICE_PLATFORM').agg({
            'DISTINCT_USER_IMPRESSIONS': 'sum',
            'DISTINCT_USER_CLICKS': 'sum'
        }).reset_index()
        
        # Calculate CTR
        platform_metrics['CTR'] = (platform_metrics['DISTINCT_USER_CLICKS'] / platform_metrics['DISTINCT_USER_IMPRESSIONS'] * 100).round(2)
        
        # Create two columns for the pie charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Impressions pie chart
            fig_impressions = px.pie(platform_metrics, 
                                   values='DISTINCT_USER_IMPRESSIONS', 
                                   names='DEVICE_PLATFORM', 
                                   title='Impressions by Platform')
            st.plotly_chart(fig_impressions, use_container_width=True)
        
        with col2:
            # CTR pie chart
            fig_ctr = px.pie(platform_metrics, 
                           values='CTR', 
                           names='DEVICE_PLATFORM', 
                           title='Click-Through Rate by Platform (%)')
            st.plotly_chart(fig_ctr, use_container_width=True)
    
    with tab5:
        # Raw data view with download option
        st.subheader("Raw Data")
        
        # Display the raw data in a collapsible section
        with st.expander("View Raw Data", expanded=False):
            st.dataframe(df, use_container_width=True)
            
            # Download button for filtered data
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )

def display_notes():
    """Display and manage notes"""
    # Add new note
    with st.expander("Add New Note"):
        note_date = st.date_input("Note Date", value=datetime.now().date())
        note_content = st.text_area("Note Content")
        note_category = st.selectbox("Category", ["General", "Issue", "Observation", "Action Item"])
        
        if st.button("Add Note"):
            if note_content:
                add_note(note_content, note_date, note_category)
                st.success("Note added successfully!")
            else:
                st.error("Please enter note content")
    
    # Display existing notes
    st.subheader("Recent Notes")
    notes_df = get_note_dates()
    if not notes_df.empty:
        for date in notes_df['note_date']:
            with st.expander(f"Notes for {date}"):
                notes = get_notes_by_date(date)
                if not notes.empty:
                    st.dataframe(notes[['timestamp', 'content', 'category']], use_container_width=True)
                else:
                    st.info("No notes for this date")
    else:
        st.info("No notes available")

# Initialize DuckDB connection and create notes table if it doesn't exist
@st.cache_resource
def init_duckdb():
    conn = duckdb.connect('notes.db')
    
    # Create the table with the new schema using DuckDB's syntax
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS notes_id_seq;
        
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY DEFAULT nextval('notes_id_seq'),
            note_date DATE DEFAULT CURRENT_DATE,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'General'
        )
    """)
    return conn

# Function to add a new note
def add_note(content, note_date, category='General'):
    db = init_duckdb()
    db.execute("""
        INSERT INTO notes (content, note_date, category)
        VALUES (?, ?, ?)
    """, [content, note_date, category])

# Function to get notes for a specific date
def get_notes_by_date(date):
    db = init_duckdb()
    return db.execute("""
        SELECT * FROM notes 
        WHERE CAST(note_date AS DATE) = CAST(? AS DATE)
        ORDER BY timestamp DESC
    """, [date]).fetchdf()

# Function to get all unique dates with notes
def get_note_dates():
    db = init_duckdb()
    return db.execute("""
        SELECT DISTINCT CAST(note_date AS DATE) as note_date
        FROM notes 
        ORDER BY note_date DESC
    """).fetchdf()

# Initialize the database connection
init_duckdb()

# Debug information in sidebar
st.sidebar.header("Debug Information")
st.sidebar.write("Python Version:", sys.version)
st.sidebar.write("Snowflake Connector Version:", snowflake.connector.__version__)

# Environment variables status
st.sidebar.header("Environment Variables")
user = os.getenv('SNOWFLAKE_USER')
account = os.getenv('SNOWFLAKE_ACCOUNT')
warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
password = os.getenv('SNOWFLAKE_PASSWORD')
st.sidebar.write(f"SNOWFLAKE_USER: {'✅ Set' if user else '❌ Not Set'}")
st.sidebar.write(f"SNOWFLAKE_ACCOUNT: {'✅ Set' if account else '❌ Not Set'}")
st.sidebar.write(f"SNOWFLAKE_WAREHOUSE: {'✅ Set' if warehouse else '❌ Not Set'}")

# Add these helper functions
def get_date_range():
    """Safely get the date range from the data"""
    if st.session_state.data is not None and 'BASE_DATE' in st.session_state.data.columns:
        min_date = st.session_state.data['BASE_DATE'].min()
        max_date = st.session_state.data['BASE_DATE'].max()
        return min_date, max_date
    return None, None

def initialize_session_state():
    """Initialize all session state variables"""
    if 'conn' not in st.session_state:
        st.session_state.conn = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'metrics_data' not in st.session_state:
        st.session_state.metrics_data = None
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
    if 'device_platform_filter' not in st.session_state:
        st.session_state.device_platform_filter = []
    if 'distribution_tenant_filter' not in st.session_state:
        st.session_state.distribution_tenant_filter = []
    if 'date_preset' not in st.session_state:
        st.session_state.date_preset = None
    if 'current_query' not in st.session_state:
        st.session_state.current_query = None
    # Add precision metrics filter states
    if 'precision_metrics_date_range' not in st.session_state:
        st.session_state.precision_metrics_date_range = None
    if 'precision_metrics_registration' not in st.session_state:
        st.session_state.precision_metrics_registration = []
    if 'precision_metrics_watched' not in st.session_state:
        st.session_state.precision_metrics_watched = []
    if 'precision_metrics_distribution_tenant' not in st.session_state:
        st.session_state.precision_metrics_distribution_tenant = []
    if 'update_precision_metrics_display' not in st.session_state:
        st.session_state.update_precision_metrics_display = False
    if 'just_loaded_metrics' not in st.session_state:
        st.session_state.just_loaded_metrics = False
    # Add time period selection to session state
    if 'trend_time_period' not in st.session_state:
        st.session_state.trend_time_period = "Daily"

def validate_date_range(start_date, end_date, min_date, max_date):
    """Validate and adjust date range to be within bounds"""
    if start_date < min_date:
        start_date = min_date
    if end_date > max_date:
        end_date = max_date
    return start_date, end_date

# Initialize session state at the start
initialize_session_state()

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
            'client_session_keep_alive': True,
            'warehouse': warehouse  # Always include warehouse
        }
        
        # Use username/password authentication
        password = os.getenv('SNOWFLAKE_PASSWORD')
        
        if not password:
            st.error("❌ SNOWFLAKE_PASSWORD environment variable not set.")
            st.error("Please set SNOWFLAKE_PASSWORD for username/password authentication.")
            return None
        
        conn_params['password'] = password
        # Original authenticator logic (externalbrowser/keypair) has been removed 
        # to default to username/password authentication.

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
                playground.dani.standardize_lane_type(list_type, list_name,screen_name) AS lane_type,
                case when user_id not like 'JNAA%' then 'no' else 'yes' end as is_registered,
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
        
        # Store the query in session state for display
        st.session_state.current_query = query
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        
        # Convert base_date to datetime, coercing errors, and ensure it's timezone-naive
        df['BASE_DATE'] = pd.to_datetime(df['BASE_DATE'], errors='coerce')
        if hasattr(df['BASE_DATE'].dt, 'tz') and df['BASE_DATE'].dt.tz is not None:
            df['BASE_DATE'] = df['BASE_DATE'].dt.tz_localize(None)
        
        # Filter out rows where lane_type contains non-alphabetical characters
        original_count = len(df)
        df = df[df['LANE_TYPE'].str.match(r'^[a-zA-Z]+$', na=False)]
        filtered_count = len(df)
        
        if original_count > filtered_count:
            st.info(f"⚠️ Filtered out {original_count - filtered_count} rows where lane_type contained non-alphabetical characters.")
        
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
        # Original authenticator logic (externalbrowser/keypair) has been removed 
        # to default to username/password authentication.

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
                playground.dani.standardize_lane_type(list_type, list_name,screen_name) AS lane_type,
                case when user_id not like 'JNAA%' then 'no' else 'yes' end as is_registered,
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
        
        # Store the query in session state for display
        st.session_state.current_query = query
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        
        # Convert base_date to datetime, coercing errors, and ensure it's timezone-naive
        df['BASE_DATE'] = pd.to_datetime(df['BASE_DATE'], errors='coerce')
        if hasattr(df['BASE_DATE'].dt, 'tz') and df['BASE_DATE'].dt.tz is not None:
            df['BASE_DATE'] = df['BASE_DATE'].dt.tz_localize(None)
        
        # Filter out rows where lane_type contains non-alphabetical characters
        original_count = len(df)
        df = df[df['LANE_TYPE'].str.match(r'^[a-zA-Z]+$', na=False)]
        filtered_count = len(df)
        
        if original_count > filtered_count:
            st.info(f"⚠️ Filtered out {original_count - filtered_count} rows where lane_type contained non-alphabetical characters.")
        
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
    
    # Apply device_platform filter
    if st.session_state.device_platform_filter:
        filtered_df = filtered_df[filtered_df['DEVICE_PLATFORM'].isin(st.session_state.device_platform_filter)]
    
    # Apply distribution_tenant filter
    if st.session_state.distribution_tenant_filter:
        filtered_df = filtered_df[filtered_df['DISTRIBUTION_TENANT'].isin(st.session_state.distribution_tenant_filter)]
    
    st.session_state.filtered_data = filtered_df
    return filtered_df
    if st.session_state.device_platform_filter:
        filtered_df = filtered_df[filtered_df['DEVICE_PLATFORM'].isin(st.session_state.device_platform_filter)]
    
    # Apply distribution_tenant filter
    if st.session_state.distribution_tenant_filter:
        filtered_df = filtered_df[filtered_df['DISTRIBUTION_TENANT'].isin(st.session_state.distribution_tenant_filter)]
    
    st.session_state.filtered_data = filtered_df
    return filtered_df

def apply_date_preset(preset):
    """Apply a date preset to the date range filter"""
    if st.session_state.data is None:
        return
    
    # Use the maximum date from the data as the reference point
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
        start_date = datetime(max_date.year, max_date.month, 1)
        end_date = max_date
    elif preset == "Last month":
        if max_date.month == 1:
            start_date = datetime(max_date.year - 1, 12, 1)
        else:
            start_date = datetime(max_date.year, max_date.month - 1, 1)
        end_date = datetime(max_date.year, max_date.month, 1) - timedelta(days=1)
    elif preset == "This year":
        start_date = datetime(max_date.year, 1, 1)
        end_date = max_date
    elif preset == "Last year":
        start_date = datetime(max_date.year - 1, 1, 1)
        end_date = datetime(max_date.year, 1, 1) - timedelta(days=1)
    elif preset == "All time":
        start_date = st.session_state.data['BASE_DATE'].min()
        end_date = max_date
    else:
        return
    
    # Store the date range in session state
    st.session_state.date_range = (start_date, end_date)
    st.session_state.date_preset = preset
    
    # Apply filters and rerun the app
    apply_filters()
    st.rerun()

# Add this new function for precision metrics with its own date handling
def run_precision_metrics_query():
    """Execute the precision metrics query with optional date filtering"""
    try:
        if not st.session_state.conn:
            st.error("❌ No connection to Snowflake. Please connect first.")
            return None
            
        st.info("🔄 Running precision metrics query...")
        
        cursor = st.session_state.conn.cursor()
        cursor.execute(f"USE WAREHOUSE {warehouse}")
        
        # Base query without date filter
        query = """
        WITH recent_lane_views AS (
            SELECT
                lvf.user_id,
                lvf.base_date,
                lvf.device_platform,
                playground.dani.standardize_lane_type(lvf.list_type, lvf.list_name) AS lane_type,
                CAST(GET_PATH(flattened.value, 'asset_id') AS TEXT) AS asset_id, 
                lvf.distribution_tenant
            FROM joyn_snow.im_main.lane_views_f AS lvf,
                 LATERAL FLATTEN(INPUT => lvf.asset_list) AS flattened
            WHERE lvf.base_date > DATEADD(DAY, -180, CURRENT_DATE)
              AND playground.dani.standardize_lane_type(lvf.list_type, lvf.list_name) IN (
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
                    WHEN r.user_id LIKE 'JNAA%' THEN 'no'
                    ELSE 'yes'
                END AS is_registered,
                round(zeroifnull(distinct_vvs_from_recommendations)/nullifzero(distinct_recommended),2) as pct_watched
            FROM recent_lane_views r
            LEFT JOIN joyn_snow.im_main.video_views_epg_extended v
                ON r.user_id = v.user_id
                AND (v.tvshow_asset_id = r.asset_id OR v.asset_id = r.asset_id)
                AND v.base_date > r.base_date
                AND DATEDIFF(DAY, r.base_date, v.base_date) < 8
                and v.base_date > dateadd(day,-180,current_date) 
                and v.content_type = 'VOD'
                and r.distribution_tenant = v.distribution_tenant
            GROUP BY r.user_id, r.base_date, r.device_platform, r.distribution_tenant
        )
        SELECT
            base_date,
            is_registered,
            watched_any_recommended,
            count(distinct user_id) as total_users,
            median(pct_watched) AS median_recommendation_watch_ratio,
            distribution_tenant
        FROM watched_videos
        group by all
        order by 1 asc
        """
        
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=columns)
        
        # Convert BASE_DATE to datetime, coercing errors, and ensure it's timezone-naive
        df['BASE_DATE'] = pd.to_datetime(df['BASE_DATE'], errors='coerce')
        if hasattr(df['BASE_DATE'].dt, 'tz') and df['BASE_DATE'].dt.tz is not None:
            df['BASE_DATE'] = df['BASE_DATE'].dt.tz_localize(None)
        
        # Store the full dataset in session state
        st.session_state.metrics_data = df
        
        # Set a flag to indicate we just loaded metrics
        st.session_state.just_loaded_metrics = True
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error executing precision metrics query: {str(e)}")
        return None

def apply_precision_metrics_filters(df):
    """Apply filters to the metrics data"""
    if df is not None:
        filtered_df = df.copy()
        
        # Apply date filter
        if st.session_state.precision_metrics_date_range:
            # Handle both tuple and single date cases
            if isinstance(st.session_state.precision_metrics_date_range, tuple):
                start_date, end_date = st.session_state.precision_metrics_date_range
            else:
                start_date = end_date = st.session_state.precision_metrics_date_range
                
            # Convert dates to datetime.date objects for consistent comparison
            if isinstance(start_date, datetime):
                start_date = start_date.date()
            if isinstance(end_date, datetime):
                end_date = end_date.date()
                
            # Convert DataFrame dates to date objects for comparison
            filtered_df = filtered_df[
                (filtered_df['BASE_DATE'].dt.date >= start_date) &
                (filtered_df['BASE_DATE'].dt.date <= end_date)
            ]
        
        # Apply registration filter
        if st.session_state.precision_metrics_registration:
            filtered_df = filtered_df[filtered_df['IS_REGISTERED'].isin(st.session_state.precision_metrics_registration)]
        
        # Apply watched filter - only if it's explicitly set
        if st.session_state.precision_metrics_watched and len(st.session_state.precision_metrics_watched) < 2:
            filtered_df = filtered_df[filtered_df['WATCHED_ANY_RECOMMENDED'].isin(st.session_state.precision_metrics_watched)]
            
        # Apply distribution tenant filter
        if st.session_state.precision_metrics_distribution_tenant:
            filtered_df = filtered_df[filtered_df['DISTRIBUTION_TENANT'].isin(st.session_state.precision_metrics_distribution_tenant)]
        
        return filtered_df
    return None

def display_precision_metrics(metrics_df):
    """Display precision metrics visualization"""
    if metrics_df is not None and not metrics_df.empty:
        # Convert timestamps to datetime objects for proper display
        metrics_df = metrics_df.copy()
        if 'BASE_DATE' in metrics_df.columns:
            metrics_df['BASE_DATE'] = pd.to_datetime(metrics_df['BASE_DATE']).dt.tz_localize(None)

        # Add metrics description section
        with st.expander("📊 Precision Metrics Definitions", expanded=False):
            st.markdown("""
            ### Key Metrics Definitions
            
            #### Median Precision Ratio
            - **Definition**: The median percentage of recommended items that were watched by users
            - **Calculation**: For each user, calculate the ratio of watched recommendations to total recommendations, then take the median across all users
            - **Interpretation**: Higher values indicate better recommendation quality
            - **Range**: 0% to 100%
            
            #### Total Users
            - **Definition**: The total number of unique users who received recommendations
            - **Calculation**: Sum of distinct users across the selected time period
            - **Interpretation**: Indicates the reach of the recommendation system
            - **Note**: Affected by date range and registration status filters
            
            #### Users Who Watched Recommendations
            - **Definition**: The percentage of users who watched at least one recommended item
            - **Calculation**: (Number of users who watched recommendations / Total users) × 100
            - **Interpretation**: Higher values indicate better user engagement
            - **Range**: 0% to 100%
            
            ### Trend Analysis Metrics
            
            #### Monthly Watch Rate
            - **Definition**: The percentage of users who watched recommendations each month
            - **Calculation**: Aggregated monthly data showing user engagement trends
            - **Interpretation**: Helps identify seasonal patterns and long-term trends
            - **Segmentation**: Separated by registration status (registered vs non-registered)
            
            #### Monthly Median Precision Ratio
            - **Definition**: The median precision ratio calculated monthly
            - **Calculation**: Monthly aggregation of the median precision ratio
            - **Interpretation**: Shows how recommendation quality changes over time
            - **Segmentation**: Separated by registration status
            
            ### Additional Context
            
            #### Registration Status Impact
            - **Definition**: Analysis of how user registration affects recommendation performance
            - **Metrics**: 
                - Average Watch Ratio: Mean percentage of recommendations watched
                - Total Users: Number of users in each registration category
            - **Interpretation**: Helps understand if registered users engage differently
            
            #### Data Filters
            - **Date Range**: Filter data by specific time periods
            - **Registration Status**: Filter by user registration status
            - **Watched Status**: Filter by whether users watched recommendations
            - **Distribution Tenant**: Filter by content distribution platform
            
            ### Notes
            - All metrics are calculated using the filtered dataset
            - Percentages are rounded to 2 decimal places
            - User counts are rounded to whole numbers
            - Time-based metrics use the user's local timezone
            """)

        # Create filters section
        with st.expander("Filters", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                # Date range filter
                min_date = metrics_df['BASE_DATE'].min()
                max_date = metrics_df['BASE_DATE'].max()
                
                # Initialize date range if not set
                if st.session_state.precision_metrics_date_range is None:
                    st.session_state.precision_metrics_date_range = (min_date.date(), max_date.date())
                
                # Convert to date objects for the date input widget
                if isinstance(st.session_state.precision_metrics_date_range, tuple):
                    start_date = st.session_state.precision_metrics_date_range[0]
                    end_date = st.session_state.precision_metrics_date_range[1]
                else:
                    start_date = end_date = st.session_state.precision_metrics_date_range
                
                # Ensure we're working with date objects
                if isinstance(start_date, datetime):
                    start_date = start_date.date()
                if isinstance(end_date, datetime):
                    end_date = end_date.date()
                
                selected_dates = st.date_input(
                    "Select Date Range",
                    value=(start_date, end_date),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="precision_metrics_date_picker"
                )
                
                # Update date range in session state
                if len(selected_dates) == 2:
                    st.session_state.precision_metrics_date_range = selected_dates
                
                # Registration status filter
                st.multiselect(
                    "Filter by Registration Status",
                    options=metrics_df['IS_REGISTERED'].unique(),
                    default=st.session_state.precision_metrics_registration,
                    key="precision_metrics_registration_filter",
                    on_change=lambda: setattr(st.session_state, 'precision_metrics_registration', st.session_state.precision_metrics_registration_filter)
                )
                
                # Distribution tenant filter
                st.multiselect(
                    "Filter by Distribution Tenant",
                    options=metrics_df['DISTRIBUTION_TENANT'].unique(),
                    default=st.session_state.precision_metrics_distribution_tenant,
                    key="precision_metrics_distribution_tenant_filter",
                    on_change=lambda: setattr(st.session_state, 'precision_metrics_distribution_tenant', st.session_state.precision_metrics_distribution_tenant_filter)
                )
                
            with col2:
                # Watched status filter
                st.multiselect(
                    "Filter by Watched Status",
                    options=metrics_df['WATCHED_ANY_RECOMMENDED'].unique(),
                    default=st.session_state.precision_metrics_watched,
                    key="precision_metrics_watched_filter",
                    on_change=lambda: setattr(st.session_state, 'precision_metrics_watched', st.session_state.precision_metrics_watched_filter)
                )

        # Apply filters to get the filtered DataFrame
        filtered_df = apply_precision_metrics_filters(metrics_df)
        if filtered_df is None or filtered_df.empty:
            st.warning("No data available after applying filters")
            return

        # Create a container for metrics
        metrics_container = st.container()
        
        # Calculate metrics using filtered data
        median_ratio = filtered_df['MEDIAN_RECOMMENDATION_WATCH_RATIO'].mean()
        total_users = filtered_df['TOTAL_USERS'].sum()
        watched_users = filtered_df[filtered_df['WATCHED_ANY_RECOMMENDED'] == True]['TOTAL_USERS'].sum()
        watched_any = watched_users / total_users if total_users > 0 else 0
        
        # Display metrics in a single row
        with metrics_container:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Median Precision Ratio", f"{median_ratio:.2%}")
            with col2:
                st.metric("Total Users", f"{total_users:,}")
            with col3:
                st.metric("Users Who Watched Recommendations", f"{watched_any:.2%}", f"{watched_users:,} users")
        
        # Add trend graphs using filtered data
        st.markdown("### Monthly Trend: Users Who Watched Recommendations")
        
        # Create a copy of the filtered data for trend graphs
        trend_df = filtered_df.copy()
        
        # Group by month and registration status
        trend_df['MONTH'] = trend_df['BASE_DATE'].dt.to_period('M')
        
        # Create separate dataframes for registered and non-registered users
        registered_df = trend_df[trend_df['IS_REGISTERED'] == 'yes'].copy()
        non_registered_df = trend_df[trend_df['IS_REGISTERED'] == 'no'].copy()
        
        # Function to calculate monthly metrics
        def calculate_monthly_metrics(df):
            if df.empty:
                return pd.DataFrame()
            
            monthly_data = df.groupby('MONTH').agg({
                'TOTAL_USERS': 'sum',
                'WATCHED_ANY_RECOMMENDED': lambda x: sum(df.loc[x.index, 'TOTAL_USERS'] * (df.loc[x.index, 'WATCHED_ANY_RECOMMENDED'] == True))
            }).reset_index()
            
            # Calculate percentage
            monthly_data['PERCENTAGE'] = monthly_data['WATCHED_ANY_RECOMMENDED'] / monthly_data['TOTAL_USERS']
            
            # Convert Period to datetime for plotting
            monthly_data['MONTH_DT'] = monthly_data['MONTH'].astype(str).apply(lambda x: datetime.strptime(x, '%Y-%m'))
            
            return monthly_data
        
        # Calculate metrics for all user types using filtered data
        registered_monthly = calculate_monthly_metrics(registered_df)
        non_registered_monthly = calculate_monthly_metrics(non_registered_df)
        total_monthly = calculate_monthly_metrics(trend_df)
        
        # Create the trend graph with three lines
        fig = go.Figure()
        
        # Add line for total users (all users combined)
        if not total_monthly.empty:
            fig.add_trace(go.Scatter(
                x=total_monthly['MONTH_DT'],
                y=total_monthly['PERCENTAGE'],
                name='All Users',
                mode='lines+markers',
                line=dict(color='green', width=3),
                hovertemplate="<b>Month:</b> %{x|%B %Y}<br>" +
                              "<b>Percentage:</b> %{y:.1%}<br>" +
                              "<b>Total Users:</b> %{customdata[0]:,}<br>" +
                              "<b>Users Who Watched:</b> %{customdata[1]:,}<br>" +
                              "<extra></extra>",
                customdata=total_monthly[['TOTAL_USERS', 'WATCHED_ANY_RECOMMENDED']]
            ))
        
        # Add line for registered users
        if not registered_monthly.empty:
            fig.add_trace(go.Scatter(
                x=registered_monthly['MONTH_DT'],
                y=registered_monthly['PERCENTAGE'],
                name='Registered Users',
                mode='lines+markers',
                line=dict(color='blue'),
                hovertemplate="<b>Month:</b> %{x|%B %Y}<br>" +
                              "<b>Percentage:</b> %{y:.1%}<br>" +
                              "<b>Total Users:</b> %{customdata[0]:,}<br>" +
                              "<b>Users Who Watched:</b> %{customdata[1]:,}<br>" +
                              "<extra></extra>",
                customdata=registered_monthly[['TOTAL_USERS', 'WATCHED_ANY_RECOMMENDED']]
            ))
        
        # Add line for non-registered users
        if not non_registered_monthly.empty:
            fig.add_trace(go.Scatter(
                x=non_registered_monthly['MONTH_DT'],
                y=non_registered_monthly['PERCENTAGE'],
                name='Non-Registered Users',
                mode='lines+markers',
                line=dict(color='red'),
                hovertemplate="<b>Month:</b> %{x|%B %Y}<br>" +
                              "<b>Percentage:</b> %{y:.1%}<br>" +
                              "<b>Total Users:</b> %{customdata[0]:,}<br>" +
                              "<b>Users Who Watched:</b> %{customdata[1]:,}<br>" +
                              "<extra></extra>",
                customdata=non_registered_monthly[['TOTAL_USERS', 'WATCHED_ANY_RECOMMENDED']]
            ))
        
        # Update layout
        fig.update_layout(
            title='Monthly Percentage of Users Who Watched Recommendations',
            xaxis_title='Month',
            yaxis_title='Percentage',
            yaxis=dict(tickformat=".1%"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Display the graph
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a note about the graph
        st.info("This graph shows the percentage of users who watched recommendations over time, separated by registration status.")
        
        # Add a new trend graph for monthly median precision ratio
        st.markdown("### Monthly Trend: Median Precision Ratio")
        
        # Function to calculate monthly median precision ratio using filtered data
        def calculate_monthly_median_precision_ratio(df):
            if df.empty:
                return pd.DataFrame()
            
            monthly_data = df.groupby('MONTH').agg({
                'MEDIAN_RECOMMENDATION_WATCH_RATIO': 'mean'
            }).reset_index()
            
            # Convert Period to datetime for plotting
            monthly_data['MONTH_DT'] = monthly_data['MONTH'].astype(str).apply(lambda x: datetime.strptime(x, '%Y-%m'))
            
            return monthly_data
        
        # Calculate median precision ratio for all user types using filtered data
        total_median = calculate_monthly_median_precision_ratio(trend_df)
        registered_median = calculate_monthly_median_precision_ratio(registered_df)
        non_registered_median = calculate_monthly_median_precision_ratio(non_registered_df)
        
        # Create the trend graph with three lines
        fig_median = go.Figure()
        
        # Add line for total users (all users combined)
        if not total_median.empty:
            fig_median.add_trace(go.Scatter(
                x=total_median['MONTH_DT'],
                y=total_median['MEDIAN_RECOMMENDATION_WATCH_RATIO'],
                name='All Users',
                mode='lines+markers',
                line=dict(color='green', width=3),
                hovertemplate="<b>Month:</b> %{x|%B %Y}<br>" +
                              "<b>Median Precision Ratio:</b> %{y:.2%}<br>" +
                              "<extra></extra>"
            ))
        
        # Add line for registered users
        if not registered_median.empty:
            fig_median.add_trace(go.Scatter(
                x=registered_median['MONTH_DT'],
                y=registered_median['MEDIAN_RECOMMENDATION_WATCH_RATIO'],
                name='Registered Users',
                mode='lines+markers',
                line=dict(color='blue'),
                hovertemplate="<b>Month:</b> %{x|%B %Y}<br>" +
                              "<b>Median Precision Ratio:</b> %{y:.2%}<br>" +
                              "<extra></extra>"
            ))
        
        # Add line for non-registered users
        if not non_registered_median.empty:
            fig_median.add_trace(go.Scatter(
                x=non_registered_median['MONTH_DT'],
                y=non_registered_median['MEDIAN_RECOMMENDATION_WATCH_RATIO'],
                name='Non-Registered Users',
                mode='lines+markers',
                line=dict(color='red'),
                hovertemplate="<b>Month:</b> %{x|%B %Y}<br>" +
                              "<b>Median Precision Ratio:</b> %{y:.2%}<br>" +
                              "<extra></extra>"
            ))
        
        # Update layout
        fig_median.update_layout(
            title='Monthly Median Precision Ratio',
            xaxis_title='Month',
            yaxis_title='Median Precision Ratio',
            yaxis=dict(tickformat=".1%"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Display the graph
        st.plotly_chart(fig_median, use_container_width=True)
        
        # Add a note about the median precision ratio graph
        st.info("This graph shows the median precision ratio over time, separated by registration status.")

        # Detailed analysis tabs
        tab1, tab2 = st.tabs(["Registration Analysis", "Raw Data"])
        
        with tab1:
            # Registration impact analysis using filtered data
            reg_stats = filtered_df.groupby('IS_REGISTERED').agg({
                'MEDIAN_RECOMMENDATION_WATCH_RATIO': 'mean',
                'TOTAL_USERS': 'sum'
            }).round(4)
            
            reg_stats.columns = ['Average Watch Ratio', 'Total Users']
            
            st.subheader("Registration Impact")
            st.dataframe(reg_stats)
            
        with tab2:
            # Raw data view with download option using filtered data
            st.subheader("Raw Data")
            st.dataframe(filtered_df)
            
            # Download button for filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name="precision_metrics_filtered_data.csv",
                mime="text/csv"
            )

# Add async execution function
async def run_queries_async():
    """Run both analytics and precision metrics queries asynchronously"""
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        analytics_future = loop.run_in_executor(executor, run_analytics_query)
        metrics_future = loop.run_in_executor(executor, run_precision_metrics_query)
        
        # Wait for both queries to complete
        analytics_df, metrics_df = await asyncio.gather(analytics_future, metrics_future)
        return analytics_df, metrics_df

# Main app
st.title("❄️ Recommendation Analytics Dashboard")

# Create main layout with columns
main_col, notes_col = st.columns([0.7, 0.3])

with main_col:
    # Add a container for the main content
    main_container = st.container()

    # Add a divider after the title
    st.divider()

    # Connection section in a container
    with st.container():
        st.subheader("🔌 Connection Status")
        if st.session_state.conn:
            st.success("✅ Connected to Snowflake")
        else:
            st.warning("⚠️ Not connected to Snowflake")
            if st.button("Connect to Snowflake"):
                with st.spinner("Connecting to Snowflake..."):
                    test_connection()

    # Add a divider after connection section
    st.divider()

    # Data loading section in a container
    with st.container():
        st.subheader("📊 Data Loading")
        if st.button("Load Analytics Data"):
            with st.spinner("Loading data..."):
                run_analytics_query()

    # Add a divider after data loading section
    st.divider()

    # Lane Analysis section in a container
    with st.container():
        st.subheader("🚦 Lane Analysis")

    # Filters section in a container
    with st.container():
        st.subheader("🔍 Filters")
        if st.session_state.data_loaded:
            # Row 1: Date Preset and Date Range
            col1, col2 = st.columns(2)
            with col1:
                date_presets = [
                    "Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days", 
                    "Last 180 days", "Last 365 days", "This month", "Last month", 
                    "This year", "Last year", "All time"
                ]
                
                selected_preset = st.selectbox(
                    "Date Preset",
                    options=["Custom"] + date_presets,
                    index=0 if st.session_state.date_preset is None else date_presets.index(st.session_state.date_preset) + 1,
                    key="date_preset_select"
                )
                
                if selected_preset != "Custom" and selected_preset != st.session_state.date_preset:
                    apply_date_preset(selected_preset)
            
            with col2:
                min_date, max_date = get_date_range()
                
                # Initialize date range if not set
                if st.session_state.date_range is None:
                    st.session_state.date_range = (min_date, max_date)
                
                # Convert datetime to date objects for the date input widget
                start_date_obj = st.session_state.date_range[0].date() if hasattr(st.session_state.date_range[0], 'date') else st.session_state.date_range[0]
                end_date_obj = st.session_state.date_range[1].date() if hasattr(st.session_state.date_range[1], 'date') else st.session_state.date_range[1]
                
                # Validate the date range
                start_date_obj, end_date_obj = validate_date_range(start_date_obj, end_date_obj, min_date.date(), max_date.date())
                
                # Date range picker
                selected_dates = st.date_input(
                    "Date Range",
                    value=(start_date_obj, end_date_obj),
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
                    
                    # Validate the new date range
                    start_datetime, end_datetime = validate_date_range(start_datetime, end_datetime, min_date, max_date)
                    
                    if (start_datetime, end_datetime) != st.session_state.date_range:
                        st.session_state.date_range = (start_datetime, end_datetime)
                        st.session_state.date_preset = None  # Reset preset when custom range is selected
                        apply_filters()
                        st.rerun()
            
            # Row 2: Registration Status and Device Platform
            col1, col2 = st.columns(2)
            with col1:
                unique_registered = sorted(st.session_state.data['IS_REGISTERED'].unique().tolist())
                selected_registered = st.multiselect(
                    "Registration Status",
                    options=unique_registered,
                    default=st.session_state.is_registered_filter,
                    key="is_registered_multiselect"
                )
                
                if selected_registered != st.session_state.is_registered_filter:
                    st.session_state.is_registered_filter = selected_registered
                    apply_filters()
                    st.rerun()
            
            with col2:
                unique_platforms = sorted(st.session_state.data['DEVICE_PLATFORM'].unique().tolist())
                selected_platforms = st.multiselect(
                    "Device Platform",
                    options=unique_platforms,
                    default=st.session_state.device_platform_filter,
                    key="device_platform_multiselect"
                )
                
                if selected_platforms != st.session_state.device_platform_filter:
                    st.session_state.device_platform_filter = selected_platforms
                    apply_filters()
                    st.rerun()
            
            # Row 3: Lane Type and Distribution Tenant
            col1, col2 = st.columns(2)
            with col1:
                # Get unique lane types and handle None values
                unique_lane_types = st.session_state.data['LANE_TYPE'].unique().tolist()
                # Replace None with 'None' for display purposes
                unique_lane_types = ['None' if x is None else x for x in unique_lane_types]
                # Sort the list, handling the 'None' string appropriately
                unique_lane_types = sorted([x for x in unique_lane_types if x != 'None'] + ['None'])
                
                # Create multiselect for lane_type
                selected_lane_types = st.multiselect(
                    "Lane Type",
                    options=unique_lane_types,
                    default=st.session_state.lane_type_filter,
                    key="lane_type_multiselect"
                )
                
                # Convert 'None' string back to None for filtering
                selected_lane_types = [None if x == 'None' else x for x in selected_lane_types]
                
                if selected_lane_types != st.session_state.lane_type_filter:
                    st.session_state.lane_type_filter = selected_lane_types
                    apply_filters()
                    st.rerun()
            
            with col2:
                # Get unique distribution tenants and handle None values
                unique_tenants = st.session_state.data['DISTRIBUTION_TENANT'].unique().tolist()
                # Replace None with 'None' for display purposes
                unique_tenants = ['None' if x is None else x for x in unique_tenants]
                # Sort the list, handling the 'None' string appropriately
                unique_tenants = sorted([x for x in unique_tenants if x != 'None'] + ['None'])
                
                # Create multiselect for distribution tenant
                selected_tenants = st.multiselect(
                    "Distribution Tenant",
                    options=unique_tenants,
                    default=st.session_state.distribution_tenant_filter,
                    key="distribution_tenant_multiselect"
                )
                
                # Convert 'None' string back to None for filtering
                selected_tenants = [None if x == 'None' else x for x in selected_tenants]
                
                if selected_tenants != st.session_state.distribution_tenant_filter:
                    st.session_state.distribution_tenant_filter = selected_tenants
                    apply_filters()
                    st.rerun()

    # Add a divider after filters section
    st.divider()

    # Main content
    if st.session_state.conn:
        if st.session_state.data_loaded:
            # Display data summary in a container
            with st.container():
                st.subheader("📈 Data Summary")
                display_data_summary(st.session_state.filtered_data)

            # Add a divider after data summary section
            st.divider()

            # Display data visualization in a container
            with st.container():
                st.subheader("📊 Data Visualization")
                display_data_visualization(st.session_state.filtered_data)

            # Add a divider after data visualization section
            st.divider()

            # Display precision metrics in a container
            with st.container():
                st.subheader("📊 Precision Metrics")
                run_precision_metrics_query()
                apply_precision_metrics_filters(st.session_state.metrics_data)
                display_precision_metrics(st.session_state.metrics_data)

            # Add a divider after precision metrics section
            st.divider()

            # Display notes in a container
            with st.container():
                st.subheader("📝 Notes")
                display_notes()

            # Add a divider after notes section
            st.divider()

            # Display notes for a specific date
            with st.container():
                st.subheader("📅 Notes for a Specific Date")
                search_date = st.date_input("Search Date", value=datetime.now().date())
                notes = get_notes_by_date(search_date)
                if not notes.empty:
                    st.dataframe(notes)
                else:
                    st.info(f"No notes found for {search_date}")

# Add this helper function for date conversion
def convert_to_date_range(min_date, max_date, current_range=None):
    """Convert timestamps to date objects and handle None values"""
    if min_date is not None and max_date is not None:
        # Convert Pandas Timestamp to datetime.date
        min_date = min_date.date() if hasattr(min_date, 'date') else min_date
        max_date = max_date.date() if hasattr(max_date, 'date') else max_date
        
        # If there's a current range, convert it too
        if current_range and len(current_range) == 2:
            start_date = current_range[0].date() if hasattr(current_range[0], 'date') else current_range[0]
            end_date = current_range[1].date() if hasattr(current_range[1], 'date') else current_range[1]
            return (start_date, end_date)
        
        return (min_date, max_date)
    return None