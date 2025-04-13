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
    if 'update_precision_metrics_display' not in st.session_state:
        st.session_state.update_precision_metrics_display = False
    if 'just_loaded_metrics' not in st.session_state:
        st.session_state.just_loaded_metrics = False

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
            WHERE base_date > dateadd(DAY, -365, CURRENT_DATE)
        ),
        correct_as_f AS (
            SELECT user_id,lane_type,base_date,lane_label,event_type,
                playground.dani.standardize_lane_type(lane_type, lane_label) AS rlane_type
            FROM joyn_snow.im.asset_select_f
            WHERE base_date > dateadd(DAY, -365, CURRENT_DATE)
            union all
            SELECT user_id,lane_type,base_date,lane_label,event_type,
                playground.dani.standardize_lane_type(lane_type, lane_label) AS rlane_type
            FROM joyn_snow.im.video_playback_request_f
            WHERE base_date > dateadd(DAY, -365, CURRENT_DATE)
        )
        SELECT 
            a.base_date,a.lane_type,a.is_registered,a.device_platform,
            HLL(DISTINCT a.user_id) as distinct_user_impressions,
            HLL(DISTINCT CASE WHEN b.user_id IS NOT NULL THEN b.user_id END) AS distinct_user_clicks,
            ROUND((HLL(CASE WHEN b.user_id IS NOT NULL THEN b.user_id END) / NULLIF(COUNT(DISTINCT a.user_id), 0)) * 100, 2) AS conversion_rate_pct 
        FROM correct_lane_views_f a 
        LEFT JOIN correct_as_f b ON 
        (a.user_id = b.user_id  and a.lane_type = b.rlane_type AND datediff(day, a.base_date, b.base_date) < 8 and b.base_date >= a.base_date)
        GROUP BY all 
        order by 1 asc;
        """
        
        # Store the query in session state for display
        st.session_state.current_query = query
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        
        # Convert base_date to datetime
        df['BASE_DATE'] = pd.to_datetime(df['BASE_DATE'])
        
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
                CAST(GET_PATH(flattened.value, 'asset_id') AS TEXT) AS asset_id
            FROM joyn_snow.im.lane_views_f AS lvf,
                 LATERAL FLATTEN(INPUT => lvf.asset_list) AS flattened
            WHERE lvf.base_date > DATEADD(DAY, -120, CURRENT_DATE)
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
                COUNT(DISTINCT r.asset_id) AS distinct_recommended,
                COUNT(DISTINCT CASE 
                    WHEN v.user_id IS NOT NULL THEN COALESCE(v.tvshow_asset_id, v.asset_id)
                END) AS distinct_vvs_from_recommendations,
                IFF(COUNT_IF(v.user_id IS NOT NULL) > 0, TRUE, FALSE) AS watched_any_recommended,
                CASE 
                    WHEN r.user_id LIKE 'JNDE%' THEN 'yes'
                    ELSE 'no'
                END AS is_registered,
                round(zeroifnull(distinct_vvs_from_recommendations)/nullifzero(distinct_recommended),2) as pct_watched
            FROM recent_lane_views r
            LEFT JOIN joyn_snow.im.video_views_epg_extended v
                ON r.user_id = v.user_id
                AND (v.tvshow_asset_id = r.asset_id OR v.asset_id = r.asset_id)
                AND v.base_date > r.base_date
                AND DATEDIFF(DAY, r.base_date, v.base_date) < 8
                and v.base_date > dateadd(day,-120,current_date) 
                and v.content_type = 'VOD'
            GROUP BY r.user_id, r.base_date, r.device_platform
        )
        SELECT
            base_date,
            is_registered,
            watched_any_recommended,
            count(distinct user_id) as total_users,
            median(pct_watched) AS median_recommendation_watch_ratio
        FROM watched_videos
        group by all
        order by 1 asc
        """
        
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=columns)
        
        # Convert BASE_DATE to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['BASE_DATE']):
            df['BASE_DATE'] = pd.to_datetime(df['BASE_DATE'])
        
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
        if st.session_state.precision_metrics_date_range and len(st.session_state.precision_metrics_date_range) == 2:
            start_date, end_date = st.session_state.precision_metrics_date_range
            # Convert to datetime if they're date objects
            if hasattr(start_date, 'date') and not hasattr(start_date, 'time'):
                start_date = datetime.combine(start_date, datetime.min.time())
            if hasattr(end_date, 'date') and not hasattr(end_date, 'time'):
                end_date = datetime.combine(end_date, datetime.max.time())
                
            filtered_df = filtered_df[
                (filtered_df['BASE_DATE'].dt.date >= start_date) &
                (filtered_df['BASE_DATE'].dt.date <= end_date)
            ]
        
        # Apply registration filter
        if st.session_state.precision_metrics_registration:
            filtered_df = filtered_df[filtered_df['IS_REGISTERED'].isin(st.session_state.precision_metrics_registration)]
        
        # Apply watched filter
        if st.session_state.precision_metrics_watched:
            filtered_df = filtered_df[filtered_df['WATCHED_ANY_RECOMMENDED'].isin(st.session_state.precision_metrics_watched)]
        
        return filtered_df
    return None

def display_precision_metrics(df):
    """Display precision metrics visualizations"""
    if df is not None:
        # Create a container for metrics
        metrics_container = st.container()
        
        # Calculate metrics
        median_ratio = df['MEDIAN_RECOMMENDATION_WATCH_RATIO'].mean()
        total_users = df['TOTAL_USERS'].sum()
        
        # Calculate watched users from the original unfiltered data
        # This ensures it's not affected by the watch status filter
        if st.session_state.metrics_data is not None:
            # Apply only date and registration filters to get the total watched users
            unfiltered_df = st.session_state.metrics_data.copy()
            
            # Apply date filter if it exists
            if st.session_state.precision_metrics_date_range and len(st.session_state.precision_metrics_date_range) == 2:
                start_date, end_date = st.session_state.precision_metrics_date_range
                # Convert to datetime if they're date objects
                if hasattr(start_date, 'date') and not hasattr(start_date, 'time'):
                    start_date = datetime.combine(start_date, datetime.min.time())
                if hasattr(end_date, 'date') and not hasattr(end_date, 'time'):
                    end_date = datetime.combine(end_date, datetime.max.time())
                    
                unfiltered_df = unfiltered_df[
                    (unfiltered_df['BASE_DATE'].dt.date >= start_date) &
                    (unfiltered_df['BASE_DATE'].dt.date <= end_date)
                ]
            
            # Apply registration filter if it exists
            if st.session_state.precision_metrics_registration:
                unfiltered_df = unfiltered_df[unfiltered_df['IS_REGISTERED'].isin(st.session_state.precision_metrics_registration)]
            
            # Calculate total users and watched users from the unfiltered data
            unfiltered_total_users = unfiltered_df['TOTAL_USERS'].sum()
            watched_users = unfiltered_df[unfiltered_df['WATCHED_ANY_RECOMMENDED'] == True]['TOTAL_USERS'].sum()
            watched_any = watched_users / unfiltered_total_users
        else:
            # Fallback to using the filtered data if original data is not available
            watched_users = df[df['WATCHED_ANY_RECOMMENDED'] == True]['TOTAL_USERS'].sum()
            watched_any = watched_users / total_users
        
        # Display metrics in a single row
        with metrics_container:
            st.metric("Median Precision Ratio", f"{median_ratio:.2%}")
            st.metric("Total Users", f"{total_users:,}")
            st.metric("Users Who Watched Recommendations", f"{watched_any:.2%}", f"{watched_users:,} users")
        
        # Add a trend graph for monthly percentage of users who watched recommendations
        st.markdown("### Monthly Trend: Users Who Watched Recommendations")
        
        # Create a copy of the unfiltered data for the trend graph
        if st.session_state.metrics_data is not None:
            trend_df = st.session_state.metrics_data.copy()
            
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
            
            # Calculate metrics for all user types
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
            st.info("This graph shows the percentage of users who watched recommendations over time, separated by registration status. The data is not affected by the watch status filter.")
            
            # Add a new trend graph for monthly median precision ratio
            st.markdown("### Monthly Trend: Median Precision Ratio")
            
            # Function to calculate monthly median precision ratio
            def calculate_monthly_median_precision_ratio(df):
                if df.empty:
                    return pd.DataFrame()
                
                monthly_data = df.groupby('MONTH').agg({
                    'MEDIAN_RECOMMENDATION_WATCH_RATIO': 'mean'
                }).reset_index()
                
                # Convert Period to datetime for plotting
                monthly_data['MONTH_DT'] = monthly_data['MONTH'].astype(str).apply(lambda x: datetime.strptime(x, '%Y-%m'))
                
                return monthly_data
            
            # Calculate median precision ratio for all user types
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
            st.info("This graph shows the median precision ratio over time, separated by registration status. The median precision ratio represents the median percentage of recommended items that were watched by users.")
        else:
            st.warning("No data available for trend analysis.")

        # Detailed analysis tabs
        tab1, tab2 = st.tabs(["Registration Analysis", "Raw Data"])
        
        with tab1:
            # Registration impact analysis - only for users who watched
            reg_stats = df.groupby('IS_REGISTERED').agg({
                'MEDIAN_RECOMMENDATION_WATCH_RATIO': 'mean',
                'TOTAL_USERS': 'sum'
            }).round(4)
            
            reg_stats.columns = ['Average Watch Ratio', 'Total Users']
            
            st.subheader("Registration Impact")
            st.dataframe(reg_stats)
            
        with tab2:
            # Raw data view with download option
            st.subheader("Raw Data")
            st.dataframe(df)
            
            # Download button for filtered data
            csv = df.to_csv(index=False)
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

    # Filters section in a container
    with st.container():
        st.subheader("🔍 Filters")
        if st.session_state.data_loaded:
            # Date presets filter
            st.markdown("##### 📅 Date Presets")
            date_presets = [
                "Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days", 
                "Last 180 days", "Last 365 days", "This month", "Last month", 
                "This year", "Last year", "All time"
            ]
            
            selected_preset = st.selectbox(
                "Select Date Preset",
                options=["Custom"] + date_presets,
                index=0 if st.session_state.date_preset is None else date_presets.index(st.session_state.date_preset) + 1,
                key="date_preset_select"
            )
            
            if selected_preset != "Custom" and selected_preset != st.session_state.date_preset:
                apply_date_preset(selected_preset)
            
            # Date range filter
            st.markdown("##### 📅 Custom Date Range")
            min_date, max_date = get_date_range()
            
            # Initialize date range if not set
            if st.session_state.date_range is None:
                st.session_state.date_range = (min_date, max_date)
            
            # Date range picker
            selected_dates = st.date_input(
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
            st.markdown("##### 🛣️ Lane Type")
            # Get unique lane types and handle None values
            unique_lane_types = st.session_state.data['LANE_TYPE'].unique().tolist()
            # Replace None with 'None' for display purposes
            unique_lane_types = ['None' if x is None else x for x in unique_lane_types]
            # Sort the list, handling the 'None' string appropriately
            unique_lane_types = sorted([x for x in unique_lane_types if x != 'None'] + ['None'])
            
            # Create multiselect for lane_type
            selected_lane_types = st.multiselect(
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
            st.markdown("##### 👤 Registration Status")
            unique_registered = sorted(st.session_state.data['IS_REGISTERED'].unique().tolist())
            
            # Create multiselect for is_registered
            selected_registered = st.multiselect(
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
            
            # Device platform filter
            st.markdown("##### 📱 Device Platform")
            unique_platforms = sorted(st.session_state.data['DEVICE_PLATFORM'].unique().tolist())
            
            # Create multiselect for device_platform
            selected_platforms = st.multiselect(
                "Filter by Device Platform",
                options=unique_platforms,
                default=st.session_state.device_platform_filter,
                key="device_platform_multiselect"
            )
            
            # Update device platform filter in session state
            if selected_platforms != st.session_state.device_platform_filter:
                st.session_state.device_platform_filter = selected_platforms
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
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Records", len(st.session_state.data))
                with col2:
                    st.metric("Filtered Records", len(st.session_state.filtered_data))
            
            # Add a divider after data summary
            st.divider()
            
            # Display metric definitions in a container
            with st.container():
                with st.expander("Metric Definitions"):
                    st.markdown("""
                    ### Key Metrics Explained
                    
                    #### Conversion Rate
                    **Definition:** The percentage of unique users who clicked on a recommendation after seeing it.
                    
                    **Calculation:** (Number of unique users who clicked / Number of unique users who saw the recommendation) × 100
                    
                    **Interpretation:** A higher conversion rate indicates that a larger proportion of users who see recommendations are engaging with them by clicking. This metric helps identify which recommendation lanes are most effective at driving user engagement.
                    
                    #### Impressions
                    **Definition:** The number of times recommendations are shown to unique users.
                    
                    **Interpretation:** This metric helps understand the reach of your recommendations. Higher impressions indicate broader visibility, but should be analyzed alongside clicks and conversion rates to assess effectiveness.
                    
                    #### Clicks
                    **Definition:** The number of times unique users interact with recommendations by clicking on them.
                    
                    **Interpretation:** This metric measures user engagement with recommendations. Higher clicks relative to impressions indicate more effective recommendations.
                    """)
            
            # Add a divider after metric definitions
            st.divider()
            
            # Display the underlying query in a container
            with st.container():
                with st.expander("🔍 View Underlying Query"):
                    st.markdown("""
                    ### SQL Query Explanation
                    
                    This query analyzes user interactions with different recommendation lanes:
                    
                    1. **Data Sources**:
                       - `lane_views_f`: Contains impression data (when users see recommendations)
                       - `asset_select_f` and `video_playback_request_f`: Contain click data (when users interact with recommendations)
                    
                    2. **Key Metrics**:
                       - `distinct_user_impressions`: Unique users who saw recommendations (using HyperLogLog for efficient counting)
                       - `distinct_user_clicks`: Unique users who clicked on recommendations
                       - `conversion_rate_pct`: Percentage of users who clicked after seeing recommendations
                    
                    3. **Time Window**:
                       - The query looks at the last 365 days of data
                       - Clicks are matched to impressions within 8 days of the impression date
                    
                    4. **Lane Standardization**:
                       - Uses a custom function `standardize_lane_type` to normalize lane types across different sources
                    """)
                    
                    st.code(st.session_state.current_query, language="sql")
                    
                    st.markdown("""
                    ### How to Modify the Query
                    
                    If you need to modify this query, you can edit the `run_analytics_query()` function in the `app.py` file.
                    After making changes, you'll need to restart the Streamlit app and reload the data.
                    """)
            
            # Add a divider after query section
            st.divider()
            
            # Display filter summary in a container
            with st.container():
                with st.expander("🔍 Filter Summary"):
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
                    
                    if st.session_state.device_platform_filter:
                        st.write(f"**Device Platforms:** {', '.join(st.session_state.device_platform_filter)}")
                    else:
                        st.write("**Device Platforms:** All")
            
            # Add a divider after filter summary
            st.divider()
            
            # Display raw data in a container
            with st.container():
                with st.expander("📋 View Raw Data"):
                    st.markdown("""
                    ### Data Filtering Notes
                    
                    The data has been filtered to remove rows where the lane_type contains non-alphabetical characters.
                    This ensures that only clean, standardized lane types are included in the analysis.
                    
                    If you need to see the complete dataset including non-alphabetical lane types, you can modify the filtering logic in the `run_analytics_query()` function.
                    """)
                    st.dataframe(st.session_state.filtered_data, use_container_width=True)
            
            # Add a divider after raw data section
            st.divider()
            
            # Display trendline graphs in a container
            with st.container():
                st.subheader("📈 Trend Analysis")
                
                # Prepare data for trendlines
                if len(st.session_state.filtered_data) > 0:
                    # Group by date to get daily metrics for filtered data
                    daily_metrics = st.session_state.filtered_data.groupby('BASE_DATE').agg({
                        'DISTINCT_USER_IMPRESSIONS': 'sum',
                        'DISTINCT_USER_CLICKS': 'sum',
                        'CONVERSION_RATE_PCT': 'mean'
                    }).reset_index()
                    
                    # Calculate metrics for non-herolane lanes from the original unfiltered data
                    # This ensures these metrics are not affected by the filters
                    non_herolane_data = st.session_state.data[st.session_state.data['LANE_TYPE'] != 'herolane']
                    non_herolane_daily = non_herolane_data.groupby('BASE_DATE').agg({
                        'DISTINCT_USER_IMPRESSIONS': 'sum',
                        'DISTINCT_USER_CLICKS': 'sum',
                        'CONVERSION_RATE_PCT': 'mean'
                    }).reset_index()
                    
                    # Rename conversion rate column
                    non_herolane_daily = non_herolane_daily.rename(columns={'CONVERSION_RATE_PCT': 'NON_HEROLANE_CONV_RATE'})
                    
                    # Merge with main metrics
                    daily_metrics = daily_metrics.merge(
                        non_herolane_daily[['BASE_DATE', 'NON_HEROLANE_CONV_RATE']], 
                        on='BASE_DATE', 
                        how='left'
                    )
                    
                    # Create tabs for different metrics
                    tab1, tab2, tab3, tab4 = st.tabs(["Impressions & Clicks", "Conversion Rate", "Combined View", "Lane Type Comparison"])
                    
                    # Add year-over-year comparison toggle
                    show_yoy = st.checkbox("Show Year-over-Year Comparison", value=False)
                    
                    # Calculate year-over-year data if enabled
                    if show_yoy:
                        # Create a copy of the data for YoY comparison
                        yoy_metrics = daily_metrics.copy()
                        
                        # Get the current date range
                        current_start_date = daily_metrics['BASE_DATE'].min()
                        current_end_date = daily_metrics['BASE_DATE'].max()
                        
                        # Calculate the previous year's date range
                        prev_year_start = current_start_date - pd.DateOffset(years=1)
                        prev_year_end = current_end_date - pd.DateOffset(years=1)
                        
                        # Filter the data to get only the previous year's data for the same period
                        prev_year_data = st.session_state.data.copy()
                        prev_year_data = prev_year_data[
                            (prev_year_data['BASE_DATE'] >= prev_year_start) & 
                            (prev_year_data['BASE_DATE'] <= prev_year_end)
                        ]
                        
                        # Group by date to get daily metrics for previous year
                        prev_year_daily = prev_year_data.groupby('BASE_DATE').agg({
                            'DISTINCT_USER_IMPRESSIONS': 'sum',
                            'DISTINCT_USER_CLICKS': 'sum',
                            'CONVERSION_RATE_PCT': 'mean'
                        }).reset_index()
                        
                        # Calculate metrics for non-herolane lanes from the previous year data
                        prev_year_non_herolane = prev_year_data[prev_year_data['LANE_TYPE'] != 'herolane']
                        prev_year_non_herolane_daily = prev_year_non_herolane.groupby('BASE_DATE').agg({
                            'CONVERSION_RATE_PCT': 'mean'
                        }).reset_index()
                        
                        # Rename conversion rate column
                        prev_year_non_herolane_daily = prev_year_non_herolane_daily.rename(columns={'CONVERSION_RATE_PCT': 'NON_HEROLANE_CONV_RATE'})
                        
                        # Merge with main metrics
                        prev_year_daily = prev_year_daily.merge(
                            prev_year_non_herolane_daily[['BASE_DATE', 'NON_HEROLANE_CONV_RATE']], 
                            on='BASE_DATE', 
                            how='left'
                        )
                        
                        # Shift dates forward by a year to align with current year for plotting
                        prev_year_daily['BASE_DATE'] = prev_year_daily['BASE_DATE'] + pd.DateOffset(years=1)
                        
                        # Rename columns for clarity
                        prev_year_daily = prev_year_daily.rename(columns={
                            'DISTINCT_USER_IMPRESSIONS': 'YOY_IMPRESSIONS',
                            'DISTINCT_USER_CLICKS': 'YOY_CLICKS',
                            'CONVERSION_RATE_PCT': 'YOY_CONVERSION_RATE',
                            'NON_HEROLANE_CONV_RATE': 'YOY_NON_HEROLANE_CONV_RATE'
                        })
                        
                        # Store for use in plots
                        yoy_metrics = prev_year_daily
                        
                        # Calculate percentage changes for summary
                        current_impressions = daily_metrics['DISTINCT_USER_IMPRESSIONS'].sum()
                        prev_impressions = prev_year_daily['YOY_IMPRESSIONS'].sum()
                        impressions_pct_change = ((current_impressions - prev_impressions) / prev_impressions * 100) if prev_impressions > 0 else 0
                        
                        current_clicks = daily_metrics['DISTINCT_USER_CLICKS'].sum()
                        prev_clicks = prev_year_daily['YOY_CLICKS'].sum()
                        clicks_pct_change = ((current_clicks - prev_clicks) / prev_clicks * 100) if prev_clicks > 0 else 0
                        
                        current_conv_rate = daily_metrics['CONVERSION_RATE_PCT'].mean()
                        prev_conv_rate = prev_year_daily['YOY_CONVERSION_RATE'].mean()
                        conv_rate_pct_change = ((current_conv_rate - prev_conv_rate) / prev_conv_rate * 100) if prev_conv_rate > 0 else 0
                        
                        current_non_herolane_conv = daily_metrics['NON_HEROLANE_CONV_RATE'].mean()
                        prev_non_herolane_conv = prev_year_daily['YOY_NON_HEROLANE_CONV_RATE'].mean()
                        non_herolane_conv_pct_change = ((current_non_herolane_conv - prev_non_herolane_conv) / prev_non_herolane_conv * 100) if prev_non_herolane_conv > 0 else 0
                        
                        # Display YoY summary
                        st.subheader("Year-over-Year Comparison Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Impressions YoY Change", 
                                f"{impressions_pct_change:.1f}%",
                                delta_color="normal" if impressions_pct_change >= 0 else "inverse"
                            )
                        
                        with col2:
                            st.metric(
                                "Clicks YoY Change", 
                                f"{clicks_pct_change:.1f}%",
                                delta_color="normal" if clicks_pct_change >= 0 else "inverse"
                            )
                        
                        with col3:
                            st.metric(
                                "Conversion Rate YoY Change", 
                                f"{conv_rate_pct_change:.1f}%",
                                delta_color="normal" if conv_rate_pct_change >= 0 else "inverse"
                            )
                        
                        with col4:
                            st.metric(
                                "Non-Herolane Conv Rate YoY Change", 
                                f"{non_herolane_conv_pct_change:.1f}%",
                                delta_color="normal" if non_herolane_conv_pct_change >= 0 else "inverse"
                            )
                    
                    with tab1:
                        # Plot impressions and clicks
                        fig_imp_clicks = go.Figure()
                        
                        # Current year data
                        fig_imp_clicks.add_trace(go.Scatter(
                            x=daily_metrics['BASE_DATE'], 
                            y=daily_metrics['DISTINCT_USER_IMPRESSIONS'],
                            name='Current Year Impressions',
                            line=dict(color='blue')
                        ))
                        fig_imp_clicks.add_trace(go.Scatter(
                            x=daily_metrics['BASE_DATE'], 
                            y=daily_metrics['DISTINCT_USER_CLICKS'],
                            name='Current Year Clicks',
                            line=dict(color='red')
                        ))
                        
                        # Add YoY comparison if enabled
                        if show_yoy:
                            fig_imp_clicks.add_trace(go.Scatter(
                                x=yoy_metrics['BASE_DATE'], 
                                y=yoy_metrics['YOY_IMPRESSIONS'],
                                name='Previous Year Impressions',
                                line=dict(color='blue', dash='dash')
                            ))
                            fig_imp_clicks.add_trace(go.Scatter(
                                x=yoy_metrics['BASE_DATE'], 
                                y=yoy_metrics['YOY_CLICKS'],
                                name='Previous Year Clicks',
                                line=dict(color='red', dash='dash')
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
                        fig_conv = go.Figure()
                        
                        # Current year data
                        fig_conv.add_trace(go.Scatter(
                            x=daily_metrics['BASE_DATE'], 
                            y=daily_metrics['CONVERSION_RATE_PCT'],
                            name='Current Year Overall Conversion Rate',
                            line=dict(color='green')
                        ))
                        fig_conv.add_trace(go.Scatter(
                            x=daily_metrics['BASE_DATE'], 
                            y=daily_metrics['NON_HEROLANE_CONV_RATE'],
                            name='Current Year Non-Herolane Median Conversion Rate',
                            line=dict(color='orange', dash='dash')
                        ))
                        
                        # Add YoY comparison if enabled
                        if show_yoy:
                            fig_conv.add_trace(go.Scatter(
                                x=yoy_metrics['BASE_DATE'], 
                                y=yoy_metrics['YOY_CONVERSION_RATE'],
                                name='Previous Year Overall Conversion Rate',
                                line=dict(color='green', dash='dot')
                            ))
                            fig_conv.add_trace(go.Scatter(
                                x=yoy_metrics['BASE_DATE'], 
                                y=yoy_metrics['YOY_NON_HEROLANE_CONV_RATE'],
                                name='Previous Year Non-Herolane Median Conversion Rate',
                                line=dict(color='orange', dash='dot')
                            ))
                        
                        fig_conv.update_layout(
                            title='Daily Conversion Rate Over Time',
                            xaxis_title='Date',
                            yaxis_title='Conversion Rate (%)',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_conv, use_container_width=True)
                    
                    with tab3:
                        # Combined view with secondary y-axis
                        fig_combined = go.Figure()
                        
                        # Current year data
                        fig_combined.add_trace(go.Scatter(
                            x=daily_metrics['BASE_DATE'], 
                            y=daily_metrics['DISTINCT_USER_IMPRESSIONS'],
                            name='Current Year Impressions',
                            line=dict(color='blue')
                        ))
                        fig_combined.add_trace(go.Scatter(
                            x=daily_metrics['BASE_DATE'], 
                            y=daily_metrics['DISTINCT_USER_CLICKS'],
                            name='Current Year Clicks',
                            line=dict(color='red')
                        ))
                        fig_combined.add_trace(go.Scatter(
                            x=daily_metrics['BASE_DATE'], 
                            y=daily_metrics['CONVERSION_RATE_PCT'],
                            name='Current Year Conversion Rate (%)',
                            line=dict(color='green'),
                            yaxis='y2'
                        ))
                        fig_combined.add_trace(go.Scatter(
                            x=daily_metrics['BASE_DATE'], 
                            y=daily_metrics['NON_HEROLANE_CONV_RATE'],
                            name='Current Year Non-Herolane Conv Rate (%)',
                            line=dict(color='orange', dash='dash'),
                            yaxis='y2'
                        ))
                        
                        # Add YoY comparison if enabled
                        if show_yoy:
                            fig_combined.add_trace(go.Scatter(
                                x=yoy_metrics['BASE_DATE'], 
                                y=yoy_metrics['YOY_IMPRESSIONS'],
                                name='Previous Year Impressions',
                                line=dict(color='blue', dash='dot')
                            ))
                            fig_combined.add_trace(go.Scatter(
                                x=yoy_metrics['BASE_DATE'], 
                                y=yoy_metrics['YOY_CLICKS'],
                                name='Previous Year Clicks',
                                line=dict(color='red', dash='dot')
                            ))
                            fig_combined.add_trace(go.Scatter(
                                x=yoy_metrics['BASE_DATE'], 
                                y=yoy_metrics['YOY_CONVERSION_RATE'],
                                name='Previous Year Conversion Rate (%)',
                                line=dict(color='green', dash='dot'),
                                yaxis='y2'
                            ))
                            fig_combined.add_trace(go.Scatter(
                                x=yoy_metrics['BASE_DATE'], 
                                y=yoy_metrics['YOY_NON_HEROLANE_CONV_RATE'],
                                name='Previous Year Non-Herolane Conv Rate (%)',
                                line=dict(color='orange', dash='dot'),
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
                        
                    with tab4:
                        # Calculate daily conversion rates by lane type
                        # Start with the filtered data that has all global filters applied
                        filtered_data_for_comparison = st.session_state.filtered_data.copy()
                        
                        # Calculate daily conversion rates by lane type
                        lane_type_daily = filtered_data_for_comparison.groupby(['BASE_DATE', 'LANE_TYPE']).agg({
                            'CONVERSION_RATE_PCT': 'mean'
                        }).reset_index()
                        
                        # Get unique lane types and sort them
                        unique_lane_types = sorted(lane_type_daily['LANE_TYPE'].unique())
                        
                        # Check if any lane types are selected in the global filter
                        if st.session_state.lane_type_filter and len(st.session_state.lane_type_filter) > 0:
                            # Create a figure for lane type comparison
                            fig_lane_types = go.Figure()
                            
                            # Add a trace for each lane type in the filtered data
                            for lane_type in unique_lane_types:
                                lane_data = lane_type_daily[lane_type_daily['LANE_TYPE'] == lane_type]
                                fig_lane_types.add_trace(go.Scatter(
                                    x=lane_data['BASE_DATE'],
                                    y=lane_data['CONVERSION_RATE_PCT'],
                                    name=lane_type,
                                    mode='lines+markers'
                                ))
                            
                            # Add YoY comparison if enabled
                            if show_yoy:
                                # Calculate previous year's data for lane types
                                prev_year_lane_data = prev_year_data.copy()
                                prev_year_lane_daily = prev_year_lane_data.groupby(['BASE_DATE', 'LANE_TYPE']).agg({
                                    'CONVERSION_RATE_PCT': 'mean'
                                }).reset_index()
                                
                                # Shift dates forward by a year to align with current year for plotting
                                prev_year_lane_daily['BASE_DATE'] = prev_year_lane_daily['BASE_DATE'] + pd.DateOffset(years=1)
                                
                                # Add a trace for each lane type in the previous year data
                                for lane_type in unique_lane_types:
                                    prev_lane_data = prev_year_lane_daily[prev_year_lane_daily['LANE_TYPE'] == lane_type]
                                    if not prev_lane_data.empty:
                                        fig_lane_types.add_trace(go.Scatter(
                                            x=prev_lane_data['BASE_DATE'],
                                            y=prev_lane_data['CONVERSION_RATE_PCT'],
                                            name=f"{lane_type} (Previous Year)",
                                            mode='lines+markers',
                                            line=dict(dash='dash')
                                        ))
                            
                            # Update layout
                            fig_lane_types.update_layout(
                                title='Lane Type Conversion Rates Over Time',
                                xaxis_title='Date',
                                yaxis_title='Conversion Rate (%)',
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                ),
                                height=600
                            )
                            
                            st.plotly_chart(fig_lane_types, use_container_width=True)
                            
                            # Add explanation about the graph
                            st.info("""
                            **Note:** This graph shows the conversion rates for lane types over time. 
                            The data is filtered by all global filters including the lane type filter.
                            To compare different lane types, use the lane type filter in the sidebar.
                            """)
                            
                            # Add statistics for lane types
                            st.subheader("Lane Type Statistics")
                            
                            # Calculate statistics for lane types
                            lane_stats = lane_type_daily.groupby('LANE_TYPE').agg({
                                'CONVERSION_RATE_PCT': ['mean', 'median', 'std', 'min', 'max']
                            }).round(2)
                            
                            # Flatten the multi-index columns
                            lane_stats.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
                            lane_stats = lane_stats.reset_index()
                            
                            # Rename columns for display
                            lane_stats = lane_stats.rename(columns={'LANE_TYPE': 'Lane Type'})
                            
                            # Add YoY comparison to statistics if enabled
                            if show_yoy:
                                # Calculate statistics for previous year's lane types
                                prev_year_lane_stats = prev_year_lane_daily.groupby('LANE_TYPE').agg({
                                    'CONVERSION_RATE_PCT': ['mean', 'median']
                                }).round(2)
                                
                                # Flatten the multi-index columns
                                prev_year_lane_stats.columns = ['Prev Year Mean', 'Prev Year Median']
                                prev_year_lane_stats = prev_year_lane_stats.reset_index()
                                
                                # Rename columns for display
                                prev_year_lane_stats = prev_year_lane_stats.rename(columns={'LANE_TYPE': 'Lane Type'})
                                
                                # Merge with current year statistics
                                lane_stats = lane_stats.merge(prev_year_lane_stats, on='Lane Type', how='left')
                                
                                # Calculate YoY change
                                lane_stats['Mean YoY Change'] = ((lane_stats['Mean'] - lane_stats['Prev Year Mean']) / lane_stats['Prev Year Mean'] * 100).round(1)
                                lane_stats['Median YoY Change'] = ((lane_stats['Median'] - lane_stats['Prev Year Median']) / lane_stats['Prev Year Median'] * 100).round(1)
                                
                                # Reorder columns
                                lane_stats = lane_stats[['Lane Type', 'Mean', 'Median', 'Mean YoY Change', 'Median YoY Change', 'Std Dev', 'Min', 'Max']]
                            
                            # Display the statistics table
                            st.dataframe(
                                lane_stats,
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            # Show a message encouraging users to select lane types
                            st.info("""
                            ## Select Lane Types to Compare
                            
                            Please select at least one lane type in the global lane type filter in the sidebar to see the comparison graph.
                            
                            This will help you focus on specific lane types you're interested in rather than showing all lane types at once.
                            """)
                            
                            # Show a placeholder for the graph
                            st.empty()
                            
                            # Show a placeholder for the statistics
                            st.empty()
                    
                    # Add explanation about non-herolane metrics
                    st.info("""
                    **Note:** The non-herolane metrics (orange dashed lines) are calculated from the complete dataset and are not affected by the filters in the sidebar. 
                    This provides a consistent baseline for comparison regardless of the filters applied.
                    """)
                    
                    # Add a collapsible section for mean and median calculations
                    with st.expander("📊 View Mean and Median Calculations"):
                        st.markdown("""
                        ### How Metrics Are Calculated in Trend Analysis
                        
                        The trend analysis section uses the following calculations:
                        """)
                        
                        # Create a table showing the calculations
                        calc_data = {
                            "Metric": [
                                "Daily Impressions", 
                                "Daily Clicks", 
                                "Daily Conversion Rate", 
                                "Non-Herolane Conversion Rate"
                            ],
                            "Calculation Method": [
                                "Sum of DISTINCT_USER_IMPRESSIONS for each date", 
                                "Sum of DISTINCT_USER_CLICKS for each date", 
                                "Mean of CONVERSION_RATE_PCT for each date", 
                                "Mean of CONVERSION_RATE_PCT for non-herolane lanes for each date"
                            ],
                            "Aggregation Function": [
                                "sum()", 
                                "sum()", 
                                "mean()", 
                                "mean()"
                            ],
                            "Example": [
                                f"{daily_metrics['DISTINCT_USER_IMPRESSIONS'].sum():,.0f} total impressions", 
                                f"{daily_metrics['DISTINCT_USER_CLICKS'].sum():,.0f} total clicks", 
                                f"{daily_metrics['CONVERSION_RATE_PCT'].mean():.2f}% average conversion rate", 
                                f"{daily_metrics['NON_HEROLANE_CONV_RATE'].mean():.2f}% average non-herolane conversion rate"
                            ]
                        }
                        
                        st.dataframe(
                            pd.DataFrame(calc_data),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        st.markdown("""
                        ### Additional Statistics
                        
                        The following statistics are calculated across the entire date range:
                        """)
                        
                        # Calculate additional statistics
                        stats_data = {
                            "Statistic": [
                                "Overall Mean Conversion Rate", 
                                "Overall Median Conversion Rate", 
                                "Non-Herolane Mean Conversion Rate", 
                                "Non-Herolane Median Conversion Rate",
                                "Total Impressions", 
                                "Total Clicks"
                            ],
                            "Value": [
                                f"{daily_metrics['CONVERSION_RATE_PCT'].mean():.2f}%", 
                                f"{daily_metrics['CONVERSION_RATE_PCT'].median():.2f}%", 
                                f"{daily_metrics['NON_HEROLANE_CONV_RATE'].mean():.2f}%", 
                                f"{daily_metrics['NON_HEROLANE_CONV_RATE'].median():.2f}%",
                                f"{daily_metrics['DISTINCT_USER_IMPRESSIONS'].sum():,.0f}", 
                                f"{daily_metrics['DISTINCT_USER_CLICKS'].sum():,.0f}"
                            ]
                        }
                        
                        st.dataframe(
                            pd.DataFrame(stats_data),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        st.markdown("""
                        ### Calculation Details
                        
                        - **Daily Aggregation**: Data is first grouped by date, then aggregated using the specified functions
                        - **Mean vs. Median**: 
                          - Mean is used for daily conversion rates to show the average performance
                          - Median is used in the lane performance analysis to better represent typical performance
                        - **Non-Herolane Calculation**: Calculated from the complete dataset (not affected by filters)
                        """)
                else:
                    st.warning("No data available for trend analysis with current filters.")
            
            # Add a divider after trend analysis
            st.divider()
            
            # Lane Performance Analysis in a container
            with st.container():
                st.subheader("🛣️ Lane Performance Analysis")
                
                if len(st.session_state.filtered_data) > 0:
                    # Calculate median conversion rate by lane type
                    lane_performance = st.session_state.filtered_data.groupby('LANE_TYPE').agg({
                        'CONVERSION_RATE_PCT': 'median',
                        'DISTINCT_USER_IMPRESSIONS': 'sum',
                        'DISTINCT_USER_CLICKS': 'sum'
                    }).reset_index()
                    
                    # Sort by conversion rate for better visualization
                    lane_performance = lane_performance.sort_values('CONVERSION_RATE_PCT', ascending=False)
                    
                    # Plot median conversion rate by lane
                    fig_lane_conv = go.Figure()
                    fig_lane_conv.add_trace(go.Bar(
                        x=lane_performance['LANE_TYPE'],
                        y=lane_performance['CONVERSION_RATE_PCT'],
                        text=lane_performance['CONVERSION_RATE_PCT'].round(2),
                        textposition='auto',
                        marker_color='green'
                    ))
                    
                    # Add a horizontal line for the overall median
                    overall_median = lane_performance['CONVERSION_RATE_PCT'].median()
                    fig_lane_conv.add_shape(
                        type="line",
                        x0=-0.5,
                        y0=overall_median,
                        x1=len(lane_performance) - 0.5,
                        y1=overall_median,
                        line=dict(
                            color="red",
                            width=2,
                            dash="dash",
                        ),
                    )
                    
                    # Add annotation for the median line
                    fig_lane_conv.add_annotation(
                        x=len(lane_performance) - 1,
                        y=overall_median,
                        text=f"Overall Median: {overall_median:.2f}%",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="red", size=12, weight="bold")
                    )
                    
                    # Add a second annotation at the beginning of the line for better visibility
                    fig_lane_conv.add_annotation(
                        x=0,
                        y=overall_median,
                        text=f"Overall Median: {overall_median:.2f}%",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="red", size=12, weight="bold")
                    )
                    
                    # Add a title to the graph that explains the median line
                    fig_lane_conv.update_layout(
                        title='Median Conversion Rate by Lane Type',
                        xaxis_title='Lane Type',
                        yaxis_title='Conversion Rate (%)',
                        showlegend=False,
                        height=500,
                        annotations=[
                            dict(
                                x=0.5,
                                y=1.05,
                                xref="paper",
                                yref="paper",
                                text=f"<b>Red dashed line</b> represents the overall median conversion rate ({overall_median:.2f}%) across all lane types",
                                showarrow=False,
                                font=dict(size=12)
                            )
                        ]
                    )
                    
                    # Rotate x-axis labels for better readability
                    fig_lane_conv.update_xaxes(tickangle=45)
                    
                    st.plotly_chart(fig_lane_conv, use_container_width=True)
                    
                    # Display the data table
                    st.subheader("Lane Performance Data")
                    st.dataframe(
                        lane_performance[['LANE_TYPE', 'CONVERSION_RATE_PCT', 'DISTINCT_USER_IMPRESSIONS', 'DISTINCT_USER_CLICKS']]
                        .rename(columns={
                            'LANE_TYPE': 'Lane Type',
                            'CONVERSION_RATE_PCT': 'Median Conversion Rate (%)',
                            'DISTINCT_USER_IMPRESSIONS': 'Total Impressions',
                            'DISTINCT_USER_CLICKS': 'Total Clicks'
                        }),
                        use_container_width=True
                    )
                else:
                    st.warning("No data available for lane performance analysis with current filters.")

            # Add a divider after Lane Performance Analysis
            st.divider()

            # Precision Metrics section in a container
            with st.container():
                st.subheader("🎯 Precision Metrics")
                
                # Add metrics description expander
                with st.expander("📊 Metrics Explained", expanded=False):
                    st.markdown("""
                    ### Precision Metrics Definitions
                    
                    #### Watch Ratio Metrics
                    
                    **Median Recommendation Watch Ratio**
                    - Definition: The median percentage of recommended items that were watched by users
                    - Calculation: `(Number of items watched / Number of items recommended) × 100`
                    - Interpretation: Higher ratios indicate better recommendation precision
                    
                    **Watched Any Recommended**
                    - Definition: Whether a user watched at least one recommended item
                    - Values: True (watched) / False (not watched)
                    - Use: Measures basic engagement with recommendations
                    
                    #### User Metrics
                    
                    **Total Users**
                    - Definition: Count of unique users who received recommendations
                    - Segmentation: Split by registration status (registered vs. non-registered)
                    - Time frame: Aggregated by day
                    
                    **Distinct Recommended**
                    - Definition: Number of unique items recommended to each user
                    - Use: Measures recommendation diversity
                    
                    **Distinct Views from Recommendations**
                    - Definition: Number of unique recommended items actually watched
                    - Time window: Counts views within 30 days of recommendation
                    - Limitation: Only counts VOD content
                    
                    #### Registration Impact
                    
                    **Registration Status**
                    - Definition: Whether a user is registered ('yes') or not ('no')
                    - Format: Based on user ID pattern (JNDE prefix for registered users)
                    - Use: Analyze recommendation effectiveness by user type
                    
                    ### Notes
                    - All metrics are calculated on a daily basis
                    - Watch events are counted only if they occur after the recommendation
                    - A 7-day window is used for attributing views to recommendations
                    - Only VOD (Video on Demand) content is included in the analysis
                    """)

                # Step 1: Filters section
                st.markdown("### Filters")
                # Create columns for filters - always show these regardless of data state
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                
                with filter_col1:
                    # Date range filter
                    st.markdown("##### 📅 Date Range")
                    date_range = st.date_input(
                        "Select date range",
                        value=st.session_state.precision_metrics_date_range or (datetime.now().date() - timedelta(days=30), datetime.now().date()),
                        max_value=datetime.now().date(),
                        key="precision_metrics_date"
                    )
                    
                    if len(date_range) == 2 and date_range != st.session_state.precision_metrics_date_range:
                        st.session_state.precision_metrics_date_range = date_range
                        st.session_state.update_precision_metrics_display = True
                
                with filter_col2:
                    # Registration status filter
                    st.markdown("##### 👤 Registration Status")
                    # Always show the filter with default options if no data is loaded
                    if st.session_state.metrics_data is not None:
                        unique_registered = sorted(st.session_state.metrics_data['IS_REGISTERED'].unique().tolist())
                        selected_registered = st.multiselect(
                            "Filter by Registration Status",
                            options=unique_registered,
                            default=st.session_state.precision_metrics_registration or unique_registered,
                            key="precision_metrics_registration_select"
                        )
                        
                        if selected_registered != st.session_state.precision_metrics_registration:
                            st.session_state.precision_metrics_registration = selected_registered
                            st.session_state.update_precision_metrics_display = True
                    else:
                        # Default options when no data is loaded
                        selected_registered = st.multiselect(
                            "Filter by Registration Status",
                            options=['yes', 'no'],
                            default=['yes', 'no'],
                            key="precision_metrics_registration_default"
                        )
                        # Update session state with default values
                        st.session_state.precision_metrics_registration = selected_registered
                
                with filter_col3:
                    # Watched any recommended filter
                    st.markdown("##### 👁️ Watch Status")
                    watched_options = [True, False]
                    selected_watched = st.multiselect(
                        "Filter by Watch Status",
                        options=watched_options,
                        default=st.session_state.precision_metrics_watched or watched_options,
                        format_func=lambda x: "Watched" if x else "Not Watched",
                        key="precision_metrics_watched_select"
                    )
                    
                    if selected_watched != st.session_state.precision_metrics_watched:
                        st.session_state.precision_metrics_watched = selected_watched
                        st.session_state.update_precision_metrics_display = True

                # Step 2: Load data button
                if st.button("Load Data for Precision Metrics", key="update_precision"):
                    with st.spinner("Calculating precision metrics..."):
                        metrics_df = run_precision_metrics_query()
                        if metrics_df is not None:
                            # Store the data in session state
                            st.session_state.metrics_data = metrics_df
                            st.session_state.update_precision_metrics_display = True
                            st.session_state.just_loaded_metrics = True
                        else:
                            st.warning("Failed to load precision metrics data.")
                
                # Step 3: Display metrics section
                if st.session_state.metrics_data is not None:
                    # Apply filters to the data
                    filtered_metrics = apply_precision_metrics_filters(st.session_state.metrics_data)
                    
                    if filtered_metrics is not None and not filtered_metrics.empty:
                        # Display metrics in a separate section
                        st.markdown("### Metrics")
                        display_precision_metrics(filtered_metrics)
                    else:
                        st.warning("No data matches the selected filters.")
                elif st.session_state.metrics_data is None:
                    st.info("Click 'Load Data for Precision Metrics' to load the data.")
        else:
            st.info("👈 Click 'Load Analytics Data' in the sidebar to analyze the dataset.")
    else:
        st.info("👈 Please connect to Snowflake using the sidebar button to begin.")

    # Add a divider at the end
    st.divider()

    # Instructions in a container
    with st.container():
        with st.expander("ℹ️ Connection Instructions"):
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

with notes_col:
    st.title("📝 Notes")
    
    # Add new note
    with st.form("new_note"):
        note_content = st.text_area("Add a new note")
        note_date = st.date_input("Date", datetime.now().date())
        note_category = st.selectbox("Category", ["General", "Analysis", "Todo", "Question"])
        submit_button = st.form_submit_button("Add Note")
        
        if submit_button and note_content:
            try:
                add_note(note_content, note_date, note_category)
                st.success("Note added successfully!")
                st.rerun()  # Refresh to show new note
            except Exception as e:
                st.error(f"Error adding note: {str(e)}")

    # Display notes by date
    st.subheader("Notes by Date")
    
    try:
        # Get all dates with notes
        dates_df = get_note_dates()
        
        if not dates_df.empty:
            for _, row in dates_df.iterrows():
                date = row['note_date']
                with st.expander(f"📅 {date.strftime('%Y-%m-%d')}"):
                    notes_df = get_notes_by_date(date)
                    for _, note in notes_df.iterrows():
                        st.markdown(f"**{note['timestamp'].strftime('%H:%M')} - {note['category']}**")
                        st.write(note['content'])
                        if st.button(f"🗑️ Delete", key=f"delete_{note['id']}"):
                            db = init_duckdb()
                            db.execute("DELETE FROM notes WHERE id = ?", [note['id']])
                            st.rerun()
                        st.divider()
        else:
            st.info("No notes yet. Add your first note above!")
    except Exception as e:
        st.error(f"Error loading notes: {str(e)}")
        if st.button("Reset Notes Database"):
            init_duckdb()
            st.rerun()

    # Add search functionality
    st.subheader("🔍 Search Notes")
    search_date = st.date_input("Search notes for date", datetime.now().date(), key="search_date")
    search_results = get_notes_by_date(search_date)
    
    if not search_results.empty:
        st.write(f"Found {len(search_results)} notes for {search_date}")
        for _, note in search_results.iterrows():
            st.markdown(f"**{note['timestamp'].strftime('%H:%M')} - {note['category']}**")
            st.write(note['content'])
            st.divider()
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