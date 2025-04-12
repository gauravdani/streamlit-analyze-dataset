# Streamlit Snowflake Analytics Dashboard

A Streamlit application that connects to Snowflake using SSO authentication, provides interactive data filtering capabilities, and includes a notes system for data analysis documentation.

## Features

- SSO authentication with Snowflake
- Interactive data filtering by:
  - Date range (using a date picker)
  - Lane type
  - Registration status
  - Device platform
- Real-time data updates
- Filter summary display
- Raw data viewing
- Notes System:
  - Date-based note organization
  - Categorized notes (General, Analysis, Todo, Question)
  - Persistent storage using DuckDB
  - Real-time note updates
  - Note deletion capability

## Setup

1. Clone this repository
2. Install dependencies using Poetry:
   ```
   poetry install
   ```
3. Create a `.env` file based on `.env.example` with your Snowflake credentials
4. Run the application:
   ```
   poetry run streamlit run app.py
   ```

## Environment Variables

Required:
- `SNOWFLAKE_USER`: Your Snowflake username
- `SNOWFLAKE_ACCOUNT`: Your Snowflake account identifier (format: account.region.cloud)
- `SNOWFLAKE_WAREHOUSE`: Your Snowflake warehouse name

Optional:
- `SNOWFLAKE_ROLE`: Your Snowflake role
- `SNOWFLAKE_DATABASE`: Your Snowflake database
- `SNOWFLAKE_SCHEMA`: Your Snowflake schema

## Application Generation Prompt

Below is a prompt you can use to generate a similar application with AI assistance:

I need to create a Streamlit application that connects to Snowflake using SSO authentication, provides interactive data filtering capabilities, and includes a notes system for data analysis documentation. The application should:

1. Connect to Snowflake using SSO authentication (externalbrowser)
2. Execute a SQL query to retrieve data from Snowflake
3. Provide interactive filtering capabilities for:
   - Date range (using a date picker)
   - Lane type (using multiselect dropdowns)
   - Registration status (using multiselect dropdowns)
   - Device platform (using multiselect dropdowns)
4. Display a summary of the filtered data
5. Allow viewing the raw filtered data in a table
6. Include a notes system that:
   - Organizes notes by date
   - Supports note categorization
   - Persists data using DuckDB
   - Allows note deletion
   - Updates in real-time

The application should handle:
- Environment variable configuration for Snowflake credentials
- Session state management for connection, data, and notes
- Error handling for connection and query execution
- Proper handling of NULL values in filters
- Real-time updates when filters change
- Persistent storage of notes between sessions

Please create a Streamlit application with the following structure:
1. Environment setup and imports
2. Page configuration
3. Session state initialization
4. Database connections (Snowflake and DuckDB)
5. Query execution functions
6. Filter application functions
7. Notes management functions
8. UI components:
   - Sidebar filters
   - Main content area
   - Notes panel
9. Error handling and user feedback
10. Manage dependencies using poetry

The application should follow best practices for:
- Code organization
- Error handling
- User experience
- Performance optimization
- Data persistence
- Real-time updates

Additional requirements:
- Use wide layout for better space utilization
- Implement proper date handling for notes
- Provide clear feedback for all user actions
- Ensure proper database connection management
- Handle concurrent access to the notes database

```