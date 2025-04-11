# Streamlit Snowflake Analytics Dashboard

A Streamlit application that connects to Snowflake using SSO authentication and provides interactive data filtering capabilities.

## Features

- SSO authentication with Snowflake
- Interactive data filtering by date range, lane type, and registration status
- Real-time data updates
- Filter summary display
- Raw data viewing

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

```
I need to create a Streamlit application that connects to Snowflake using SSO authentication and provides interactive data filtering capabilities. The application should:

1. Connect to Snowflake using SSO authentication (externalbrowser)
2. Execute a SQL query to retrieve data from Snowflake
3. Provide interactive filtering capabilities for:
   - Date range (using a date picker)
   - Categorical columns (using multiselect dropdowns)
   - Boolean columns (using multiselect dropdowns)
4. Display a summary of the filtered data
5. Allow viewing the raw filtered data in a table

The application should handle:
- Environment variable configuration for Snowflake credentials
- Session state management for connection and data
- Error handling for connection and query execution
- Proper handling of NULL values in filters
- Real-time updates when filters change

Please create a Streamlit application with the following structure:
1. Environment setup and imports
2. Page configuration
3. Session state initialization
4. Connection functions
5. Query execution functions
6. Filter application functions
7. UI components (sidebar filters, main content)
8. Error handling and user feedback
9. Manage dependencies using poetry.

The application should follow best practices for:
- Code organization
- Error handling
- User experience
- Performance optimization

Create the files as instructed.
```

## Troubleshooting

- Make sure your account identifier is in the format: `account.region.cloud`
- Do not include `https://` or `.snowflakecomputing.com` in the account
- Ensure you have SSO access configured in Snowflake
- Check that your network allows browser-based authentication
- Verify that your warehouse is correctly specified in the `.env` file 