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

## Deployment in a Google Cloud Container

This guide provides instructions for deploying the application in a Google Cloud container.

### Authentication Methods

The application supports two authentication methods for Snowflake:

1. **SSO Authentication (Local Development)**
   - Uses browser-based authentication
   - Configured via environment variables in `.env` file
   - Ideal for local development and debugging
   - Requires user interaction for authentication

2. **Service Account Authentication (Containerized Deployment)**
   - Uses key-based authentication
   - Requires a Snowflake service account with appropriate permissions
   - Ideal for automated deployments and containerized environments
   - No user interaction required

### Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and configured
- Docker installed locally
- A Google Cloud project with billing enabled
- Appropriate permissions in your Google Cloud project
- A Snowflake service account with appropriate permissions (for containerized deployment)

### Step 1: Set Up Snowflake Service Account (for Containerized Deployment)

1. Create a service account in Snowflake:
   ```sql
   CREATE USER service_account_user PASSWORD = 'your_secure_password';
   CREATE ROLE service_account_role;
   GRANT ROLE service_account_role TO USER service_account_user;
   GRANT USAGE ON WAREHOUSE your_warehouse TO ROLE service_account_role;
   GRANT USAGE ON DATABASE your_database TO ROLE service_account_role;
   GRANT USAGE ON SCHEMA your_schema TO ROLE service_account_role;
   GRANT SELECT ON ALL TABLES IN SCHEMA your_schema TO ROLE service_account_role;
   ```

2. Generate a key pair for the service account:
   ```bash
   # Generate private key
   openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out rsa_key.p8 -nocrypt
   
   # Generate public key
   openssl rsa -in rsa_key.p8 -pubout -out rsa_key.pub
   ```

3. Register the public key with Snowflake:
   ```sql
   ALTER USER service_account_user SET RSA_PUBLIC_KEY='<contents_of_rsa_key.pub>';
   ```

4. Store the private key securely (you'll need it for deployment)

### Step 2: Create a Container Registry Repository

1. Create a new repository in Google Container Registry:
   ```bash
   gcloud container repositories create streamlit-app \
     --repository-format=docker \
     --location=us-central1 \
     --description="Streamlit Analytics Dashboard"
   ```

### Step 3: Build and Tag the Docker Image

1. Build the Docker image:
   ```bash
   docker build -t gcr.io/YOUR_PROJECT_ID/streamlit-app:latest .
   ```

2. Tag the image for Google Container Registry:
   ```bash
   docker tag streamlit-app gcr.io/YOUR_PROJECT_ID/streamlit-app:latest
   ```

### Step 4: Push the Image to Container Registry

1. Configure Docker to use Google Cloud credentials:
   ```bash
   gcloud auth configure-docker
   ```

2. Push the image to Container Registry:
   ```bash
   docker push gcr.io/YOUR_PROJECT_ID/streamlit-app:latest
   ```

### Step 5: Deploy to Google Cloud Run

#### Option A: Using Environment Variables (Not Recommended for Production)

1. Deploy the container to Cloud Run with environment variables:
   ```bash
   gcloud run deploy streamlit-app \
     --image gcr.io/YOUR_PROJECT_ID/streamlit-app:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 1Gi \
     --cpu 1 \
     --port 8501 \
     --set-env-vars="SNOWFLAKE_USER=service_account_user,SNOWFLAKE_ACCOUNT=your_account.region.cloud,SNOWFLAKE_WAREHOUSE=your_warehouse,SNOWFLAKE_DATABASE=your_database,SNOWFLAKE_SCHEMA=your_schema,SNOWFLAKE_AUTHENTICATOR=keypair,SNOWFLAKE_PRIVATE_KEY=your_private_key"
   ```

#### Option B: Using Secret Manager (Recommended for Production)

1. Create secrets in Secret Manager:
   ```bash
   # Create secrets
   gcloud secrets create snowflake-user --replication-policy="automatic"
   gcloud secrets create snowflake-account --replication-policy="automatic"
   gcloud secrets create snowflake-warehouse --replication-policy="automatic"
   gcloud secrets create snowflake-database --replication-policy="automatic"
   gcloud secrets create snowflake-schema --replication-policy="automatic"
   gcloud secrets create snowflake-private-key --replication-policy="automatic"
   
   # Add secret values
   echo -n "service_account_user" | gcloud secrets versions add snowflake-user --data-file=-
   echo -n "your_account.region.cloud" | gcloud secrets versions add snowflake-account --data-file=-
   echo -n "your_warehouse" | gcloud secrets versions add snowflake-warehouse --data-file=-
   echo -n "your_database" | gcloud secrets versions add snowflake-database --data-file=-
   echo -n "your_schema" | gcloud secrets versions add snowflake-schema --data-file=-
   # Add the private key (make sure to preserve newlines)
   cat rsa_key.p8 | gcloud secrets versions add snowflake-private-key --data-file=-
   ```

2. Deploy with secrets:
   ```bash
   gcloud run deploy streamlit-app \
     --image gcr.io/YOUR_PROJECT_ID/streamlit-app:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 1Gi \
     --cpu 1 \
     --port 8501 \
     --set-env-vars="SNOWFLAKE_AUTHENTICATOR=keypair" \
     --set-secrets="SNOWFLAKE_USER=snowflake-user:latest,SNOWFLAKE_ACCOUNT=snowflake-account:latest,SNOWFLAKE_WAREHOUSE=snowflake-warehouse:latest,SNOWFLAKE_DATABASE=snowflake-database:latest,SNOWFLAKE_SCHEMA=snowflake-schema:latest,SNOWFLAKE_PRIVATE_KEY=snowflake-private-key:latest"
   ```

### Step 6: Access Your Deployed Application

After deployment, Cloud Run will provide a URL where your application is accessible. You can also find this URL in the Google Cloud Console under Cloud Run services.

### Step 7: Set Up Continuous Deployment (Optional)

1. Create a `cloudbuild.yaml` file in your repository:
   ```yaml
   steps:
   - name: 'gcr.io/cloud-builders/docker'
     args: ['build', '-t', 'gcr.io/$PROJECT_ID/streamlit-app:$COMMIT_SHA', '.']
   
   - name: 'gcr.io/cloud-builders/docker'
     args: ['push', 'gcr.io/$PROJECT_ID/streamlit-app:$COMMIT_SHA']
   
   - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
     entrypoint: gcloud
     args:
     - 'run'
     - 'deploy'
     - 'streamlit-app'
     - '--image'
     - 'gcr.io/$PROJECT_ID/streamlit-app:$COMMIT_SHA'
     - '--region'
     - 'us-central1'
     - '--platform'
     - 'managed'
     - '--allow-unauthenticated'
     - '--memory'
     - '1Gi'
     - '--cpu'
     - '1'
     - '--port'
     - '8501'
     - '--set-env-vars'
     - 'SNOWFLAKE_AUTHENTICATOR=keypair'
     - '--set-secrets'
     - 'SNOWFLAKE_USER=snowflake-user:latest,SNOWFLAKE_ACCOUNT=snowflake-account:latest,SNOWFLAKE_WAREHOUSE=snowflake-warehouse:latest,SNOWFLAKE_DATABASE=snowflake-database:latest,SNOWFLAKE_SCHEMA=snowflake-schema:latest,SNOWFLAKE_PRIVATE_KEY=snowflake-private-key:latest'
   
   images:
   - 'gcr.io/$PROJECT_ID/streamlit-app:$COMMIT_SHA'
   ```

2. Set up a Cloud Build trigger in the Google Cloud Console to run this configuration when changes are pushed to your repository.

### Troubleshooting

1. **Container fails to start**: Check the Cloud Run logs for error messages. Common issues include missing environment variables or incorrect port configuration.

2. **Snowflake connection issues**: 
   - For SSO authentication: Verify that your Snowflake credentials are correct and that the Cloud Run service has network access to Snowflake.
   - For service account authentication: Verify that the service account has the correct permissions and that the private key is correctly formatted.

3. **Memory issues**: If your application is using too much memory, increase the memory allocation in the Cloud Run deployment command.

### Viewing Logs

To view logs for your Cloud Run service:

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=streamlit-app" --limit 50
```

### Additional Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Google Container Registry Documentation](https://cloud.google.com/container-registry/docs)
- [Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [Snowflake Service Account Authentication](https://docs.snowflake.com/en/user-guide/key-pair-auth.html)
