from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Print environment variables
print("Environment Variables Status:")
print(f"SNOWFLAKE_USER: {os.getenv('SNOWFLAKE_USER')}")
print(f"SNOWFLAKE_ACCOUNT: {os.getenv('SNOWFLAKE_ACCOUNT')}")

# Check if .env file exists
if os.path.exists('.env'):
    print("\n.env file exists")
    with open('.env', 'r') as f:
        print("\nContents of .env file:")
        for line in f:
            # Mask sensitive information
            if '=' in line:
                key, value = line.strip().split('=', 1)
                print(f"{key}=****")
            else:
                print(line.strip())
else:
    print("\n.env file does not exist") 