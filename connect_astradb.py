from astrapy import DataAPIClient

# Initialize the client
client = DataAPIClient("YOUR_TOKEN")
db = client.get_database_by_api_endpoint(
  "https://adcb34b7-21e2-4a1a-9346-70e8eb48b15d-us-east1.apps.astra.datastax.com"
)

print(f"Connected to Astra DB: {db.list_collection_names()}")