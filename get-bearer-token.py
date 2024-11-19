import requests
from requests.structures import CaseInsensitiveDict

# Get access token
apiKey="H4xkroS9c6pqbRIxaNzvKTzhWUOMIa8C4xtNE9Is5yDm"
url = "https://iam.cloud.ibm.com/oidc/token"
headers = CaseInsensitiveDict()
headers["Content-Type"] = "application/x-www-form-urlencoded"
data = "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=" + apiKey
resp = requests.post(url, headers=headers, data=data)
if resp.status_code == 200:
    json_resp = resp.json()
    access_token = json_resp.get('access_token')
    print(access_token)
else:
    print("Failed to retrieve access token. Status code:", resp.status_code)