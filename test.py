import requests

url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

body = {
	"input": """Input: The following paragraph is a consumer complaint.
The complaint is about one of these options: credit cards, credit reporting, mortgages and loans, retail banking, or debt collection. Read the following paragraph and list all the issues.

I bought a GPS from your store and the instructions included are in Spanish, not English. I have to use Google Translate to figure it out. The mounting bracket was broken, and so I need information on how to get a replacement. Moreover, the information seems to be outdated because I cannot see the new roads put in around my house within the last 12 months.

Output: The list of issues is as follows:
1) The instructions are in Spanish, not English.
2) The mounting bracket is broken.
3) The information is outdated.


Input: The following paragraph is a consumer complaint. The complaint is about one of these
options: credit cards, credit reporting, mortgage and loans, retail banking, or debt
collection. Read the following paragraph and list all the issues.

I called your help desk multiple times and every time I waited 10-15 minutes before I
gave up. When I finally got through like after 3 days (yes, 3 days)
your agent kept going over a long checklist of trivial things and asking me to verify, after
I repeatedly told the agent that I am an experienced user and I know what I am doing,
It was a complete waste of time. After like an eternity of this pointless conversation, I
was told that an SME will contact me. That - was 2 days ago. What is the problem with
your support system?

Output:""",
	"parameters": {
		"decoding_method": "greedy",
		"max_new_tokens": 100,
		"repetition_penalty": 1
	},
	"model_id": "google/flan-ul2",
	"project_id": "d301b336-53dd-49ca-945c-ad690fb7964e",
	"moderations": {
		"hap": {
			"input": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			},
			"output": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			}
		},
		"pii": {
			"input": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			},
			"output": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			}
		}
	}
}

headers = {
	"Accept": "application/json",
	"Content-Type": "application/json",
	"Authorization": "Bearer YOUR_ACCESS_TOKEN"
}

response = requests.post(
	url,
	headers=headers,
	json=body
)

if response.status_code != 200:
	raise Exception("Non-200 response: " + str(response.text))

data = response.json()