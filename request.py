import requests

url = 'http://localhost:5000/results'
r = requests.post(url, json={'cust_idx': 0})

print(r.json())
