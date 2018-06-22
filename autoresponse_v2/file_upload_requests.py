import requests

files = {'file': open('responses_mars_pedigree_2018-6-14_db_ingestion.txt','rb')}
values = {'DB': 'photcat', 'OUT': 'csv', 'SHORT': 'short'}
headers = {'auth':'A0Zr98j3yXRXHH!jmN]LWXRT'}
r = requests.post('http://35.154.1.238:9292/load-responses', headers=headers, files=files, data=values)
print r.text