import requests

files = {'file': open('responses_mars_pedigree_25th_april_2018_final.txt','rb')}
values = {'DB': 'photcat', 'OUT': 'csv', 'SHORT': 'short'}
headers = {'auth':'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'}
r = requests.post('http://35.154.1.238:9292/load-responses', headers=headers, files=files, data=values)
print r.text