
import json
import os
import requests
import time
import traceback
from requests.auth import HTTPBasicAuth

def wait_for_system(system_name, url):
    ready = False

    while not ready:
        try:
            print("Attempting to connect to {} at {}".format(system_name, url))
            r = requests.get(url, auth=HTTPBasicAuth('elastic', 'changeme'))

            if r.ok:
                ready = True
            else:
                time.sleep(5)

        except:
            print("Failed to reach {}, trying again in 5s.".format(system_name))
            time.sleep(5)

def send_to_elastic(index, type, json_data):
    try:
        r = requests.post("{0}/{1}/{2}".format(os.environ["ES_URL"], index, type), 
                          json=json_data, 
                          auth=HTTPBasicAuth('elastic', 'changeme'))
        if not r.ok:
            print("Unable to send data to elastic: {0}\n{1}\n{2}\n{3}".format(r.status_code,
                                                                                        index,
                                                                                        type,
                                                                                        json_data))
    except:
        print(traceback.format_exc())

def process_survey_data():
    path = "/results/results.json"

    with open(path, "r") as fh:
        data = fh.read()
        survey_data = json.loads(data)
 
        index = 0
        for record in survey_data:
            index +=1
            print("processing article {0} of {1}".format(index, len(survey_data)))
   
            send_to_elastic("survey_data2", "record", record)

def dep_process_census_data():
    print("processing census data")
    for key in census_data:
        if os.path.exists(os.path.join(data_store, os.path.basename(key))):
            continue
        download_from_s3(key)
        csv_df = pandas.read_csv("{0}/{1}".format(data_store,os.path.basename(key)))
        json_string = csv_df.to_json(orient="records")
        json_records = json.loads(json_string)
        index = 0
        for record in json_records:
            index +=1
            print("processing article {0} of {1}".format(index, len(json_records)))
            record["source"] = "Census"
            record["title"] = os.path.basename(key)
            record["author"] = "Census"
            record["year"] = "2019-01-01"
            send_to_elastic("census_data", "record", record)

if __name__ == "__main__":
    wait_for_system("elastic", os.environ["ES_URL"])
    process_survey_data()