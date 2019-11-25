# ELKSurveyAnalysis
Processing short answer surveys can be difficult. This tools attempts to use the ELK stack to conduct the survey analysis.


# ETL:
Specifically setup to process "Interviews-for-data-science" dataset

```

## first have to convert the .rft files to .txt
```
docker-compose.convert_rtf.yaml up
```

## parse raw interviews
```
docker-compose.docker.parse_interviews.yaml up
```

# Elastic stack

To speed up development we're using a community of the elk stack located here: https://github.com/deviantony/docker-elk
This has been added submodule to the pivot repository. To you must have a github account to use this repo.

## initialize the submodule
```sh
cd ELKSurveyAnalysis
git submodule init
git submodule update

```

## Set the license:
taken from: https://www.elastic.co/guide/en/elasticsearch/reference/current/license-settings.html

edit: ELKSurveyAnalysis/docker-elk/elasticsearch/config/elasticsearch.yaml
```sh
# update to basic
xpack.license.self_generated.type: basic
```

##

start the containers:
```sh
docker-compose -f docker-compose.elastic.yaml up 
```