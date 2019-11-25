
import glob
import json
import os
from striprtf import striprtf
import shutil
 


'''
#sample data:

This is a sample question?
-this is a sample answser

This is another a sample question?
-this is a sample answer

'''

def get_file_lines(path_to_file):
    with open(path_to_file) as fh:
        lines = fh.readlines()
        return lines

def parse_interviews_to_json():
    data_path = "/data/txt_data"

    data_files = glob.glob(os.path.join(data_path, "*.txt"))
    print(data_files)
    # metadata we can ignore for now
    data_files.remove(os.path.join(data_path, "Interviewee_List.rtf.txt"))
    data_files.remove(os.path.join(data_path, "Pre-Patient_Interview_Template.rtf.txt"))
    data_files.remove(os.path.join(data_path, "Interview_-_Jacqueline_Smith.rtf.txt"))
    
    final_data = list()

    print("found {0} files!".format(len(data_files)))
    for index, data_file in enumerate(data_files):
        print("processing: {0} of {1}".format(index +1, len(data_files)))

        processed_data = list()

        file_lines = get_file_lines(data_file)

        question = None
        question_answer_dict = dict()
        question_answer_dict["name"] = None
        question_answer_dict["age"] = None

        # here there be dragons: 
        for index, line in enumerate(file_lines):

            # couldnt find a tool that would parse rtf properly.
            line = line.replace("### invalid font number 0","").replace("\n", "").replace("contains 0 fonts total","")
    
            if not line:
                continue

            if len(line) < 4:
                continue
            
            elif "###" in line:
                continue

            elif "?" in line:
                if question:
                    if question not in question_answer_dict:
                        question_answer_dict[question] = ["na"]    
                    question = line
                    #raise Exception("Anamoly: the key is already populated, something has gone wrong")
                else:
                    question = line
            
            elif "-" in line:
                line = line.replace("-", "")
                if question:
                    if question in question_answer_dict:
                        question_answer_dict[question].append(line.strip())
                        #raise Exception( "Anamoly: this user has already answered this question")
                    else:
                        question_answer_dict[question] = [line.strip()]
                  
 
            elif not question_answer_dict["name"] and not question_answer_dict["age"]:
              
                data = line.split()

                if data:
                    if len(data) >= 2:
                        if data[1].isdigit():
                            question_answer_dict["name"] = data[0]
                            question_answer_dict["age"] = int(data[1])
                        elif data[0] == "Tamra":
                            # special case
                            question_answer_dict["name"] = data[0]
                            question_answer_dict["age"] = -1
                        elif data[0] == "Anexis,":
                            question_answer_dict["name"] = data[0]
                            question_answer_dict["age"] =  int(data[1].strip(","))
                        elif data[0] == "Dr.":
                            question_answer_dict["name"] = "{0} {1}".format(data[0], data[1])
                            question_answer_dict["age"] = -1
                        else:
                            print("""WARNING: 
skipping: {0}
in: {1}\n""".format(line, data_file))
                    else:
                        if question:
                            question_answer_dict[question] = [line]
                           
                        else:
                            print("ignoring data: {0}".format(line) )

            else:

                if question:
                    question_answer_dict[question] = [line]
                    
                else:
                    print("""Warning anomoly detected
skipping data: 
{0}""".format(line))
                    raise Exception("Anamoly: data not being captured")
            
        final_data.append(question_answer_dict)
        
        with open("/results/results.json", "w") as fh:
            json_data = json.dumps(final_data, indent=4, sort_keys=True)
            fh.write(json_data)

    return json_data

def remove_file_spaces():
    data_path = "/data/Interviews-for-data-science/interviews"
    new_path = "/data/renamed_data"
    data_files = glob.glob(os.path.join(data_path, "*.rtf"))     

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    for data_file in data_files:
        shutil.copyfile(data_file,
                        os.path.join(new_path, os.path.basename(data_file).replace(" ","_")))
 

if __name__ == "__main__":
    print("getting started")
 
    data = parse_interviews_to_json()
