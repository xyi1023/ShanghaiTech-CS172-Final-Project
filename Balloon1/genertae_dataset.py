import json
with open("transforms_train.json","r") as f:
    data = json.load(f)
new_data = {}
new_data["ids"] = []
new_data["count"] = len(data["frames"])
for i in range(len(data["frames"])):
    new_data["ids"].append(data["frames"][i]["file_path"].split("/")[-1])
with open("dataset.json","w") as f:
    json.dump(new_data,f,indent=4)