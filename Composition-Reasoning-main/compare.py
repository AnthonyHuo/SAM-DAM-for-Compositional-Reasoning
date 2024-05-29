import torch

# Load the data from the .pt files
ins_infos = torch.load('/home/mhuo/Composition-Reasoning-main/ins_infos.pt')
ins_infos1 = torch.load('/home/mhuo/Composition-Reasoning-main/ins_infos1.pt')

# Assuming both ins_infos and ins_infos1 contain a list with a single dictionary as shown in your example,
# and you want to compare the first (and only) item in each list.
obj1 = ins_infos[0][0]
obj2 = ins_infos1[0][0]

# Comparing positions (x-axis)
if obj1['position'][0] > obj2['position'][0]:
    position_relation_x = f"{obj1['category']} is on the right side of {obj2['category']}"
else:
    position_relation_x = f"{obj1['category']} is on the left side of {obj2['category']}"

# Comparing positions (y-axis)
if obj1['position'][1] < obj2['position'][1]:
    position_relation_y = f"{obj1['category']} is up to {obj2['category']}"
else:
    position_relation_y = f"{obj1['category']} is under the {obj2['category']}"

# Comparing depth
if obj1['depth'] > obj2['depth']:
    depth_relation = f"{obj1['category']} is nearer to us in this camera view than {obj2['category']}"
else:
    depth_relation = f"{obj1['category']} is farther to us in this camera view than {obj2['category']}"

# Printing the results
print(position_relation_x)
print(position_relation_y)
print(depth_relation)
