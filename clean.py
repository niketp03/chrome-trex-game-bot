import os, random

for x in range(4881):
    y = random.choice(os.listdir("data_cl/wait"))
    os.remove(f"data_cl/wait/{y}")
    print(f"file{x} ({y}) deleted")
