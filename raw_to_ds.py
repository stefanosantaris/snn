import sys
import random
import time

def process_meta_json(file):
    fi = open(file, "r")
    fo = open(".tmp/item-info", "w")
    for line in fi:
        obj = eval(line)
        cat = obj["categories"][0][-1]
        print(obj["asin"] + "\t" + cat , end="\n", file=fo)

def process_reviews_json(file):
    fi = open(file, "r")
    user_map = {}
    fo = open(".tmp/reviews-info", "w")
    for line in fi:
        obj = eval(line)
        userID = obj["reviewerID"]
        itemID = obj["asin"]
        rating = obj["overall"]
        time = obj["unixReviewTime"]
        print(userID + "\t" + itemID + "\t" + str(rating) + "\t" + str(time), end="\n", file=fo)

def join_reviews_with_meta():
    f_rev = open(".tmp/reviews-info", "r")
    item_list = []
    sample_list = []
    for line in f_rev:
        line = line.strip()
        items = line.split("\t")
        sample_list.append((line,float(items[-1])))
        item_list.append(items[1])
    sample_list = sorted(sample_list, key=lambda x:x[1])
    f_meta = open(".tmp/item-info", "r")
    meta_map = {}
    for line in f_meta:
        arr = line.strip().split("\t")
        if arr[0] not in meta_map:
            meta_map[arr[0]] = arr[1]
            arr = line.strip().split("\t")
    fo = open(".tmp/joined-reviews-with-meta", "w")
    for line in sample_list:
        items = line[0].split("\t")
        asin = items[1]
        j = 0
        while True:
            asin_neg_index = random.randint(0, len(item_list) - 1)
            asin_neg = item_list[asin_neg_index]
            if asin_neg == asin:
                continue
            items[1] = asin_neg
            print("0" + "\t" + "\t".join(items) + "\t" + meta_map[asin_neg], end="\n", file=fo)
            j += 1
            if j == 1:             #negative sampling frequency
                break
        if asin in meta_map:
            print("1" + "\t" + line[0] + "\t" + meta_map[asin], end="\n", file=fo)
        else:
            print("1" + "\t" + line[0] + "\t" + "default_cat", end="\n", file=fo)

def transform_data_to_final_form():
   
    maxlen = 50
    user_maxlen = 50
    # Open reviews with metadata and also create the final sample file
    fin = open(".tmp/joined-reviews-with-meta", "r")
    ffin = open(".tmp/dataset_all_samples.csv", "w")

    user_history_items = {}
    user_history_categories = {}
    item_history_users = {}
    print("click;userID;itemID;category;user_items;len_user_items;user_categories;len_user_categories;item_users;len_item_users", end="\n", file=ffin)

    for line in fin:

        # Break line to separate items
        items = line.strip().split("\t")
        
        # Split each item to a separate variable
        click = int(items[0])
        user = items[1]
        item_id = items[2]
        timestamp = items[4]
        cat = items[5]

        # Collect all the items that the user has selected
        if user in user_history_items:
            user_items = user_history_items[user][-maxlen:]
        else:
            user_items = []

        # Collect all categories that the user has selected
        if user in user_history_categories:
            user_categories = user_history_categories[user][-maxlen:]
        else:
            user_categories = []
        

        # Collect the users that have selected the current item
        if item_id in item_history_users:
            item_users = item_history_users[item_id][-user_maxlen:]
        else:
            item_users = []

        # If item is selected
        if click:
            
            # Store the item and the categories that the user has selected
            if user not in user_history_items:
                user_history_items[user] = []
                user_history_categories[user] = []
            user_history_items[user].append(item_id)
            user_history_categories[user].append(cat)

            # Store the user inside item list
            if item_id not in item_history_users:
                item_history_users[item_id] = []
            item_history_users[item_id].append(user)

        user_history_click_num = len(user_items)
        if(user_history_click_num >= 1):
            print(str(click) + ";" + user + ";" + item_id + ";" + cat + ";"+ "|".join(user_items) + ";" + str(len(user_items)) + ";"+ "|".join(user_categories)+ ";" + str(len(user_categories)) + ";" +"|".join(item_users)+ ";"+  str(len(item_users)), end="\n", file=ffin)
            #print(str(click) + ";" + cat + ";" + str(len(user_items)) + ";" + str(len(user_categories)) + ";" + str(len(item_users)), end="\n", file=ffin)
    

if __name__ == "__main__":
    print("Processing metadata json...")
    process_meta_json(sys.argv[1])
    print("Processing reviews json...")
    process_reviews_json(sys.argv[2])
    print("Joining reviews with metadata...")
    join_reviews_with_meta()
    print("Transform data to final form...")
    transform_data_to_final_form()
