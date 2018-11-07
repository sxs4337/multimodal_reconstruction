import numpy as np
import json 
import pdb
import random, os
from shutil import copyfile
import pickle

train_class_path = '<path to mscoco>/instances_train2014.json'
val_class_path = '<path to mscoco>/instances_val2014.json'

root_src_path = '<path to mscoco>/train2014/'
root_dst_path = './images/'

root_caption_path = '<path to mscoco>/train_captions/'

def get_area(set_ids, value):
	# Calculates the area of the each category in the image
	set_id_areas=[]
	for id in set_ids:
		id_area=0
		for ele in list(value):
			if ele[0]==id: 
				id_area+=ele[1]
			
		set_id_areas.append(id_area)
	return set_id_areas
				
def get_non_match_ids(imageId_to_cat, select_cat_ids):
    # Get all image ids with category ids not corresponding to other classes.
    image_map={}
    for id in select_cat_ids:
        image_map[id] = []
        for key, value in imageId_to_cat.items():
            # area={}
            rest_list = [x for x in select_cat_ids if x!=id]
            set_ids = set([ele[0] for ele in value])
            if id in set_ids and len((set(rest_list) & set_ids))==0 and len(set_ids)<=3:
                # if len(set_ids)==1: 
                image_map[id].append(key)
                # else:
                    # area_ids = get_area(list(set_ids),value)
                    # highest_area_index = area_ids.index(max(area_ids))
                    # if id ==  highest_area_index:
                        # image_map[id].append(key)
                
    return image_map

def get_captions(): 
    caption_json = json.load(open('<path to mscoco>/captions_val2014.json','r'))
    id_to_captions={}
    for annotation in caption_json["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]
        id_to_captions.setdefault(image_id, [])
        id_to_captions[image_id].append(caption)
    print('Done loading Image to Caption dictionary')
    
    return id_to_captions
	
def get_cat_mappings(instances):	
    # Category ID to name mapping
    category_id_to_name = {}
    category_name_to_id = {}
    for ele in instances['categories']:
        category_id_to_name[ele['id']]=ele['name']
        category_name_to_id[ele['name']]=ele['id']

    return category_id_to_name, category_name_to_id
		
def get_imageIds_of_cat(instances):
	imageId_to_cat = {}
	for ele in instances['instances']: 
		if ele['image_id'] not in imageId_to_cat : imageId_to_cat[ele['image_id']] = set()
		imageId_to_cat[ele['image_id']].add((ele['category_id'],ele['area']))
		
	return imageId_to_cat
	
def get_single_cat(imageId_to_cat, select_cat_ids):
	only_one_cat = {}
	for id, value in imageId_to_cat.items():
		if len(value)==1 or len(value)==2:
			if set(select_cat_ids).issubset(list(value)): 
				if list(value)[0] not in only_one_cat.keys():
					only_one_cat[list(value)[0]] = []
				only_one_cat[list(value)[0]].append(id)

	return only_one_cat
	
def get_n_images(only_one_cat, id_to_captions, category_id_to_name, n, rep):
    """
    Get n images from each category
    """
    unique_cat_to_image = {}
    for id, element in only_one_cat.items():
        rand_image_ids = np.random.choice(element,n,replace=rep)
        im_caption_list = []
        for ele in rand_image_ids: 
            im_caption_list.append((ele,'val', id_to_captions[ele]))
        
        unique_cat_to_image[category_id_to_name[id]]=im_caption_list

    return unique_cat_to_image

def main():
    instances = json.load(open(val_class_path,'r'))
   
    category_id_to_name, category_name_to_id = get_cat_mappings(instances)
    
    train_data = pickle.load( open( "train_data_810images_3labels.pickle", "rb" ) ) # 810 images per category
    val_data = pickle.load( open( "val_data_376images_3labels.pickle", "rb" ) ) # 376 images per category
    
    train_val={}
    for key in train_data.keys():
        train_val[key] = train_data[key]+ val_data[key] 
        
    with open('full_data_1186images_3labels.pickle', 'wb') as handle:
        pickle.dump(train_val, handle)
        
    pdb.set_trace()  
    # Select categories
    select_cat = ['bird', 'giraffe',  'train',  \
                   'pizza', 'horse', 'clock', 'toilet', \
                      'airplane', \
                   'sheep', 'elephant', 'dog', 'cat', 'zebra', 'boat','stop sign', 'vase']
    		   
    select_cat_ids=[category_name_to_id[ele] for ele in select_cat]

    # Get all unique categories of all images
    imageId_to_cat = get_imageIds_of_cat(instances)
    
    # Get all samples with single categories
    only_one_cat = get_non_match_ids(imageId_to_cat, select_cat_ids)
    
    # Print the statistics for single categoric images
    stats = [(category_id_to_name[x], len(only_one_cat[x])) for x in only_one_cat.keys()]
    print (stats)
    
    # Get caption data for image ids    
    id_to_captions = get_captions()
 
    # Sample n images from each category with no replacement
    unique_cat_to_image = get_n_images(only_one_cat, id_to_captions, category_id_to_name, 376, rep = False)
    pdb.set_trace()
    # Form the train data file.
    with open('val_data_376images_3labels.pickle', 'wb') as handle:
        pickle.dump(unique_cat_to_image, handle)
	pdb.set_trace()
    # Save images according to category folders
    for cat in select_cat:
        if not os.path.exists(os.path.join(root_dst_path,cat)):
            os.makedirs(os.path.join(root_dst_path,cat))
    
    cat_wise_data = []
    with open('data.txt','w') as file:
        for i in range(142):
            for id in only_one_cat.keys():
                image_id = image_small_map_3[id][i]
                copyfile(os.path.join(root_src_path,'COCO_train2014_'+str(image_id).zfill(12)+'.jpg'), \
                            os.path.join(root_dst_path,category_id_to_name[id],'COCO_train2014_'+str(image_id).zfill(12)+'.jpg'))
                
                categories = [category_id_to_name[ele] for ele in imageId_to_cat[image_id]]
                file.write(str(image_id) + ' ' + str(' '.join(categories)))
                file.write('\n')


    file.close()
    
if __name__=="__main__":
    main()