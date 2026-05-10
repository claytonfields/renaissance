from .base_dataset import BaseDataset
import io
from PIL import Image
import torch
import numpy as np
import random
import transformers
from torch.utils.data._utils.collate import default_collate


class Refcoco2Dataset(BaseDataset):
    def __init__(self, *args, split="", max_bb = 40, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.max_bb = max_bb

        if split == "train":
            names = ['refcoco_unc_train']
        elif split == "val":
            # names = ["coco_caption_karpathy_val"]
            names = ['refcoco_unc_val']
        elif split == "test":
            names = ['refcoco_unc_test']

        super().__init__(*args, names=names, text_column_name="sentences", remove_duplicate=False, **kwargs)

    # Generalize padding funtion for use in different methods
    def __getitem__(self, index):
        max_bb = self.max_bb
        image_index, ref_index = self.index_mapper[index]
        try:
            label = self.table[image_index]["labels"]
            raw_image = np.array(self.get_raw_image(index))
            image = self.processor(
                            raw_image,
                            return_tensors='pt',
                            size={'height':self.image_size, 'width':self.image_size}
                        )['pixel_values'][0]
            bboxes = self.table[image_index]['bboxes']
        except IndexError:
            print("Hello World")
            print("Index: ", index)
            print("Image Index: ", image_index)
            print("Ref Index: ", ref_index)
            return_dict = {
                
                'image' : [torch.zeros(max_bb,3,self.image_size,self.image_size)],
                'target' : 0,
                'text' : '',
                'text_ids' : torch.zeros(max_bb,self.max_text_len,dtype=torch.int8),
                'text_labels' : torch.zeros(max_bb,self.max_text_len,dtype=torch.int8),
                'text_masks' : torch.zeros(max_bb,self.max_text_len,dtype=torch.int8),
            }
            
            return return_dict
        num_bboxes = len(bboxes)
        num_pad = max_bb - num_bboxes
        target = self.adjust_bbox(bboxes[label])
        
        if num_bboxes > max_bb:
            truth = bboxes.pop(label)
            bboxes = random.choices(bboxes, k=max_bb-1)
            bboxes.append(truth)
            random.shuffle(bboxes)
            label = bboxes.index(truth)
        elif num_bboxes < max_bb:
            pad_boxes = [[0.0, 0.0, 0.0, 0.0] for _ in range(num_pad)]
            bboxes = bboxes + pad_boxes

        resized_bboxes = []
        for bbox in bboxes:
            resized_bboxes.append(self.adjust_bbox(bbox))
        
        text = self.get_text(index)
        text_tokenized = text['text'][1]
        ids = torch.tensor(text_tokenized['input_ids'])
        masks = torch.tensor(text_tokenized['attention_mask'])
        labels = torch.full((self.max_text_len,),-100, dtype=torch.int8)

        return_dict = {
            'image' : [image],#.to(self.device)],
            'bboxes' : torch.tensor(resized_bboxes),#.to(self.device),
            'target' : torch.tensor(target),#.to(self.device),
            'text_ids' : ids,#.unsqueeze(dim=0),#.to(self.device),
            'text_labels' : labels,#.unsqueeze(dim=0),#.to(self.device),
            'text_masks' : masks,#.unsqueeze(dim=0),#.to(self.device)
        }
        
        return return_dict
    
    def collate(self, batch, mlm_collator=None):
        return default_collate(batch)

    def adjust_bbox(self, bbox):
        ret=[0.0,0.0,0.0,0.0]
        ret[1] = bbox[1]*(self.image_size/640)
        ret[3] = bbox[3]*(self.image_size/640)
        ret[0] = bbox[0]*(self.image_size/427)
        ret[2] = bbox[2]*(self.image_size/427)

        ret[3] = ret[1]+ret[3] 
        ret[2] = ret[0]+ret[2]
        return ret



# class Refcoco2Dataset(BaseDataset):
#     def __init__(self, *args, split="", max_bb = 20, **kwargs):
#         assert split in ["train", "val", "test"]
#         self.split = split
#         self.max_bb = max_bb

#         if split == "train":
#             names = ['refcoco_unc_train']
#         elif split == "val":
#             # names = ["coco_caption_karpathy_val"]
#             names = ['refcoco_unc_val']
#         elif split == "test":
#             names = ['refcoco_unc_test']

#         super().__init__(*args, names=names, text_column_name="sentences", remove_duplicate=False, **kwargs)

#     # Generalize padding funtion for use in different methods
#     def __getitem__(self, index):
#         max_bb = self.max_bb
#         image_index, ref_index = self.index_mapper[index]
#         try:
#             label = self.table["labels"][image_index].as_py()
#             raw_image = np.array(self.get_raw_image(index))
#             image = self.processor(
#                             raw_image, 
#                             return_tensors='pt',
#                             size={'height':self.image_size, 'width':self.image_size}
#                         )['pixel_values'][0]
#             bboxes = self.table['bboxes'][image_index].as_py()
#         except IndexError:
#             print("Hello World")
#             print("Index: ", index)
#             print("Image Index: ", image_index)
#             print("Ref Index: ", ref_index)
#             return_dict = {
                
#                 'image' : [torch.zeros(max_bb,3,self.image_size,self.image_size)],
#                 'target' : 0,
#                 'text' : '',
#                 'text_ids' : torch.zeros(max_bb,self.max_text_len,dtype=torch.int8),
#                 'text_labels' : torch.zeros(max_bb,self.max_text_len,dtype=torch.int8),
#                 'text_masks' : torch.zeros(max_bb,self.max_text_len,dtype=torch.int8),
#             }
            
#             return return_dict
#         num_bboxes = len(bboxes)
#         num_pad = max_bb - num_bboxes
        
#         if num_bboxes > max_bb:
#             truth = bboxes.pop(label)
#             bboxes = random.choices(bboxes, k=max_bb-1)
#             bboxes.append(truth)
#             random.shuffle(bboxes)
#             label = bboxes.index(truth)
#         elif num_bboxes < max_bb:
#             pad_boxes = [[0.0, 0.0, 0.0, 0.0] for _ in range(num_pad)]
#             bboxes = bboxes + pad_boxes
            
#         resized_bboxes = []
#         for bbox in bboxes:
#             resized_bboxes.append(self.resize_bbox(bbox))

        
#         text = self.get_text(index)
#         text_tokenized = text['text'][1]
#         ids = torch.tensor(text_tokenized['input_ids'])
#         masks = torch.tensor(text_tokenized['attention_mask'])
#         labels = torch.full((self.max_text_len,),-100, dtype=torch.int8)

#         return_dict = {
#             'image' : [image],#.to(self.device)],
#             'bboxes' : torch.tensor(resized_bboxes),#.to(self.device),
#             'target' : label,#.to(self.device),
#             'text_ids' : ids,#.unsqueeze(dim=0),#.to(self.device),
#             'text_labels' : labels,#.unsqueeze(dim=0),#.to(self.device),
#             'text_masks' : masks,#.unsqueeze(dim=0),#.to(self.device)
#         }
        
#         return return_dict
    
#     def collate(self, batch, mlm_collator=None):
#         return default_collate(batch)
    
#     def resize_bbox(self, bbox):
#         ret=[0.0,0.0,0.0,0.0]
#         ret[1] = bbox[1]*(self.image_size/640)
#         ret[3] = bbox[3]*(self.image_size/640)
#         ret[0] = bbox[0]*(self.image_size/427)
#         ret[2] = bbox[2]*(self.image_size/427)
#         return ret