import os
from PIL import Image
import json
import numpy as np
import tensorflow as tf
from . import util 
from aws_utils.s3_dual_paths import DualPath
from aws_utils.s3_utils import download_prefix_multiprocess 
# from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
# tf.executing_eagerly()
# tf.compat.v1.disable_eager_execution()
import glob 
import json
import math
from pycocotools.coco import COCO
VALID_IMAGE_FORMATS = ['.jpg', '.png', '.tif', '.bmp']

# TODO:use ragged tensor for bboxes, cls, labels ?


VALID_IMAGE_FORMATS = ['.jpg', '.png', '.tif', '.bmp']


class DataLoader():

    """
    DataLoader class containing list of images, labeled masks, bboxes and classes and tf.data.Dataset
    folder: base folder containing images, masks and annotations in images/ labels/ and metadata/ subfolders
    cls_ind: dictionnary mapping classes names and indices
    mode: either 'tf' or 'py: if tf create a tf.Dataset generator else create a pure python generator
    """

    def __init__(self, folder, cls_ind,dataset_type="train",data="aws"):
        self.img_w =1024
        self.img_h=1024
        if data.lower() !="aws":
            self.data_dir = os.path.join(folder, "annotations")
            self.images_dir = os.path.join(folder, "images")
            self.masks_dir = os.path.join(folder, "labels")
            
            # images ids and ids to img name dict
            self.imgs_ids = {os.path.splitext(f)[0]: i for i, f in enumerate(os.listdir(self.data_dir))
                            if os.path.isfile(os.path.join(self.data_dir, f))
                            and os.path.splitext(f)[-1].lower() == ".json" and not f.startswith('.')}

            self.ids_to_imgs = {v: k for k, v in self.imgs_ids.items()}

            self.data_dict = {k: os.path.join(
                self.data_dir, k + ".json") for k in self.imgs_ids}

            # List of imgs and masks (because we do not know the extensions)
            self.image_dict = {
                os.path.splitext(f)[0]: os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir)
                if os.path.isfile(os.path.join(self.images_dir, f)) and os.path.splitext(f)[-1].lower() in
                VALID_IMAGE_FORMATS}

            self.mask_dict = {os.path.splitext(f)[0]: os.path.join(self.masks_dir, f) for f in os.listdir(self.masks_dir)
                            if os.path.isfile(os.path.join(self.masks_dir, f))
                            and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS}

            # reorder keys
            self.image_dict = {k: self.image_dict[k]
                            for k in self.data_dict.keys()}
            self.mask_dict = {k: self.mask_dict[k]
                            for k in self.data_dict.keys()}
            # dict: {'CLASSNAME':idx,...}
            self.cls_ind = cls_ind
            self.ncls = np.unique(list(self.cls_ind.values())).size
            self.length = len(self.image_dict)
        else:
            self.data_dir = os.path.join(folder, "labels")
            self.images_dir = os.path.join(folder, dataset_type)
            self.coco = COCO(os.path.join(self.data_dir,dataset_type+".json"))
            self.get_ids =  self.coco.getImgIds()
            self.load_Imgs_details = self.coco.loadImgs(self.get_ids)
            self.get_anns_ids = self.coco.getAnnIds()
            self.load_Anns_details = self.coco.loadAnns(self.get_anns_ids)    

            # self.data_dir = os.path.join(folder, "annotations")
            # self.images_dir = os.path.join(folder, "images")
            # self.masks_dir = os.path.join(folder, "labels")
            
            # images ids and ids to img name dict
            # self.imgs_ids = {os.path.splitext(f)[0]: i for i, f in enumerate(os.listdir(self.data_dir))
            #                 if os.path.isfile(os.path.join(self.data_dir, f))
            #                 and os.path.splitext(f)[-1].lower() == ".json" and not f.startswith('.')}
            # self.img_ids = self.data_dict.getImgIds()
            # print("selg.loadP_img_details",self.load_Imgs_details)
            self.image_dict = {f['id']:os.path.join(self.images_dir,f['file_name']) for f in self.load_Imgs_details }
            # print(self.image_dict)
            temp_ann_dict ={}
            temp_mask_dict={}
            temp_img_dict={}
            for f in self.load_Anns_details:
                # print("f[]",f['image_id'])
                # print(temp_ann_dict[[f['id']]])
                
                try:
                    temp_ann_dict[f['image_id']].update({f['id']:{"class":f["category_id"],"bbox":f['bbox']}})
                    temp_mask_dict[f['image_id']].update({f['id']:f})
                except KeyError:
                    temp_ann_dict.update({f['image_id']:{f['id']:{"class":f["category_id"],"bbox":f['bbox']}}})
                    temp_mask_dict.update({f['image_id']:{f['id']:f}})
                temp_img_dict[f['image_id']] = self.image_dict[f['image_id']]
                # if f["image_id"]==1:
                    # print(1)
                    # print(temp_ann_dict[1])
            # print("check")
            # print(temp_ann_dict[1])
            self.image_dict = temp_img_dict.copy()
            # print(temp_img_dict)
            # for
            # self.data
            self.data_dict={}
            self.mask_dict={}
            for key in temp_ann_dict:
                ann_json_object = json.dumps(temp_ann_dict[key])

                ann_path = os.path.join(folder,"annotations",dataset_type,str(key)+".json")
                mask_path = os.path.join(folder,"mask",dataset_type,str(key)+".json")
                self.data_dict[key]=ann_path
                self.mask_dict[key]=mask_path
                # print()
                os.makedirs(os.path.dirname(ann_path), exist_ok=True)
                with open(ann_path, "w",encoding='utf8') as outfile:
                    outfile.write(ann_json_object)
                mask_json_object = json.dumps(temp_mask_dict[key])

                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                with open(mask_path, "w",encoding='utf8') as outfile:
                    outfile.write(mask_json_object)
            # self.ann_dict ={f['image_id']:{f['id']:{"class":f["category_id"],"bbox":f['bbox']}} for f in self.load_Anns_details}
            # self.mask_dict = {f['image_id']:{f['id']:{"class":f["category_id"],"bbox":f['bbox']}} for f in self.load_Anns_details}
            
            temp_ann_dict.clear()
            temp_mask_dict.clear()
            temp_img_dict.clear()
            # self.data_dict = [f]
            
            # self.ids_to_imgs = {v: k for k, v in self.imgs_ids.items()}
            # self.data_dict.loadAnns()
            # self.data_dict.ge
            # self.data_dict = {k: os.path.join(
            #     self.data_dir, k + ".json") for k in self.imgs_ids}

            # List of imgs and masks (because we do not know the extensions)
            # self.image_dict = {
            #     os.path.splitext(f)[0]: os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir)
            #     if os.path.isfile(os.path.join(self.images_dir, f)) and os.path.splitext(f)[-1].lower() in
            #     VALID_IMAGE_FORMATS}

            # self.mask_dict = {os.path.splitext(f)[0]: os.path.join(self.masks_dir, f) for f in os.listdir(self.masks_dir)
            #                 if os.path.isfile(os.path.join(self.masks_dir, f))
            #                 and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS}

            # # reorder keys
            # self.image_dict = {k: self.image_dict[k]
            #                 for k in self.data_dict.keys()}
            # self.mask_dict = {k: self.mask_dict[k]
            #                 for k in self.data_dict.keys()}
            # dict: {'CLASSNAME':idx,...}
            self.cls_ind = cls_ind
            self.ncls = np.unique(list(self.cls_ind.values())).size
            self.length = len(self.image_dict)
            self.masks_dir = os.path.join(folder, "labels")

        self.build()

    def build(self):
        """Build the tf.data.Dataset
        each element contains:
        basename, image, mask, bboxes, classes, labels
        """
        # print(len(list(self.image_dict.values())))
        # print(len(list(self.mask_dict.values())))
        self.dataset = tf.data.Dataset.from_tensor_slices((list(self.image_dict.values()),
                                                           list(
            self.mask_dict.values()),
            list(self.data_dict.values())))
        # self.parse_dataset(self.image_dict[0],self.mask_dict[0],self.data_dict[0])

        self.dataset = self.dataset.map(
            lambda x, y, z: tf.py_function(
                func=self.parse_dataset, inp=[x, y, z],
                Tout=[
                    tf.TensorSpec(shape=(None,), dtype=tf.string),
                    tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                    tf.RaggedTensorSpec(shape=(None, None, 4), ragged_rank=1, dtype=tf.float32),
                    tf.RaggedTensorSpec(shape=(None, None,), dtype=tf.int32),
                    tf.RaggedTensorSpec(shape=(None, None,), dtype=tf.int32),
                ]),
            num_parallel_calls=tf.data.AUTOTUNE)
        # for x,y,z,_,_,_ in self.dataset:
            # print("xxxxxxxxxxx",x,y,z)
            # break
        # # Squeeze the unecessary dimension for boxes, class ids and box labels
       
        self.dataset = self.dataset.map(lambda a, b, c, d, e, f: (
            a, b, c, tf.squeeze(d, 0), tf.squeeze(e, 0), tf.squeeze(f, 0)),
            num_parallel_calls=tf.data.AUTOTUNE)
        # for x,y,z,a,b,c in self.dataset:
        #     print("xxxxxxxxxxx",x,y,z)
        #     print("yyyy",a,b,c)
        #     break
        # for i,x,z in self.dataset:
        #     print(i)
        #     break
        # self.dataset.save('test')
        # print(self.dataset.as_numpy())
    def create_mask(self,mask_json,img_h,img_w):
        # ann_id=self.coco.getAnnIds(img_id)
        # anns=self.coco.loadAnns(ann_id)
        # print(mask_json)
        # tf.print(img_h)
        mask =tf.zeros((img_h,img_w),dtype=tf.int32)
        # print("img_h",img_h)
        for m in mask_json:
            # self.
            mask = tf.math.maximum(self.coco.annToMask(mask_json[m])*mask_json[m]['id'],mask)
            # self.coco.ann
        # print(np.unique(mask.numpy()))
        # if tf.unique_with_counts(mask)!=1:
        #     print(tf.unique_with_counts(mask))
        return mask
        # for i in range(len(anns)):
        #     mask = tf.math.maximum(self.coco.annToMask(anns[i]),mask)
        #     self.coco.ann
        # return mask


    def parse_dataset(self, imgfn, maskfn, datafn):
        """For an unkown reason, it is not possible to batch tensors with different length using
        tf.data.experimental.dense_to_ragged_batch
        It is necessary to return RaggedTensor for boxes, class indices and labels, but
        to do so, we must add a leading dimension
        if not, tensorflow throw an error "The rank of a RaggedTensor must be greater than 1"
        the extra dim must be deleted later...
        """

        # Image is already normalized when using tf.io
        # print(imgfn)
        image = tf.io.decode_image(tf.io.read_file(imgfn), channels=3, dtype=tf.float32)
        img_s = tf.shape(image).numpy()
        # tf.print(img_s)
        # print("maskfn",maskfn)
        with open(maskfn.numpy(),'r',encoding ='utf-8') as json_file:
            maskfn = json.load(json_file)
        mask = self.create_mask(maskfn,img_s[0],img_s[1])
        # mask = np.array(Image.open(maskfn.numpy())).astype(np.int32)
        # mask = tf.io.decode_image(tf.io.read_file(maskfn), channels=1)
        nx, ny = tf.shape(image)[0], tf.shape(image)[1]
        mask = tf.image.resize(mask[...,tf.newaxis], size=(nx//2, ny//2), method='nearest')
        # print(nx,ny)

        with open(datafn.numpy(), "r") as jsonfile:
            
            data = json.load(jsonfile)
            bboxes = np.array([v['bbox'] for v in data.values()]).astype(np.float32)
            # bboxes =0
            try:
                bboxes = util.normalize_bboxes(bboxes, nx, ny)
            except:
                tf.print("Error, cannot read boxes", bboxes, imgfn.numpy())
            # print(self.cls_ind)
            # print([[v['class']]
            #                     for v in data.values()])
            classes = np.array([v['class']
                                for v in data.values()]).astype(np.int32)
            labels = np.array(list(data.keys())).astype(np.int32)
            # print(labels)

        return os.path.splitext(os.path.basename(imgfn.numpy()))[0], \
            image, \
            mask[..., 0], \
            tf.RaggedTensor.from_tensor(tf.convert_to_tensor(bboxes, dtype=tf.float32)[tf.newaxis, ...]), \
            tf.RaggedTensor.from_tensor(tf.convert_to_tensor(classes, dtype=tf.int32)[tf.newaxis, ...]), \
            tf.RaggedTensor.from_tensor(tf.convert_to_tensor(labels, dtype=tf.int32)[tf.newaxis, ...])

    def get_img(self, imgid, normalize=True):
        key = self.ids_to_imgs[imgid]
        if normalize:
            return np.array(Image.open(self.image_dict[key])) / 255.

        return np.array(Image.open(self.image_dict[key]))

    def get_mask(self, imgid):
        key = self.ids_to_imgs[imgid]
        return np.array(Image.open(self.mask_dict[key]))

    def get_data(self, imgid):
        """Return instances data of image imgid, ie. a dict keyed by instance label and giving bboxes and class
        """
        key = self.ids_to_imgs[imgid]
        with open(self.data_dict[key], "r") as jsonfile:
            data = json.load(jsonfile)

        classes = np.array([self.cls_ind[v['class']]
                            for v in data.values()]).astype(np.int32)
        labels = np.array(list(data.keys())).astype(np.int32)
        bboxes = np.array([v['bbox'] for v in data.values()])

        return bboxes, classes, labels

    def get_bboxes(self, imgid):

        data = self.get_data(imgid)
        bboxes = np.array([v['bbox'] for v in data.values()])
        return bboxes
    
class aws_dataloader():
    
    def __init__(self,aws_path,cls_ind,local_path="data",download_data=True,val_split=False):
        
        if aws_path =="" or aws_path is None:
            raise ValueError("Must specify at least one of s3 path")
        self.local_path = local_path
        if not os.path.isabs(self.local_path):
            # os.path.getas
            self.local_path =os.path.join(os.path.abspath("."),self.local_path)
            # self.local_path = os.path.normpath(self.local_path)
        
        # with Pool() as pool:
        #     result = pool.map_async(self.data_local(aws_path,local_path))
        #     result.wait()
        self.val_split_no = 0.1
        if download_data:
            # with ThreadPoolExecutor(max_workers=10) as executor:

                self.data_local(aws_path,local_path)
                self.validation_split()
        if not download_data and  val_split:
            self.validation_split()
        print("done")
        self.trainset = DataLoader(local_path,cls_ind,dataset_type="train").dataset
        self.valset = DataLoader(local_path,cls_ind,dataset_type="val").dataset

        
        # super(aws_dataloader,self).__init__(local_path,cls_ind)
    def data_local(self,aws_path,local_path):
        # data_path = "s3://cvdb-data/sandbox2/training_data/instance_segmentation/polygon-corn_leaf/BATCH-Darwin-Leaf-Instance-Seg-11-30/20221202-144935-1024_1024/"
        dp =  DualPath(aws_path,'r',local_path =local_path,is_file=False)
        dp.to_local()
    
    def validation_split(
        self,
        seed=888,
    ):
        # print("seed",seed)
        """Create training and validation dataloader from folders of data."""
        
        paths = glob.glob(os.path.join(self.local_path, "images","*"))
        unique_names = list(set(os.path.split(x)[-1].split("_")[0] for x in paths))
        print(paths)
        current_length = len(unique_names)
        validation_length = int(math.ceil(self.val_split_no * current_length))
        def randperm(n,seed):
            indices=tf.range(n)
            shuffle_indices =  tf.random.shuffle(indices,seed=seed)
            return shuffle_indices
            # while True:
            #     yieldf np.random
        print("curren_nmame",current_length)
        indices = randperm(
            current_length, seed
        ).numpy().tolist()
        
        training_name = [
            unique_names[x] for x in indices[: (current_length - validation_length)]
        ]
        validation_name = [
            unique_names[x] for x in indices[(current_length - validation_length) :]
        ]
        # print("training_name",training_name)

        # Make the training and validation datasets
        if not os.path.isdir(os.path.join(self.local_path, "train")):
            os.mkdir(os.path.join(self.local_path, "train"))
        if not os.path.isdir(os.path.join(self.local_path, "val")):
            os.mkdir(os.path.join(self.local_path, "val"))

        for path in paths:
            im_name = os.path.split(path)[-1]
            if im_name.split("_")[0] in training_name:
                os.rename(path, os.path.join(self.local_path, "train", im_name))
            elif im_name.split("_")[0] in validation_name:
                os.rename(path, os.path.join(self.local_path, "val", im_name))

        train_ann = {}
        val_ann = {}

        if len(glob.glob(os.path.join(self.local_path, "dataset.json"))) > 0:
            local_ann_file = os.path.join(self.local_path, "dataset.json")
        else:
            local_ann_file = os.path.join(self.local_path, "labels", "dataset.json")

        f = open(local_ann_file)
        ann = json.load(f)
        train_ann["info"], val_ann["info"] = ann["info"], ann["info"]
        train_ann["categories"], val_ann["categories"] = (
            ann["categories"],
            ann["categories"],
        )
        train_ann["images"], val_ann["images"] = [], []
        train_ann["annotations"], val_ann["annotations"] = [], []

        annotations = {}
        for a in ann["annotations"]:
            if a["image_id"] in annotations:
                a["iscrowd"] = a["is_crowd"]
                a.pop("is_crowd")
                annotations[a["image_id"]].append(a)
            else:
                a["iscrowd"] = a["is_crowd"]
                a.pop("is_crowd")
                annotations[a["image_id"]] = [a]

        for i in ann["images"]:
            if i["file_name"].split("_")[0] in training_name:
                train_ann["images"].append(i)
                if annotations.get(i["id"]) is not None:
                    train_ann["annotations"].extend(annotations[i["id"]])
            elif i["file_name"].split("_")[0] in validation_name:
                val_ann["images"].append(i)
                if annotations.get(i["id"]) is not None:
                    val_ann["annotations"].extend(annotations[i["id"]])
        
        if not os.path.isdir(os.path.join(self.local_path, "labels")):
            os.mkdir(os.path.join(self.local_path, "labels"))
        # print("train_ann",train_ann)
        with open(os.path.join(self.local_path, "labels", "train.json"), "w") as f:
            json.dump(train_ann, f)
        with open(os.path.join(self.local_path, "labels", "val.json"), "w") as f:
            json.dump(val_ann, f)
        # print(COCO(os.path.join(self.local_path, "labels", "train.json")))
        
                   
# if __name__ == '__main__':    
#     cls_ind = {"corn_leaf":1,}
#     trainset = aws_dataloader("s3://cvdb-data/sandbox2/training_data/instance_segmentation/polygon-corn_leaf/SET-two-image-leaf-segmentation/20221123-150107-1024_1024/",cls_ind=cls_ind,download_data=False)
    # print(trainset)
