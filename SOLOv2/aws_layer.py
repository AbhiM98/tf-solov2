from aws_utils.s3_dual_paths import DualPath

data_path = "s3://cvdb-data/sandbox2/training_data/instance_segmentation/polygon-corn_leaf/BATCH-Darwin-Leaf-Instance-Seg-11-30/20221202-144935-1024_1024/"
local_path = "data"
dp =  DualPath(data_path,'r',local_path =local_path,is_file=False)
dp.to_local()