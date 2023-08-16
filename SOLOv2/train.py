import SOLOv2
from tensorflow.keras.optimizers.experimental import SGD 
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
def get_train_and_valset(aws_path):
    dataset = SOLOv2.aws_dataloader(aws_path,cls_ind=cls_ind,download_data=False,val_split=False)
    trainset = dataset.trainset
    valset = dataset.valset
    return trainset,valset
# tf.compat.v1.disable_eager_execution()
if __name__=='__main__':
    # disable_eager_execution()
    cls_ind = {"corn_leaf":1,}
    config = SOLOv2.Config_tf()
    mySOLOv2model = SOLOv2.model.SOLOv2Model(config)
    aws_path = "s3://cvdb-data/sandbox2/training_data/instance_segmentation/polygon-corn_leaf/SET-two-image-leaf-segmentation/20221123-150107-1024_1024/"
    trainset,valset = get_train_and_valset(aws_path)
    # for x,y,z,a,b,c in trainset:
    #     print(x)
    #     if len(a)==0:
    #         print(1)
    # for x,y,z,a,b,c in valset:
        # print(tf.shape(a))
        # if len(a)==0:
        #     print(1)
        # print(x,y,z,a,b,c)
        # break
    # print(trainset)
    batch_size = 2
    optimizer = SGD(lr=0.01,momentum=0.9)
    SGD._name="SGD"
    input=[]

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_total_loss', patience=4)
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="./checkpoint",
    save_weights_only=False,
    monitor='val_total_loss',
    mode='min',
    save_best_only=True)
    callbacks = [model_checkpoint_cb]
    for a,b,c,d,e,f in trainset:
        a
    #     inp = tf.expand_dims(b,axis=0)
    #     # tf.print(tf.shape(c))
    #     # tf.print(d)
    #     # tf.print(e)
    #     # tf.print(f)
        break
    # m = mySOLOv2model(inp)
    # print(mySOLOv2model.summary())
    SOLOv2.model.train(mySOLOv2model,
                    trainset,
                    epochs=10,
                    val_dataset=None,
                    steps_per_epoch=1,#len(trainset) // batch_size,
                    validation_steps= None,
                    batch_size=batch_size,
                    callbacks = callbacks,
                    optimizer=optimizer,
                    prefetch=tf.data.AUTOTUNE,
                    buffer=150)
    # SOLOv2.evaluate(mySOLOv2model,trainset,maxiter=4)