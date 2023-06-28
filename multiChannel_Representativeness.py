
import os
import copy
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from numpy import genfromtxt
import pandas as pd
import random
import cv2
from glob import glob
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from scipy import spatial
from scipy.stats import entropy

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))


#==========================================================================
#start loading data
def read_image(path):
     path = path.decode()
     x = cv2.imread(path, cv2.IMREAD_COLOR)
     x = cv2.resize(x, (256, 256))
     x = x/255.0
     return x

def read_multiMask(path1,path2):
    path1 = path1.decode()
    path2 = path2.decode()
    x1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    x1 = cv2.resize(x1, (256, 256))
    x1 = x1/255.0

    x2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    x2 = cv2.resize(x2, (256, 256))
    x2 = x2/255.0

    x = np.stack((x1, x2), axis=-1)

    return x

def check_empty_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    x=x.flatten()
    return np.sum(x)==0


def tf_parse(x, y, z):
    def _parse(x, y, z):
        x = read_image(x)
        y = read_multiMask(y,z)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y, z], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 2])
    return x, y

def tf_dataset(x, y, z, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y, z))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#Data augmentation ,each image 5 transformations 
def augment_data(images, masks1, masks2, save_path, augRatio = 0.2):
    H = 256
    W = 256

    for x, y, z in tqdm(zip(images, masks1, masks2), total=len(images)):
        name = x.split("/")[-1].split(".")
        """ Extracting the name and extension of the image and the mask. """
        image_name = name[0]
        image_extn = name[1]

        name = y.split("/")[-1].split(".")
        mask1_name = name[0]
        mask1_extn = name[1]

        name = z.split("/")[-1].split(".")
        mask2_name = name[0]
        mask2_extn = name[1]

        """ Reading image and mask. """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        #contrast enhence
        r_image, g_image, b_image = cv2.split(x)
        r_image_eq = cv2.equalizeHist(r_image)
        g_image_eq = cv2.equalizeHist(g_image)
        b_image_eq = cv2.equalizeHist(b_image)
        x = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
        #make mask(1 channel) to white(3 channel)
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        y[:,:,0]=y[:,:,2]
        y[:,:,1]=y[:,:,2]
        z = cv2.imread(z, cv2.IMREAD_COLOR)
        z[:,:,2]=z[:,:,0]
        z[:,:,1]=z[:,:,0]

        #randomNum = random.uniform(0,1)
        """ Augmentation """
        '''if randomNum <= augRatio:
            #aug = CenterCrop(H, W)
            #augmented = aug(image=x, mask=y)
            #x1 = augmented["image"]
            #y1 = augmented["mask"]

            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = GridDistortion(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            save_images = [x, x2, x3, x4, x5]
            save_masks =  [y, y2, y3, y4, y5]

        else:'''
        save_images = [x]
        save_masks1 = [y]
        save_masks2 = [z]

        """ Saving the image and mask. """
        idx = 0
        for i, m1, m2 in zip(save_images, save_masks1,save_masks2):
            i = cv2.resize(i, (W, H))
            m1 = cv2.resize(m1, (W, H))
            m2 = cv2.resize(m2, (W, H))

            if len(images) == 1:
                tmp_img_name = f"{image_name}.{image_extn}"
                tmp_mask1_name = f"{mask1_name}.{mask1_extn}"
                tmp_mask2_name = f"{mask2_name}.{mask2_extn}"
            else:
                tmp_img_name = f"{image_name}_{idx}.{image_extn}"
                tmp_mask1_name = f"{mask1_name}_{idx}.{mask1_extn}"
                tmp_mask2_name = f"{mask2_name}_{idx}.{mask2_extn}"

            image_path = os.path.join(save_path, "images", tmp_img_name)
            mask1_path = os.path.join(save_path, "masks1", tmp_mask1_name)
            mask2_path = os.path.join(save_path, "masks2", tmp_mask2_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask1_path, m1)
            cv2.imwrite(mask2_path, m2)

            idx += 1


def load_data(path,  split2=0.2,split3=0.2, augRatio=0.2):
    images = sorted(glob(os.path.join(path, "tiff/*")))
    masks1 = sorted(glob(os.path.join(path, "lesion/*")))
    masks2 = sorted(glob(os.path.join(path, "solid/*")))
    print("original image num:", len(images))
    
    
    #Creating folders.
    create_dir("new_data_multi/images")
    create_dir("new_data_multi/masks1")
    create_dir("new_data_multi/masks2")

    #Applying data augmentationon training dataset
    augment_data(images,masks1,masks2, "new_data_multi",augRatio)
    path2 = "new_data_multi/"
    images = sorted(glob(os.path.join(path2, "images/*")))
    masks1 = sorted(glob(os.path.join(path2, "masks1/*")))
    masks2 = sorted(glob(os.path.join(path2, "masks2/*")))
    print("after augmentation, image num:", len(images))
    #size1 = int(len(images) * split1)
    size2 = int(len(images) * split2)
    size3 = int(len(images) * split3)


    #train_x, valid_x = train_test_split(images, test_size=size1, random_state=42)
    #train_y, valid_y = train_test_split(masks, test_size=size1, random_state=42)

    train_x, test_x = train_test_split(images, test_size=size2, random_state=42)
    train_y, test_y = train_test_split(masks1, test_size=size2, random_state=42)
    train_z, test_z = train_test_split(masks2, test_size=size2, random_state=42)

    pool_x, train_x = train_test_split(train_x, test_size=size3, random_state=42)
    pool_y, train_y = train_test_split(train_y, test_size=size3, random_state=42)
    pool_z, train_z = train_test_split(train_z, test_size=size3, random_state=42)

    return (train_x, train_y, train_z), (pool_x, pool_y, pool_z),  (test_x, test_y, test_z)

def saveList(path,saveList):
    with open(path, "w") as f:
        for s in saveList:
            f.write(str(s) +"\n")

def openList(path):
    openedList=[]
    with open(path, "r") as f:
        for line in f:
            openedList.append(line.strip())
    return openedList

""" Dataset """
dataset_path = "dataset_to_use/ProData/"
(train_x, train_y, train_z),(pool_x, pool_y, pool_z), (test_x, test_y, test_z) = load_data(dataset_path, split2=0.2,split3=0.05, augRatio=0)


saveList("multiInitialSet/train_xm.txt",train_x)
saveList("multiInitialSet/train_ym.txt",train_y)
saveList("multiInitialSet/train_zm.txt",train_z)
saveList("multiInitialSet/pool_xm.txt",pool_x)
saveList("multiInitialSet/pool_ym.txt",pool_y)
saveList("multiInitialSet/pool_zm.txt",pool_z)
saveList("multiInitialSet/test_xm.txt",test_x)
saveList("multiInitialSet/test_ym.txt",test_y)
saveList("multiInitialSet/test_zm.txt",test_z)
print('======data loading finished======')


#==========================================================================
# Build Unet and train model
def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def build_model():
    size = 256
    num_filters = [16, 32, 48, 64, 128]
    inputs = Input((size, size, 3))

    skip_x = []
    x = inputs
    ## Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)

    ## Bridge
    x = conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    ## Output
    x = Conv2D(2, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)

#Evaluation Matrix
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        y_truef = tf.keras.layers.Flatten()(y_true)
        y_predf = tf.keras.layers.Flatten()(y_pred)
        #if np.sum(y_predf)==0 or np.sum(y_truef)==0:
        if np.sum(y_truef)==0:
            y_true=1-y_true
            y_pred=1-y_pred
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred):
    #smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    #if np.sum(y_pred)==0 or np.sum(y_true)==0:
    if np.sum(y_true)==0:
        y_true=1-y_true
        y_pred=1-y_pred
    intersection = tf.reduce_sum(y_true * y_pred)
    outcome=(2* intersection) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    #print(outcome)
    return outcome

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

#train the model using augmented data
def trainModel(train_x, train_y, train_z, valid_x, valid_y,valid_z, folder,
               batch_size = 5, lr = 0.001, num_epochs = 80):
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    #Hyperparaqmeters 
    #batch_size = 5
    #lr = 0.001   ## 0.0001
    #num_epochs = 80
    
    model_path = "model"+str(folder)+".h5"
    #csv_path = "data1.csv"
    
    """Load Augmented Dataset For Training """
    #new_path = "new_data/"
    print(f"start new folder, Train: {len(train_x)} - {len(train_y)} - {len(train_z)}")
   
    
    train_dataset = tf_dataset(train_x, train_y, train_z, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, valid_z, batch=batch_size)


    train_steps = (len(train_x)//batch_size)
    valid_steps = (len(valid_x)//batch_size)     
    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    """ Model """
    model = build_model()
    #metrics = [dice_coef, iou, Recall(), Precision()]
    metrics = [Recall(), Precision()]
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=metrics)
    #else:
        #with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
            #model = tf.keras.models.load_model("fullmodel.h5")
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        #CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )

    return model

def crossVali(n_split,epoch):
    X=np.array(train_x)
    Y=np.array(train_y)
    Z=np.array(train_z)
    folderNum=0
    for train_index,valid_index in KFold(n_split).split(X):
        x_train,x_valid=X[train_index],X[valid_index]
        y_train,y_valid=Y[train_index],Y[valid_index]
        z_train,z_valid=Z[train_index],Z[valid_index]
        model=trainModel(train_x=x_train.tolist(), train_y=y_train.tolist(), train_z=z_train.tolist(), 
                         valid_x=x_valid.tolist(), valid_y=y_valid.tolist(), valid_z=z_valid.tolist(),folder=folderNum,
               batch_size = 5, lr = 0.001, num_epochs = epoch)
        folderNum += 1


#==========================================================================
#Evaluation and Pick new training instances
H = 256
W = 256

'''def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
'''
def read_testimage(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)   ## (1, 256, 256, 3)
    return ori_x, x

def read_testmultimask(path1, path2):
    #path1 = path1.decode()
    #path2 = path2.decode()
    x1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    x2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    ori_x=np.stack((x1, x2), axis=-1)

    x1 = cv2.resize(x1, (256, 256))
    x1 = x1/255.0
    x2 = cv2.resize(x2, (256, 256))
    x2 = x2/255.0

    x = np.stack((x1, x2), axis=-1)
    x = x>0.5
    x = x.astype(np.int32)
    return ori_x, x

def read_testsinglemask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x>0.5
    x = x.astype(np.int32)
    return ori_x, x

def save_result(ori_x, ori_y, y_pred, save_path):
    line = np.ones((H, 10, 3)) * 255

    #ori_y = np.expand_dims(ori_y, axis=-1) ## (256, 256, 1)
    oriy_3d = np.zeros((256, 256, 3))
    ori_y = 255-ori_y
    oriy_3d[:,:,0:2] = ori_y
    #oriy_3d=255-oriy_3d
    #print("oriy shape:",oriy_3d.shape)

    #y_pred = np.expand_dims(y_pred, axis=-1)
    #print(y_pred.shape)
    predy_3d = np.zeros((256, 256, 3))
    y_pred = (1-y_pred) * 255.0
    predy_3d[:,:,0:2] = y_pred
    #predy_3d = (1-predy_3d) * 255.0
    #print("predy 3d shape:", predy_3d.shape)
    
    cat_images = np.concatenate([ori_x, line, oriy_3d, line, predy_3d], axis=1)
    cv2.imwrite(save_path, cat_images)

""" Prediction and metrics values """
def evaluation(SCORE,kFold):
    models=[]
    for foldNum in range(kFold):
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
                modelName="model"+str(foldNum)+".h5"
                models.append(tf.keras.models.load_model(modelName))
    for x, y, z in tqdm(zip(test_x, test_y, test_z), total=len(test_x)):
        name = x.split("/")[-1]

        """ Reading the image and mask """
        ori_x, x = read_testimage(x)
        ori_y, y = read_testsinglemask(y)
        ori_z, z = read_testsinglemask(z)
        
        for foldNum in range(kFold):
            model=models[foldNum]
            if foldNum==0:
                y_pred=model.predict(x)
            else:
                y_pred+=model.predict(x)
        y_pred = (y_pred/foldNum)[0]>0.5
        #y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.int32)
        #print(y_pred.shape)
        #save_path = f"results/{name}"
        #save_result(ori_x, ori_y, y_pred, save_path)
        y_pred1=y_pred[:,:,0]
        y_pred2=y_pred[:,:,1]

        """ Flattening the numpy arrays. """
        y = y.flatten()
        z = z.flatten()
        y_pred1 = y_pred1.flatten()
        y_pred2 = y_pred2.flatten()
        print(f"y length: {len(y)}, z length: {len(z)}, y_pred1 length: {len(y_pred1)} ,y_pred2 length: {len(y_pred2)}")
        #acc_value = accuracy_score(y, y_pred)
        iou_value1=iou(y,y_pred1)
        iou_value2=iou(z,y_pred2)
        dic_coef_value1=dice_coef(y,y_pred1)
        dic_coef_value2=dice_coef(z,y_pred2)
        dice_loss_value1=1-dic_coef_value1
        dice_loss_value2=1-dic_coef_value2
        f1_value1 = f1_score(y, y_pred1, labels=[0, 1], average="binary")
        f1_value2 = f1_score(z, y_pred2, labels=[0, 1], average="binary")
        #jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        #recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
        #precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        SCORE.append([name, iou_value1, iou_value2, dic_coef_value1, dic_coef_value2,
                      dice_loss_value1, dice_loss_value2, f1_value1, f1_value2])        

#methods: shannon entropy=0, least confidence=1, margin = 2
def scanPool(kFold):
    method = 2
    models=[]
    selected=[]
    for foldNum in range(kFold):
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
            modelName="model"+str(foldNum)+".h5"
            models.append(tf.keras.models.load_model(modelName))
    for idx in range(len(pool_x)):
        name_x=pool_x[idx]
        name_y=pool_y[idx]
        name_z=pool_z[idx]
        name = name_x.split("/")[-1]

        """ Reading the image and mask """
        ori_x, x = read_testimage(name_x)
        #ori_y, y = read_testmask(name_y)
        for foldNum in range(kFold):
            model=models[foldNum]
            if foldNum==0:
                y_pred=model.predict(x)
            else:
                y_pred+=model.predict(x)
        y_pred = (y_pred/foldNum)[0]
        #y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.flatten()
        if method == 0:
            entro=entropy(y_pred)
            selected.append([name_x,name_y, name_z,entro])
        elif method == 1:
            leastConfidence = 1 - np.maximum(y_pred,1-y_pred)
            leastConfidence = np.mean(leastConfidence)
            selected.append([name_x,name_y, name_z,leastConfidence])
        elif method ==2:
            margin=-abs(y_pred-(1-y_pred))
            margin=np.mean(margin)
            selected.append([name_x,name_y,name_z,idx,margin])
    selected.sort(key=lambda x:x[4],reverse=True)
    return selected[:15]

def scanUncertainSet(uncertain,sim_matrix):
    selected=[]
    for i in range(len(uncertain)):
        name_x = uncertain[i][0]
        name_y = uncertain[i][1]
        name_z = uncertain[i][2]
        idx = uncertain[i][3]
        #calculate the sum of similarity
        sim_sum=0
        for j in range(len(uncertain)):
            idx2=uncertain[j][3]
            sim_sum += sim_matrix[idx][idx2]
        selected.append([name_x,name_y,name_z,idx,sim_sum])
    selected.sort(key=lambda x:x[4],reverse=True)
    return selected[:6]

def selectNextRepresen(iteration, selected, restidx,sim_matrix):
    rank = []
    if iteration==0:
        for idx in restidx:
            sim_sum = 0
            for j in restidx:
                sim_sum += sim_matrix[idx][j]
                #print("iteration, sim_sum:",iteration, sim_sum)
            rank.append([idx,sim_sum])
    else:
        for idx in restidx:
            #temporily selected next one
            selected1=copy.deepcopy(selected)
            selected1.append(idx)
            restidx1=copy.deepcopy(restidx)
            restidx1.remove(idx)
            sim_sum=0
            for idx in restidx1:
                #scan the rest instances
                sims=[]
                for i in selected1:
                    sims.append(sim_matrix[idx][i])
                #max sim(Ix, Ij)
                maxsim = np.max(sims)
                sim_sum += maxsim
                #print("iteration, sim_sum:",iteration, sim_sum)
            rank.append([idx,sim_sum])
    rank.sort(key=lambda x:x[1],reverse=True)
    selectedIdx = rank[0][0]
    selected.append(selectedIdx)
    restidx.remove(selectedIdx) 
    return rank[0][1]       

def coverUncertainSet(uncertain,sim_matrix):
    #uncertain: [namex,namey,idx,margin]
    idxdict = {}
    selectedidx=[]
    restidx = []
    
    for i in range(len(uncertain)):
        restidx.append(uncertain[i][3])
        idxdict[uncertain[i][3]]=i
    for iteration in range(6):
        sim_sum = selectNextRepresen(iteration, selectedidx, restidx,sim_matrix)
        #print("current selected num & rest sum $ sim:", len(selectedidx),len(restidx),sim_sum)

    selected=[]
    for idx in selectedidx:
        uncerIdx = idxdict.get(idx)
        name_x = uncertain[uncerIdx][0]
        name_y = uncertain[uncerIdx][1]
        name_z = uncertain[uncerIdx][2]
        selected.append([name_x,name_y,name_z,idx])
    return selected

def saveSampleResult(x,y,z,iteration,kFold):
    models=[]
    for foldNum in range(kFold):
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
                modelName="model"+str(foldNum)+".h5"
                models.append(tf.keras.models.load_model(modelName))
    name = x.split("/")[-1]
    ori_x, x = read_testimage(x)
    ori_y, y = read_testmultimask(y,z)

    for foldNum in range(kFold):
        model=models[foldNum]
        if foldNum==0:
            y_pred=model.predict(x)
        else:
            y_pred+=model.predict(x)
    y_pred = (y_pred/foldNum)[0]>0.5
    #y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred.astype(np.int32)
    
    filename="multiRepre_"+str(iteration)+"_"+name
    save_path = f"results2/{filename}"
    save_result(ori_x, ori_y, y_pred, save_path)

def deleteSimMatrix(sim_matrix, selected_idx):
    for i in range(len(selected_idx)):
        sim_matrix=np.delete(sim_matrix,selected_idx[i]-i,0)
        sim_matrix=np.delete(sim_matrix,selected_idx[i]-i,1)
    return sim_matrix


#get similarity matrix
def featureModel(validRatio,epoch):
    X=np.array(train_x)
    Y=np.array(train_y)
    Z=np.array(train_z)
    lenValid=int(len(X)*validRatio)
    x_train,x_valid=X[:lenValid],X[lenValid:]
    y_train,y_valid=Y[:lenValid],Y[lenValid:]
    z_train,z_valid=Z[:lenValid],Z[lenValid:]
    model=trainModel(train_x=x_train.tolist(), train_y=y_train.tolist(), train_z=z_train.tolist(), 
                        valid_x=x_valid.tolist(), valid_y=y_valid.tolist(), valid_z=z_valid.tolist(),folder='feature',
            batch_size = 5, lr = 0.001, num_epochs = epoch)

train_x=openList("multiInitialSet/train_xm.txt")
train_y=openList("multiInitialSet/train_ym.txt")
train_z=openList("multiInitialSet/train_zm.txt")
featureModel(0.2,100)

def get_encoder_output(model):
    inputs = model.input  
    encoder_output = model.layers[40].output
    encoder_model = Model(inputs=inputs, outputs=encoder_output)
    
    return encoder_model

#get the convolutional outcome, AKA features
def get_featureMatrix(modelName):
    feature=[]
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
        #modelName="feature_modelM.h5"
        feature_model=tf.keras.models.load_model(modelName)
    for x in pool_x:
        name = x.split("/")[-1]
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        #print(x)
        x = cv2.resize(x, (256, 256))
        x = x/255.0
        x = np.expand_dims(x, axis=0)
        #print("x shape ",x.shape)
        #x= tf.expand_dims(x, axis=0)
        outcomes=[]
        encoder_model = get_encoder_output(feature_model)
        encoder_output = encoder_model.predict(x)
        print(encoder_output.shape)
        #print(meanOutcome[0].shape)
        cur=[]
        
        cur.append(name)
        cur.append(encoder_output.flatten())
        feature.append(cur)
    sim_matrix=np.zeros(shape=(len(feature), len(feature)))
    for i in range(len(feature)):
        for j in range(i):
            if i==j:
                sim_matrix[i][i]=1.0
            else:
                sim = -1 * (spatial.distance.cosine(feature[i][1], feature[j][1]) - 1)
                sim_matrix[i][j]=sim*1.0
                sim_matrix[j][i]=sim*1.0
    return sim_matrix

sim_matrix=get_featureMatrix('modelfeature.h5')
np.savetxt("similarityM.csv", sim_matrix, delimiter=",")
print('======similarity matrix finished======')

#==========================================================================
#Execute
if __name__ == "__main__":
    summary = pd.DataFrame(columns=['iou_value1', 'iou_value2', 
                                                    'dic_coef_value1', 'dic_coef_value2', 
                                                    'dice_loss_value1', 'dice_loss_value2', 
                                                    'f1_value1', 'f1_value2'])
    selectedRecord=[]
    kfold=3
    print('=========================================')
    print('======start active learning process======')
    sim_matrix = genfromtxt('similarityM.csv', delimiter=',')
    train_x=openList("multiInitialSet/train_xm.txt")
    train_y=openList("multiInitialSet/train_ym.txt")
    train_z=openList("multiInitialSet/train_zm.txt")
    pool_x=openList("multiInitialSet/pool_xm.txt")
    pool_y=openList("multiInitialSet/pool_ym.txt")
    pool_z=openList("multiInitialSet/pool_zm.txt")
    test_x=openList("multiInitialSet/test_xm.txt")
    test_y=openList("multiInitialSet/test_ym.txt")
    test_z=openList("multiInitialSet/test_zm.txt")
    for i in range(20):
        print("===current iteration:"+str(i))
        crossVali(kfold,100)
        SCORE=[]
        print("evaluation ...")
        evaluation(SCORE,kfold)
        pureScore = [s[1:]for s in SCORE]
        pureScore = np.mean(pureScore, axis=0)
        print(pureScore)
        pureScore=pureScore.tolist()
        #pureScore.insert(0,method)
        summary.loc[len(summary.index)] = pureScore
        print("scanning pool...")
        uncertain=scanPool(kFold = 3)
        print("scanning uncertain set.")
        selected = coverUncertainSet(uncertain,sim_matrix)
        selectedList=[]
        selectedIdx=[]
        for j in range(6):
            cur=selected[j]
            pool_x.remove(cur[0])
            pool_y.remove(cur[1])
            pool_z.remove(cur[2])
            train_x.append(cur[0])
            train_y.append(cur[1])
            train_z.append(cur[2])
            selectedIdx.append(cur[3])
            name = cur[0].split("/")[-1]
            selectedList.append(name)
        sim_matrix = deleteSimMatrix(sim_matrix,selectedIdx)
        print("sim matrix shape ", sim_matrix.shape)
        selectedRecord.append(selectedList)
        print("save sample result of this iteration")
        saveSampleResult(test_x[10],test_y[10], test_z[10],i,kfold)
        #saveSampleResult(test_x[30],test_y[30],method,i)
    #print("method done, save sample final result")
    #saveFinalResult(test_x[0],test_y[0],method)
    #saveFinalResult(test_x[10],test_y[10],method)
    #saveFinalResult(test_x[30],test_y[30],method)
    summary.to_csv("summaryMultiRepre.csv")
    print('======metrics summary saved========')
    np.savetxt("selectedRecordmulti.csv", selectedRecord,delimiter =", ", fmt ='% s')
    print('======selected instances recording saved======')    

'''
#used to clear gpu
from numba import cuda 

device = cuda.get_current_device()
device.reset()
'''