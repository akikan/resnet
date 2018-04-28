from keras.layers import Dense, Dropout, Activation, Flatten, Input, add
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import Model

def rescell(data, filters, kernel_size, option=False):
    strides=(1,1)
    if option:
        strides=(2,2)
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same")(data)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    
    data=Conv2D(filters=int(x.shape[3]), kernel_size=(1,1), strides=strides, padding="same")(data)
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=(1,1),padding="same")(x)
    x=BatchNormalization()(x)
    x=add([x,data])
    x=Activation('relu')(x)
    return x

def ResNet(img_rows, img_cols, img_channels, x_train):
	input=Input(shape=(img_rows,img_cols,img_channels))
	x=Conv2D(32,(7,7), padding="same", input_shape=x_train.shape[1:],activation="relu")(input)
	x=MaxPooling2D(pool_size=(2,2))(x)

	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))

	x=rescell(x,128,(3,3),True)

	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))

	x=rescell(x,256,(3,3),True)

	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))

	x=rescell(x,512,(3,3),True)

	x=rescell(x,512,(3,3))
	x=rescell(x,512,(3,3))

	x=AveragePooling2D(pool_size=(int(x.shape[1]),int(x.shape[2])),strides=(2,2))(x)

	x=Flatten()(x)
	x=Dense(units=10,kernel_initializer="he_normal",activation="softmax")(x)
	model=Model(inputs=input,outputs=[x])
	return model