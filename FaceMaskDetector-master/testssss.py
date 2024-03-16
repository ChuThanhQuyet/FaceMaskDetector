import numpy as np
from keras.models import load_model
from keras.preprocessing import image
mymodel=load_model('mymodel.h5')
#test_image=image.load_img('C:/Users/Karan/Desktop/ML Datasets/Face Mask Detection/Dataset/test/without_mask/30.jpg',target_size=(150,150,3))
test_image=image.load_img(r'mask_.jpg',
                          target_size=(150,150,3))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
a = mymodel.predict(test_image)[0][0]
print(a)