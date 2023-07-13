from tensorflow.keras import models
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
X_test = pickle.load(open("X_test_apple.pickle", "rb"))
Y_test = pickle.load(open("Y_test_apple.pickle", "rb"))

'''
model = models.load_model("apple_diseases_inception__V3_2")

# results = model.evaluate(X_test, Y_test , batch_size=64)
# print("test loss, test acc:", results)

yhat = model.predict(X_test)

pickle_out = open("test_result_apple_diseases_2.pickle", "wb")
pickle.dump(yhat, pickle_out)
pickle_out.close()
'''
yhat = pickle.load(open("test_result_apple_diseases_2.pickle", "rb"))

y = []
for i in yhat:
    y.append(np.argmax(i))
'''
count = 0

for i in range(0, len(Y_test)):
    if Y_test[i] == 1 and y[i] == 2:
        print("case 0 : ", i, yhat[i], count)
        imgplot = plt.imshow(X_test[i])
        plt.show()
        count +=1

    elif Y_test[i] == 2 and y[i] == 1:
        print("case 1 : ", i, yhat[i])
        imgplot = plt.imshow(X_test[i])
        plt.show()
    elif Y_test[i] == 2 and y[i] == 3:
        print("case 2 : ", i, yhat[i])
        imgplot = plt.imshow(X_test[i])
        plt.show()
'''


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('cool')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


cm1 = confusion_matrix(Y_test, y)

plot_confusion_matrix(cm1,
                      ["Apple_scab", "Black_rot", "Apple_rust", "Healthy"],
                      title='Apple Confusion matrix',
                      cmap=None,
                      normalize=False)

# precision tp / (tp + fp)
precision = precision_score(Y_test, y, average='micro')
print('Precision : ', precision)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, y, average='micro')
print('Recall : ', recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, y, average='micro')
print('F1 score : ', f1)
