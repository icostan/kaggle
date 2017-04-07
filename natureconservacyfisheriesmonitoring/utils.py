import time
import matplotlib.pyplot as plt

def benchmark(func):
    def wrapper(*arg, **kw):
        t1 = time.process_time()
        res = func(*arg, **kw)
        t2 = time.process_time()
        return (t2 - t1), res, func.__name__
    return wrapper

def log(label, text):
    print(str(label) + ': ' + str(text))

def plot_image(img):
    plt.imshow(img)

def normalize(X):
    X_train = X.astype('float32')
    X_train /= 255
    return X_train

#
# Bytes to MB, GB, ...
# https://gist.github.com/shawnbutts/3906915
#
def bytesto(bytes, to, bsize=1024):
    """convert bytes to megabytes, etc.
       sample code:
           print('mb= ' + str(bytesto(314575262000000, 'm')))
       sample output:
           mb= 300002347.946
    """
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    for i in range(a[to]):
        r = r / bsize

    return(r)


