from trdg.generators import GeneratorFromStrings
import random
import csv
import numpy as np
from numpy.random import uniform, choice
from multiprocessing import Process
from multiprocessing import Semaphore, Manager
import cv2
from PIL import Image

def worker(name, sema, strings, q):
    batch_size=309
    print('process {} starting doing business'.format(name))
    skewing_angle= choice([0, 1, 2, 3])
    random_skew=True
    blur=0
    random_blur=False
    background_type= choice([0, 1, 2])
    distorsion_type= choice([0, 1, 2, 3])
    space_width=uniform(0.0, 8.0)
    character_spacing= choice(range(-15, 5))
    margins=(choice(range(31)), choice(range(31)), choice(range(31)), choice(range(31)))
    fit=True
    generator = GeneratorFromStrings(count=batch_size, 
                                     strings=strings,
                                     size=128,
                                     skewing_angle=skewing_angle,
                                     random_skew=random_skew,
                                     blur=blur,
                                     random_blur=random_blur,
                                     background_type=background_type,
                                     distorsion_type=distorsion_type,
                                     space_width=space_width,
                                     character_spacing=character_spacing,
                                     margins=margins,
                                     fit=fit)
    img_id = 0
    for img, lbl in generator:
        # img = distort_elastic(img)
        img_name = f"synth/{name}_{img_id}.png"
        img.save(img_name, "PNG")
        # q1.put((img, img_name))
        q.put(f"{img_name}\t{lbl}\n")
        img_id+=1

    sema.release()

def file_saver(q):
    '''listens for messages on the q, writes to file. '''

    with open('synth/synthetic.tsv', 'a') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                print('killed')
                break
            f.write(str(m))
            f.flush()

def img_saver(q2):

    while 1:
        m = q1.get()
        if isinstance(m, str):
            if m == 'kill':
                print('killed')
                break
        else:
            img, img_name = m
            img.save(img_name, "PNG")

def distort_elastic(image, alpha=34, sigma=4, random_state=None):
    """Elastic deformation of images as per [Simard2003].
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    image = np.asarray(image)
    # rate = 0.01
    # alpha = image.shape[1] * rate
    # sigma = image.shape[1] * rate
    shape_size = image.shape[:2]
    # alpha = choice(range(alpha, 40))
    # sigma = choice(range(sigma, 6))
    # Downscaling the random grid and then upsizing post filter
    # improves performance. Approx 3x for scale of 4, diminishing returns after.
    grid_scale = 4
    alpha //= grid_scale  # Does scaling these make sense? seems to provide
    sigma //= grid_scale  # more similar end result when scaling grid used.
    grid_shape = (shape_size[0]//grid_scale, shape_size[1]//grid_scale)

    blur_size = int(4 * sigma) | 1
    rand_x = cv2.GaussianBlur(
        (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    rand_y = cv2.GaussianBlur(
        (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    if grid_scale > 1:
        rand_x = cv2.resize(rand_x, shape_size[::-1])
        rand_y = cv2.resize(rand_y, shape_size[::-1])

    grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
    grid_x = (grid_x + rand_x).astype(np.float32)
    grid_y = (grid_y + rand_y).astype(np.float32)

    distorted_img = cv2.remap(image, grid_x, grid_y,
        borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)

    return Image.fromarray(distorted_img)

def distorter(q1, q2):

    while 1:
        m = q1.get()
        if isinstance(m, str):
            if m == 'kill':
                print('killed')
                break
        else:
            img, img_name = m
            q2.put((distort_elastic(img), img_name))

print("Opening corpus")
corpus = " "
with open('corpus.txt') as f:
    corpus = corpus.join(line.strip() for line in f)

print("Chunking corpus")
img_id = 0
chars = len(corpus)
_corpus = corpus
_strings = []
while chars != 0:
    if chars == 1:
        _strings.append(_corpus)
        break
    l = choice(range(1, min(100, chars)))
    _strings.append(_corpus[:l])
    _corpus = _corpus[l:]
    chars -= l
random.shuffle(_strings)
print(len(_strings))

print("Making batches")
batches = []
batch_size=309
max_batches = 500
i=0
while True:
    batch = _strings[i*batch_size:(i+1)*batch_size]
    if len(batch) == 0 or i >= max_batches:
        break
    batches.append(batch)
    i+=1

print("Starting workers")

sema = Semaphore(6)
all_processes = []

manager = Manager()
q = manager.Queue()
# q2 = manager.Queue()    
# q3 = manager.Queue()    

# distort_worker = Process(target=distorter, args=(q1, q2,))
# isaver_worker = Process(target=img_saver, args=(q2,))
fsaver_worker = Process(target=file_saver, args=(q,))

# isaver_worker.start()
fsaver_worker.start()
# distort_worker.start()

print("Working")

for name, batch in enumerate(batches):
    sema.acquire()
    p = Process(target=worker, args=(name, sema, batch, q,))
    all_processes.append(p)
    p.start()

print("Finishing work")

for p in all_processes:
    p.join()

q1.put('kill')
q2.put('kill')
q3.put('kill')

fsaver_worker.join()
# isaver_worker.join()
# distort_worker.join()

