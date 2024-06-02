
import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
import tempfile

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, IterableDataset, dataloader, distributed
from tqdm import tqdm

import tarfile
import webdataset as wds
import braceexpand

from utils.augmentations import (
    Albumentations,
    augment_hsv,
    classify_albumentations,
    classify_transforms,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)
from utils.general import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    check_dataset,
    check_requirements,
    check_yaml,
    clean_str,
    cv2,
    is_colab,
    is_kaggle,
    segments2boxes,
    unzip_file,
    xyn2xy,
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywhn,
)
from utils.torch_utils import torch_distributed_zero_first



# Parameters
HELP_URL = "See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data"
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def create_webdataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    seed=0,
    cache_path='./webdataset_cache'
):
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = WebDatasetLoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            rank=rank,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None #if rank == -1 else SmartDistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader #if image_weights #else InfiniteDataLoader  # only DataLoader allows for attribute updates
    
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(
        dataset,
        batch_size=None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=WebDatasetLoadImagesAndLabels.collate_fn4 if quad else WebDatasetLoadImagesAndLabels.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset


def process_batch_dict(f):
    f['jpg'] = np.array(f['jpg'])
    if 'txt' not in f.keys():
        f['txt'] = ''
    f['path'] = os.path.join(f['__url__'], f['__key__'])
    return f

def labels_from_str(label_str: str):
    if label_str == '':
        labels = np.empty((0, 5), dtype=np.float32)
    else:
        labels = []
        for line in label_str.splitlines():
            labels.append([float(num) for num in line.strip().split()])
        labels = np.array(labels, dtype=np.float32)
    return labels

def webdataset_collate_fn(batch):
    # breakpoint()
    images, labels, paths = zip(*batch)  # transposed
    return images, labels, paths


class WebDatasetLoadImagesAndLabels(IterableDataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        min_items=0,
        prefix="",
        rank=-1,
        seed=0,
        cache_path='./webdataset_cache'
    ):
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.pad = pad
        self.path = path
        self.albumentations = Albumentations(size=img_size) if augment else None
        self.web_dataset = wds.WebDataset(
            urls=path
        ).shuffle(1000
        ).decode('rgb8', wds.autodecode.basichandlers, #basic handler converts .txt from bytes to string
        ).map(process_batch_dict
        ).rename(image='jpg', labels='txt'
        ).map_dict(labels=labels_from_str, image=lambda img : img[:, :, ::-1] #RGB -> BGR
        ).to_tuple("image", "labels", "path"
        ).batched(batch_size, collation_fn=webdataset_collate_fn
        )

        self.im_files = dict()
        try:
            f = []
            for tarpath_family in path if isinstance(path, list) else [path]:
                for tarpath in braceexpand.braceexpand(tarpath_family):
                    with tarfile.open(tarpath) as tarf:
                        f += tarf.getnames()
                    self.im_files[tarpath] = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            assert self.im_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {path}: {e}\n{HELP_URL}") from e
        
        # Check cache
        temp = []
        for key, val in self.im_files.items():
            for im_path in val:
                temp.append(os.path.join(key, im_path))
        self.label_files = img2label_paths(temp)  # labels
        cache_path = (Path(cache_path)).with_suffix(".cache")
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version  # matches current version
            assert cache["hash"] == get_hash(list(self.label_files.keys()) + list(self.im_files.keys()))  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings
        assert nf > 0 or not augment, f"{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0 or not augment, f"{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}"
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        # self.im_files = list(cache.keys())  # update
        self.im_files = dict()
        for archived_path in list(cache.keys()):
            tarpath, im_path = os.path.split(archived_path)
            if tarpath not in self.im_files.keys():
                self.im_files[tarpath] = [im_path]
            else:
                self.im_files[tarpath].append(im_path)
        self.label_files = img2label_paths(cache.keys())  # update

        # Filter images
        if min_items:
            raise NotImplementedError("min_items not implemented")
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            LOGGER.info(f"{prefix}{n - len(include)}/{n} images filtered from dataset")
            self.im_files = [self.im_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]  # wh

        # Create indices
        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = np.arange(n)
        if rank > -1:  # DDP indices (see: SmartDistributedSampler)
            # force each rank (i.e. GPU process) to sample the same subset of data on every epoch
            self.indices = self.indices[np.random.RandomState(seed=seed).permutation(n) % WORLD_SIZE == RANK]

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        self.segments = list(self.segments)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = dict()
            self.label_files = dict()
            for tarpath_family in path if isinstance(path, list) else [path]:
                for tarpath in braceexpand.braceexpand(tarpath_family):
                    with tarfile.open(tarpath) as f:
                        for path in f.getnames():
                            if path.endswith('.jpg') or path.endswith('.jpeg'):
                                if tarpath not in self.im_files.keys():
                                    self.im_files[tarpath] = [path]
                                else:
                                    self.im_files[tarpath].append(path)
                            elif path.endswith('.txt'):
                                if tarpath not in self.label_files.keys():
                                    self.label_files[tarpath] = [path]
                                else:
                                    self.label_files[tarpath].append(path)
                            else:
                                raise ValueError(f"Unsupported file extension on file: {os.path.join(tarpath, path)}")
            # self.im_files = [self.im_files[i] for i in irect]
            # self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride


    def check_cache_ram(self, safety_margin=0.1, prefix=""):
        """Checks if available RAM is sufficient for caching images, adjusting for a safety margin."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.n, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.n / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(
                f'{prefix}{mem_required / gb:.1f}GB RAM required, '
                f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
        return cache

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        """Caches dataset labels, verifies images, reads shapes, and tracks dataset integrity."""
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        # desc = f"{prefix}Scanning {path.parent / path.stem}..."
        desc = 'temp desc'
        for tarpath in self.im_files.keys():
            with tempfile.TemporaryDirectory() as tmpdir:
                with tarfile.open(tarpath) as f:
                    f.extractall(tmpdir)
                    with Pool(NUM_THREADS) as pool:
                        # pbar = tqdm(
                        #     pool.imap(verify_image_label, zip(repeat(tmpdir), self.im_files[tarpath], self.label_files[tarpath], repeat(prefix))),
                        #     desc=desc,
                        #     total=len(self.im_files[tarpath]),
                        #     bar_format=TQDM_BAR_FORMAT,
                        # )
                        results = [[] for _ in range(9)]
                        for args in tqdm(zip(repeat(tmpdir), self.im_files[tarpath], self.label_files[tarpath], repeat(prefix))):
                            im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg = verify_image_label(args)
                        # for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                            nm += nm_f
                            nf += nf_f
                            ne += ne_f
                            nc += nc_f
                            if im_file:
                                x[os.path.join(tarpath, im_file)] = [lb, shape, segments]
                            # if msg:
                            #     msgs.append(msg)
                        #     pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
                        # # pool.join()
                # pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(list(self.label_files.keys()) + list(self.im_files.keys()))
        x["results"] = nf, nm, ne, nc, sum([len(val) for val in self.im_files.values()])
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}")  # not writeable
        return x

    def __len__(self):
        """Returns the number of images in the dataset."""
        n_samples = sum([len(self.im_files[tarpath]) for tarpath in self.im_files.keys()])
        n_batches = n_samples // self.batch_size + 1 if n_samples % self.batch_size != 0 else n_samples // self.batch_size
        return n_batches
    
    def __next__(self):
        try:
            batch = self.preprocess_batch(next(self.iterable_dataset))
            return batch
        except StopIteration as e:
            raise e
        
    def __iter__(self):
        self.iterable_dataset = iter(self.web_dataset)
        return self


    def preprocess_batch(self, batch):
        # temp = next(self.web_dataset)
        imgs, batch_labels, img_paths = batch
        if self.rect:
            ari = []
            rect_shape = [1, 1]
            for img in imgs:
                s = img.shape
                ari.append(s[0] / s[1]) #original code operated on PIL image so width height but here we have height width so indices reversed 
            ari = np.array(ari)
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                rect_shape = [maxi, 1]
            elif mini > 1:
                rect_shape = [1, 1 / mini]
            rect_shape = np.ceil(np.array(rect_shape) * self.img_size / self.stride + self.pad).astype(int) * self.stride
        # imgs, batch_labels = self.web_dataset.__iter__()
        resized_imgs = []
        for im in imgs if isinstance(imgs, (list, tuple)) else [imgs]:
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            resized_imgs.append((im, (h0, w0), im.shape[:2]))  # im, hw_original, hw_resized
        
        hyp = self.hyp

        final_imgs = []
        final_labels = []
        i = 0
        shapes = list() if self.rect else []
        # breakpoint()
        #current method causes variable size batches. Could modify in future, idk
        while i < len(resized_imgs):
            mosaic = self.mosaic and random.random() < hyp["mosaic"]
            if mosaic and i <= (len(resized_imgs) - 4):
                img, labels = self.create_mosaic(resized_imgs[i:i+4], batch_labels[i:i+4])
                i += 4

                if random.random() < hyp["mixup"] and i <= (len(resized_imgs) - 4):
                    img, labels = mixup(img, labels, self.create_mosaic(resized_imgs[i:i+4], batch_labels[i:i+4]))
                    i += 4

            else:
                img, (h0, w0), (h, w) = resized_imgs[i]

                shape = rect_shape if self.rect else self.img_size
                img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
                shapes.append(((h0, w0), ((h / h0, w / w0), pad))) # for COCO mAP rescaling
                labels = batch_labels[i]

                if labels.size:  # normalized xywh to pixel xyxy format
                    labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

                if self.augment:
                    img, labels = random_perspective(
                        img,
                        labels,
                        degrees=hyp["degrees"],
                        translate=hyp["translate"],
                        scale=hyp["scale"],
                        shear=hyp["shear"],
                        perspective=hyp["perspective"],
                    )
                i += 1

            nl = len(labels)  # number of labels
            if nl:
                labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)

            if self.augment:
                # Albumentations
                img, labels = self.albumentations(img, labels)
                nl = len(labels)  # update after albumentations

                # HSV color-space
                augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

                # Flip up-down
                if random.random() < hyp["flipud"]:
                    img = np.flipud(img)
                    if nl:
                        labels[:, 2] = 1 - labels[:, 2]

                # Flip left-right
                if random.random() < hyp["fliplr"]:
                    img = np.fliplr(img)
                    if nl:
                        labels[:, 1] = 1 - labels[:, 1]

                # Cutouts
                # labels = cutout(img, labels, p=0.5)
                # nl = len(labels)  # update after cutout

            labels_out = torch.zeros((nl, 6))
            if nl:
                labels_out[:, 1:] = torch.from_numpy(labels)

            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            # img = np.ascontiguousarray(img)

            final_imgs.append(img)
            final_labels.append(labels_out)

        final_imgs = torch.from_numpy(np.ascontiguousarray(final_imgs))
        if shapes is not None:
            shapes = tuple(shapes)
        return final_imgs, final_labels, img_paths, shapes


    def load_image(self, i):
        """
        Loads an image by index, returning the image, its original dimensions, and resized dimensions.

        Returns (im, original hw, resized hw)
        """
        im, f, fn = (
            self.ims[i],
            self.im_files[i],
            self.npy_files[i],
        )
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f"Image Not Found {f}"
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images_to_disk(self, i):
        """Saves an image to disk as an *.npy file for quicker loading, identified by index `i`."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def load_mosaic(self, index):
        """Loads a 4-image mosaic for YOLOv5, combining 1 selected and 3 random images, with labels and segments."""
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        img4, labels4 = random_perspective(
            img4,
            labels4,
            segments4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img4, labels4

    def create_mosaic(self, images, labels):
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y

        for i, ((img, _, (h, w)), label) in enumerate(zip(images, labels)):
        # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            # labels, segments = self.labels[index].copy(), self.segments[index].copy()
            label, segments = label, []
            if label.size:
                label[:, 1:] = xywhn2xyxy(label[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(label)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        img4, labels4 = random_perspective(
            img4,
            labels4,
            segments4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img4, labels4


    def load_mosaic9(self, index):
        """Loads 1 image + 8 random images into a 9-image mosaic for augmented YOLOv5 training, returning labels and
        segments.
        """
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady :, x1 - padx :]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc : yc + 2 * s, xc : xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        img9, labels9, segments9 = copy_paste(img9, labels9, segments9, p=self.hyp["copy_paste"])
        img9, labels9 = random_perspective(
            img9,
            labels9,
            segments9,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img9, labels9

    @staticmethod
    def collate_fn(batch):
        """Batches images, labels, paths, and shapes, assigning unique indices to targets in merged label tensor."""
        imgs, labels, paths, shapes = batch
        for i, lb in enumerate(labels):
            lb[:, 0] = i  # add target image index for build_targets()
        labels = torch.cat(labels, 0)
        paths = tuple(paths)
        return imgs, labels, paths, shapes


    @staticmethod
    def collate_fn4(batch):
        """Bundles a batch's data by quartering the number of shapes and paths, preparing it for model input."""
        im, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im1 = F.interpolate(im[i].unsqueeze(0).float(), scale_factor=2.0, mode="bilinear", align_corners=False)[
                    0
                ].type(im[i].type())
                lb = label[i]
            else:
                im1 = torch.cat((torch.cat((im[i], im[i + 1]), 1), torch.cat((im[i + 2], im[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(im1)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4
    


def img2label_paths(img_paths):
    """
    """
    # breakpoint()
    label_paths = dict()
    for path in img_paths:
        tarpath, img_path = os.path.split(path)
        if tarpath not in label_paths.keys():
            label_paths[tarpath] = [os.path.splitext(img_path)[0] + '.txt']
        else:
            label_paths[tarpath].append(os.path.splitext(img_path)[0] + '.txt')
    return label_paths


def verify_image_label(args):
    """Verifies a single image-label pair, ensuring image format, size, and legal label values."""
    tmpdir, im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        # print(tmpdir, im_file, lb_file, prefix)
        with open(os.path.join(tmpdir, im_file), 'rb') as f:
            im = Image.open(f)
            im.verify()  # PIL verify
            shape = exif_size(im)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(os.path.join(tmpdir, im_file), "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                        msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # verify labels
        if os.path.isfile(os.path.join(tmpdir, lb_file)):
            nf = 1  # label found
            with open(os.path.join(tmpdir, lb_file)) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (lb[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, nm, nf, ne, nc, msg]


def exif_size(img):
    """Returns corrected PIL image size (width, height) considering EXIF orientation."""
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s



def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def seed_worker(worker_id):
    """
    Sets the seed for a dataloader worker to ensure reproducibility, based on PyTorch's randomness notes.

    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_hash(paths):
    """Generates a single SHA256 hash for a list of file or directory paths by combining their sizes and paths."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash
