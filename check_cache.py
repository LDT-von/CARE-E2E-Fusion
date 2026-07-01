# -*- coding: utf-8 -*-
import pickle, os
csv_path = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv'
cache_path = os.path.join(os.path.dirname(csv_path), 'tile_coords_cache.pkl')
with open(cache_path, 'rb') as f:
    cache = pickle.load(f)
lens = [len(v) for v in cache.values()]
import numpy as np
print('Cache: %d slides, %d total coords' % (len(cache), sum(lens)))
print('Tiles per slide: min=%d, max=%d, mean=%.0f, median=%d' % (
    min(lens), max(lens), np.mean(lens), int(np.median(lens))))
n_empty = sum(1 for v in cache.values() if len(v) == 0)
print('Empty slides: %d/%d' % (n_empty, len(cache)))
