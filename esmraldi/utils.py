import sys
import numpy as np
from functools import reduce
from PyQt5.QtWidgets import QToolTip
from numpy.lib.stride_tricks import as_strided
from scipy.stats import gmean

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def factors(n):
    return set(reduce(list.__add__,
                      ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def button_tooltip_on_hover(button):
    button.enterEvent = lambda event: QToolTip.showText(event.globalPos(), button.toolTip())
    # button.setToolTip("")
    return button


def msimage_for_visualization(msimage, transpose=True):
    msimage.is_maybe_densify = True
    msimage.spectral_axis = 0
    if transpose:
        new_order = (2, 1, 0)
        msimage = msimage.transpose(new_order)
    return msimage

def indices_search_sorted(current, target):
    n = len(target)
    indices = np.clip(np.searchsorted(target, current), 0, n-1)
    indices2 = np.clip(indices-1, 0, n-1)

    diff1 = target[indices] - current
    diff2 = current - target[indices2]

    indices = np.where(diff1 <= diff2, indices, indices2)
    return indices

def attempt_reshape(self, newdims, order='C'):
    is_f_order = False
    if order=="F":
        is_f_order = True
    newnd = len(newdims)
    newstrides = np.zeros(newnd).tolist()  # +1 is a fudge

    olddims = self.shape
    oldnd = self.ndim
    oldstrides = self.strides

    #/* oi to oj and ni to nj give the axis ranges currently worked with */

    oi,oj = 0,1
    ni,nj = 0,1
    while (ni < newnd) and (oi < oldnd):
        nep = newdims[ni];
        op = olddims[oi];
        while (nep != op):
            if (nep < op):
                # /* Misses trailing 1s, these are handled later */
                nep *= newdims[nj];
                nj += 1
            else:
                op *= olddims[oj];
                oj += 1

        #/* Check whether the original axes can be combined */
        for ok in range(oi, oj-1):
            if (is_f_order) :
                if (oldstrides[ok+1] != olddims[ok]*oldstrides[ok]):
                    # /* not contiguous enough */
                    return 0;
            else:
                #/* C order */
                if (oldstrides[ok] != olddims[ok+1]*oldstrides[ok+1]) :
                    #/* not contiguous enough */
                    return 0;
        # /* Calculate new strides for all axes currently worked with */
        if (is_f_order) :
            newstrides[ni] = oldstrides[oi];
            for nk in range(ni+1,nj):
                newstrides[nk] = newstrides[nk - 1]*newdims[nk - 1];
        else:
            #/* C order */
            newstrides[nj - 1] = oldstrides[oj - 1];
            #for (nk = nj - 1; nk > ni; nk--) {
            for nk in range(nj-1, ni, -1):
                newstrides[nk - 1] = newstrides[nk]*newdims[nk];
        ni = nj;nj += 1;
        oi = oj;oj += 1;

    # * Set strides corresponding to trailing 1s of the new shape.
    if (ni >= 1) :
        last_stride = newstrides[ni - 1];
    else :
        last_stride = self.itemsize # PyArray_ITEMSIZE(self);

    if (is_f_order) :
        last_stride *= newdims[ni - 1];

    for nk in range(ni, newnd):
        newstrides[nk] = last_stride;

    newarray = as_strided(self, shape=newdims, strides=newstrides)
    return newarray

def tolerance(mz, tolerance, is_ppm=True):
    if is_ppm:
        return mz * tolerance / 1e6
    else:
        return tolerance

def geomeans(intensities):
    if intensities.size == 0:
        return 0
    mask = np.ma.array(intensities, mask=intensities==0).min(0)
    masked_intensities = np.where(intensities==0, mask, intensities)
    geomean =  gmean(masked_intensities, axis=0)
    return geomean
