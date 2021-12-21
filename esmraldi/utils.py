import sys
from functools import reduce
from PyQt5.QtWidgets import QToolTip

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
