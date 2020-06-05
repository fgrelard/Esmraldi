import shutil
import urllib.request as request
from contextlib import closing

def download(ftp, path_distant, path_local):
    with closing(request.urlopen(ftp + path_distant)) as r:
        with open(path_local, 'wb') as f:
            shutil.copyfileobj(r, f)

ftp = "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2014/11/PXD001283/"
imzml = "HR2MSI%20mouse%20urinary%20bladder%20S096.imzML"
ibd = "HR2MSI%20mouse%20urinary%20bladder%20S096.ibd"
optical = "HR2MSI%20mouse%20urinary%20bladder%20S096%20-%20optical%20image.tif"

print("Downloading imzML")
download(ftp, imzml, "ms_image.imzML")

print("Downloading ibd")
download(ftp, ibd, "ms_image.ibd")

print("Downloading optical image")
download(ftp, optical, "optical_image.tiff")

print("Done")
