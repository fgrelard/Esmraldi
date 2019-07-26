import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

register = sitk.ReadImage(registername, sitk.sitkFloat32)
register.SetDirection( (1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0))
size = register.GetSize()
