import scipy
import scipy.ndimage as ndi

rc("image", cmap="gray", interpolation="bicubic")
figsize(10, 10)

import dlinputs as dli
import dltrainers as dlt
import dlinputs.filters as dlf
