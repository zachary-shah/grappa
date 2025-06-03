## GRAPPA in PyTorch

This library is a simple implementation of GRAPPA for cartesian MRI parallel reconstruction.

GRAPPA can be applied to basic cartesian sampling patterns in 2D or 3D, with the following requirements:
* Calibration region
* At least one fully sampled dimension
* Kernel sizes which are even for undersampled dimensions, or odd for fully sampled dimensions

#### Features:
* Functionality with limited GPU memory (automatic batching for low-memory cases)
* Auto-extraction of calibration regions from data
