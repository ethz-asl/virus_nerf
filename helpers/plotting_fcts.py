import numpy as np
import matplotlib
import scipy

def combineImgs(
    bool_imgs:list,
    colors:list,
    upsample:int=1,
):
    """
    Combines a list of boolean images into one image
    and colors the pixels according to the colors list.
    Args:
        bool_imgs: list of boolean images; list of numpy arrays
        colors: list of colors; list of strings
        upsample: upsampling factor; int
    Returns:
        rgb_img: combined image; numpy array of shape (height, width, 4)
    """
    rgb_img = np.zeros((bool_imgs[0].shape[0], bool_imgs[0].shape[1], 4), dtype=float)
    for i, img in enumerate(bool_imgs):
        # check if img is boolean or can be converted to boolean
        if img.dtype != bool:
            if not np.all(np.isin(img, [0, 1])):
                print(f"plotting_fcts.combineImgs: img {i} is not boolean and cannot be converted to boolean")
            img = img.astype(bool)

        rgb_img[img] = matplotlib.colors.to_rgba(colors[i])

    rgb_img = (255 * rgb_img).astype(np.uint8)

    if upsample > 1:
        rgb_img = scipy.ndimage.zoom(rgb_img, (upsample,upsample,1), order=0)
    
    return rgb_img