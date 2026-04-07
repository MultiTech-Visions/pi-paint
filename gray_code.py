"""Gray code structured light pattern generator.

Generates binary Gray code patterns for camera-projector calibration.
Each pattern encodes one bit of the projector's column or row index,
allowing the camera to determine which projector pixel illuminates
each camera pixel.
"""

import numpy as np


def _num_bits(resolution):
    """Number of Gray code bits needed to encode a given resolution."""
    if resolution <= 0:
        return 0
    return int(np.ceil(np.log2(resolution)))


def _to_gray(n):
    """Convert integer to Gray code."""
    return n ^ (n >> 1)


def generate_gray_patterns(width, height):
    """Generate horizontal and vertical Gray code pattern pairs.

    Returns a list of (pattern, inverse) tuples.  Each pattern is a
    uint8 numpy array of shape (height, width) where white=255 and
    black=0.  The inverse is the bitwise complement and is used to
    improve decoding robustness via thresholding.

    The list is ordered: all vertical-stripe patterns first (encoding
    the X / column axis), then all horizontal-stripe patterns (encoding
    the Y / row axis).
    """
    patterns = []

    # Vertical stripes → encode projector X (columns)
    n_bits_x = _num_bits(width)
    for bit in range(n_bits_x):
        pat = np.zeros((height, width), dtype=np.uint8)
        for col in range(width):
            if _to_gray(col) & (1 << (n_bits_x - 1 - bit)):
                pat[:, col] = 255
        patterns.append((pat, 255 - pat))

    # Horizontal stripes → encode projector Y (rows)
    n_bits_y = _num_bits(height)
    for bit in range(n_bits_y):
        pat = np.zeros((height, width), dtype=np.uint8)
        for row in range(height):
            if _to_gray(row) & (1 << (n_bits_y - 1 - bit)):
                pat[row, :] = 255
        patterns.append((pat, 255 - pat))

    return patterns


def decode_gray_captures(captures, width, height):
    """Decode captured Gray code images into a camera→projector map.

    Parameters
    ----------
    captures : list of (pos_image, neg_image)
        Each element is a pair of grayscale uint8 numpy arrays captured
        by the camera — one for the normal pattern and one for its
        inverse.  Order must match generate_gray_patterns().
    width, height : int
        Projector resolution used when generating the patterns.

    Returns
    -------
    proj_x : ndarray (cam_h, cam_w) float32
        For each camera pixel, the decoded projector X coordinate,
        or -1 if the pixel could not be decoded.
    proj_y : ndarray (cam_h, cam_w) float32
        For each camera pixel, the decoded projector Y coordinate,
        or -1 if the pixel could not be decoded.
    confidence : ndarray (cam_h, cam_w) float32
        Per-pixel confidence in [0, 1] based on pattern contrast.
    """
    n_bits_x = _num_bits(width)
    n_bits_y = _num_bits(height)
    cam_h, cam_w = captures[0][0].shape[:2]

    # Decode each bit by comparing positive and negative captures
    bits = []
    contrast = np.zeros((cam_h, cam_w), dtype=np.float32)

    for pos_img, neg_img in captures:
        pos = pos_img.astype(np.float32)
        neg = neg_img.astype(np.float32)
        bit_mask = (pos > neg).astype(np.uint8)
        bits.append(bit_mask)
        contrast += np.abs(pos - neg)

    # Average contrast across all patterns as confidence
    total_patterns = len(captures)
    confidence = contrast / (total_patterns * 255.0)

    # Reconstruct Gray code values, then convert to binary
    def gray_to_binary(gray_val, n_bits):
        """Convert Gray code integer to binary integer."""
        mask = gray_val
        while mask > 0:
            mask >>= 1
            gray_val ^= mask
        return gray_val

    gray_to_bin_vec = np.vectorize(gray_to_binary)

    # Decode X (first n_bits_x patterns)
    gray_x = np.zeros((cam_h, cam_w), dtype=np.int32)
    for i in range(n_bits_x):
        gray_x |= bits[i].astype(np.int32) << (n_bits_x - 1 - i)
    proj_x = gray_to_bin_vec(gray_x, n_bits_x).astype(np.float32)

    # Decode Y (next n_bits_y patterns)
    gray_y = np.zeros((cam_h, cam_w), dtype=np.int32)
    for i in range(n_bits_y):
        idx = n_bits_x + i
        gray_y |= bits[idx].astype(np.int32) << (n_bits_y - 1 - i)
    proj_y = gray_to_bin_vec(gray_y, n_bits_y).astype(np.float32)

    # Mask out low-confidence pixels
    low_conf = confidence < 0.1
    proj_x[low_conf] = -1
    proj_y[low_conf] = -1

    # Clamp out-of-range values
    proj_x[(proj_x >= width) & (proj_x >= 0)] = -1
    proj_y[(proj_y >= height) & (proj_y >= 0)] = -1

    return proj_x, proj_y, confidence
