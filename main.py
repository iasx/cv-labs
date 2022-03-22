import cv2
import pylab as pl
from scipy.signal import convolve2d


def Sobel(A):
    # kernel for tracking horizontal changes
    Kx = pl.array([
        [-1, 0, +1],
        [-2, 0, +2],
        [-1, 0, +1],
    ])

    # kernel for tracking vertical changes
    Ky = pl.array([
        [+1, +2, +1],
        [0,  0,  0],
        [-1, -2, -1],
    ])

    # partial gradient magnitudes
    Gx = convolve2d(A, Kx, mode='valid')
    Gy = convolve2d(A, Ky, mode='valid')

    # full gradient magnitude
    G = pl.sqrt(pl.square(Gx) + pl.square(Gy))
    G *= 255.0 / G.max()

    return G


def Prewitt(A):
    # kernel for tracking horizontal changes
    Kx = pl.array([
        [+1, 0, -1],
        [+1, 0, -1],
        [+1, 0, -1],
    ])

    # kernel for tracking vertical changes
    Ky = pl.array([
        [+1, +1, +1],
        [0,  0,  0],
        [-1, -1, -1],
    ])

    # partial gradient magnitudes
    Gx = convolve2d(A, Kx, mode='valid')
    Gy = convolve2d(A, Ky, mode='valid')

    # full gradient magnitude
    G = pl.sqrt(pl.square(Gx) + pl.square(Gy))
    G *= 255.0 / G.max()

    return G


high = cv2.imread('img/high.png', 0)
low = cv2.imread('img/low.png', 0)
sharp = cv2.imread('img/sharp.png', 0)
soft = cv2.imread('img/soft.png', 0)

# Comparison: High Contrast
fig, (ax1, ax2) = pl.subplots(1, 2)
fig.suptitle('High Contrast')
ax1.imshow(Sobel(high), cmap='gray')
ax1.set_title('Sobel')
ax2.imshow(Prewitt(high), cmap='gray')
ax2.set_title('Prewitt')
pl.show()

# Comparison: Low Contrast
fig, (ax1, ax2) = pl.subplots(1, 2)
fig.suptitle('Low Contrast')
ax1.imshow(Sobel(low), cmap='gray')
ax1.set_title('Sobel')
ax2.imshow(Prewitt(low), cmap='gray')
ax2.set_title('Prewitt')
pl.show()

# Comparison: High Detalization
fig, (ax1, ax2) = pl.subplots(1, 2)
fig.suptitle('High Detalization')
ax1.imshow(Sobel(sharp), cmap='gray')
ax1.set_title('Sobel')
ax2.imshow(Prewitt(sharp), cmap='gray')
ax2.set_title('Prewitt')
pl.show()

# Comparison: Low Detalization
fig, (ax1, ax2) = pl.subplots(1, 2)
fig.suptitle('Low Detalization')
ax1.imshow(Sobel(soft), cmap='gray')
ax1.set_title('Sobel')
ax2.imshow(Prewitt(soft), cmap='gray')
ax2.set_title('Prewitt')
pl.show()
