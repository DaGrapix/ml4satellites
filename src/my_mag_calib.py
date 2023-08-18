import numpy as np
import matplotlib.pyplot as plt

"""
Code inspired by:
https://www.mirkosertic.de/blog/2023/01/magnetometer-calibration-ellipsoid/
Mirko Sertic Jan 12 2023, accessed Aug 1 2023
"""

def fit_ellipse(df):
    X = df["tw_magn_x"].values
    Y = df["tw_magn_y"].values

    # Formulate and solve the least squares problem ||Ax - b ||^2
    A = np.vstack([X**2, X * Y, Y**2, X, Y]).T

    b = -np.ones_like(X).reshape((1, len(X))).T
    x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()

    a = x[0]
    b = x[1]
    c = x[2]
    d = x[3]
    e = x[4]
    f = 1

    return(a,b,c,d,e,f)

def cart_to_pol(coeffs):
    """
    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.
    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

def correctdata(row, rotation, x0, y0, ap, bp):
        x = row["tw_magn_x"] - x0
        y = row["tw_magn_y"] - y0
        return [(x*np.cos(rotation) - y*np.sin(rotation))/bp,
                (x*np.sin(rotation) + y*np.cos(rotation))/ap]

def mag_calib(df):
    a, b, c, d, e, f        = fit_ellipse(df)
    x0, y0, ap, bp, e, phi  = cart_to_pol(coeffs=[a,b,c,d,e,f])

    rot      = round(phi/(np.pi/ 2.0))
    rotation = -(phi - rot*np.pi/2.0)

    res = df.apply(lambda row: correctdata(row, rotation, x0, y0, ap, bp), axis=1, result_type='expand')
    return(res)
