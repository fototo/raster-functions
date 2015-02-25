__all__ = ['getTraceFunction']

def getTraceFunction():
    ctypes = __import__('ctypes')
    trace = ctypes.windll.kernel32.OutputDebugStringA
    trace.argtypes = [ctypes.c_char_p]
    return trace


def computeMapExtents(tlc, shape, props):
    nRows, nCols = shape if len(shape) == 2 else shape[1:]      # dimensions of request pixel block
    e, w, h = props['extent'], props['width'], props['height']  # dimensions of parent raster
    dX, dY = (e[2]-e[0])/w, (e[3]-e[1])/h                       # cell size of parent raster
    xMin, yMax = e[0]+tlc[0]*dX, e[3]-tlc[1]*dY                 # top-left corner of request on map
    return (xMin, yMax-nRows*dY, xMin+nCols*dX, yMax)           # extents of request on map
 

def isProductVersionOK(productInfo, major, minor, build): 
    v = productInfo['major']*1.e+10 + int(0.5+productInfo['minor']*10)*1.e+6 + productInfo['build']
    return v >= major*1e+10 + minor*1e+7 + build


class Projection():
    def __init__(self):
        pyprojModule = __import__('pyproj')
        self._inProj, self._outProj = None, None
        self._inEPSG, self._outEPSG = -1, -1

        self._projClass = getattr(pyprojModule, 'Proj')
        self._transformFunc = getattr(pyprojModule, 'transform')

    def transform(self, inEPSG, outEPSG, x, y):
        if inEPSG != self._inEPSG:
            self._inProj = self._projClass("+init=EPSG:{0}".format(inEPSG))
            self._inEPSG = inEPSG

        if outEPSG != self._outEPSG:
            self._outProj = self._projClass("+init=EPSG:{0}".format(outEPSG))
            self._outEPSG = outEPSG

        return self._transformFunc(self._inProj, self._outProj, x, y)
