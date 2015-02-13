from scipy import ndimage
import numpy as np
import math 
from Hillshade import Hillshade


class MultidirectionalHillshade():

    def __init__(self):
        self.name = "Multidirectional Hillshade Function"
        self.description = ("This function computes a hillshade surface from six different directions. "
                            "The result is a stunning visualization in both high slope and expressionless areas.")

        self.H = None
        self.azimuths   = (315.0, 270.0, 225.0, 360.0, 180.0,   0.0)
        self.elevations = ( 60.0,  60.0, 60.0,   60.0,  60.0,  90.0)
        self.weights    = (0.167, 0.278, 0.167, 0.111, 0.056, 0.222)
        self.factors    = ()


    def getParameterInfo(self):
        return [
            {
                'name': 'raster',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Input Raster",
                'description': "The primary input raster where pixel values represent elevation.",
            },
            {
                'name': 'zf',
                'dataType': 'numeric',
                'value': 1.0,
                'required': False,
                'displayName': "Z Factor",
                'description': "The multiplicative factor that converts elevation values to the units of the horizontal (xy-) coordinate system.",
            },
        ]


    def getConfiguration(self, **scalars): 
        return {
          'extractBands': (0,),                 # we only need the first band.  Comma after zero ensures it's a tuple.
          'inheritProperties': 4 | 8,           # inherit everything but the pixel type (1) and NoData (2)
          'invalidateProperties': 2 | 4 | 8,    # invalidate these aspects because we are modifying pixel values and updating key properties.
          'padding': 1,                         # one extra on each each of the input pixel block
          'inputMask': True                     # we need the input mask in .updatePixels()
        }


    def updateRasterInfo(self, **kwargs):
        zf = kwargs.get('zf', 1.0)

        kwargs['output_info']['bandCount'] = 1
        kwargs['output_info']['pixelType'] = 'u1'
        kwargs['output_info']['statistics'] = ({'minimum': 0.0, 'maximum': 255.0}, )
        kwargs['output_info']['histogram'] = ()
        kwargs['output_info']['colormap'] = ()

        e = kwargs['raster_info']
        if e['bandCount'] > 1: 
            raise Exception("Input raster must have a single band.")

        self.H = []
        for i in range(len(self.azimuths)):
            self.H.append(Hillshade())
            self.H[i].prepare(azimuth=self.azimuths[i], elevation=self.elevations[i], 
                              zFactor=zf, cellSize=e['cellSize'], sr=e['spatialReference'])
        return kwargs


    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        v = np.array(pixelBlocks['raster_pixels'], dtype='f4', copy=False)

        dx, dy = self.H[0].computeGradients(v)      # gradients depend only on the z-factor which doesn't vary across H's
        outBlock = self.weights[0] * self.H[0].computeHillshade(dx, dy)
        for i in range(1, 6):                       # 6 == len(self.azimuths)
            outBlock += (self.weights[i] * self.H[i].computeHillshade(dx, dy))
        
        pixelBlocks['output_pixels'] = outBlock[1:-1, 1:-1].astype(props['pixelType'], copy=False)  # undo padding

        m = np.array(pixelBlocks['raster_mask'], dtype='u1', copy=False)
        pixelBlocks['output_mask'] = m[:-2,:-2]  & m[1:-1,:-2]  & m[2:,:-2]  \
                                   & m[:-2,1:-1] & m[1:-1,1:-1] & m[2:,1:-1] \
                                   & m[:-2,2:]   & m[1:-1,2:]   & m[2:,2:]
        return pixelBlocks


    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        if bandIndex == -1:                             # dataset-level properties           
            keyMetadata['datatype'] = 'Processed'       # outgoing dataset is now 'Processed'
        elif bandIndex == 0:                            # properties for the first band
            keyMetadata['wavelengthmin'] = None         # reset inapplicable band-specific key metadata 
            keyMetadata['wavelengthmax'] = None
            keyMetadata['bandname'] = 'Hillshade'
        return keyMetadata



"""
References:

    [1]. Nagi, R. (2014), Introducing Esri's Next Generation Hillshade
    http://blogs.esri.com/esri/arcgis/2014/07/14/introducing-esris-next-generation-hillshade/

"""

