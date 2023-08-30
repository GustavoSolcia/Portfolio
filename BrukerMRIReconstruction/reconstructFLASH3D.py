# Author: Gustavo Solcia
# E-mail: gustavo.solcia@usp.br

"""Read Bruker MRI raw data from Paravision v5.1 of FLASH 3D images and save magnitude and phase images in NIFTI format. This is a simplified version inspired on Bernd U. Foerster reconstruction (https://github.com/bfoe/BrukerOfflineReco).

"""

import os
import gc
import sys
import numpy as np
import nibabel as nib

def readRAW(path, inputName):
    """Function to read RAW data from Paravision v5.1

    Parameters
    ----------
    inputPath: string
        Path from raw files directories.

    Returns
    -------
    rawComplexData: complex array
        Unprocessed raw data in complex notation from fid directory.

    """
    
    with open(path+inputName, 'rb') as dataFile:
        rawData = np.fromfile(dataFile, dtype=np.int32)
    
    rawComplexData = rawData[0::2] + 1j*rawData[1::2]

    return rawComplexData

def readParameters(path):
    """Function to read either basic scan parameters or base level acquisition parameters from method or acqp files.

    Parameters
    ----------
    inputPath: string
        Path from parameter files directories (method or acqp).

    Returns
    -------
    parameterDict: dict
        Parameter dictionary from method or acqp files.

    """

    parameterDict = {}

    with open(path, 'r') as parameterFile:
        while True:

            line = parameterFile.readline()

            if not line:
                break

            # '$$ /' indicates when line contains original file name
            if line.startswith('$$ /'):
                originalFileName = line[line.find('/nmr/')+5:]
                originalFileName = originalFileName[0:len(originalFileName)-8]
                originalFileName = originalFileName.replace(".", "_")
                originalFileName = originalFileName.replace("/", "_")

            # '##$' indicates when line contains parameter
            if line.startswith('##$'):
                parameterName, currentLine = line[3:].split('=')

                # checks if entry is arraysize
                if currentLine[0:2] == "( " and currentLine[-3:-1] == " )":
                    parameterValue = parseArray(parameterFile, currentLine) 

                # checks if entry is struct/list
                elif currentLine[0:2] == "( " and currentLine[-3:-1] != " )":
                    while currentLine[-2] != ")": #in case of multiple lines
                        currentLine = currentLine[0:-1] + parameterFile.readline()

                    parameterValue = [parseSingleValue(lineValue) 
                            for lineValue in currentLine[1:-2].split(', ')]

                # last option is single string or number
                else:
                    parameterValue = parseSingleValue(currentLine)

                parameterDict[parameterName] = parameterValue

    return originalFileName, parameterDict

def parseArray(parameterFile, line):
    """Parse array type from readParameters function.

    Parameters
    ----------
    parameterFile: string
        Path from parameter files directories (method or acqp).
    line: string
        Current line from file read in readParameters funciton.

    Returns
    -------
    valueList: string, int, or float
        Parsed values from input line array.
    """

    line = line[1:-2].replace(" ", "").split(",")
    arraySize = np.array([int(arrayValue) for arrayValue in line])

    valueList = parameterFile.readline().split()
    
    # If the value cannot be converted to float then it is a string
    try:
        float(valueList[0])
    except ValueError:
        return " ".join(valueList)

    while len(valueList) != np.prod(arraySize): # in case of multiple lines
        valueList = valueList + parameterFile.readline().split()

    # If the value is not an int then it is a float
    try:
        valueList = [int(singleValue) for singleValue in valueList]
    except ValueError:
        valueList = [float(singleValue) for singleValue in valueList]

    # transform list to numpy array
    if len(valueList) > 1:
        return np.reshape(np.array(valueList), arraySize)
    else:
        return valueList[0]

def parseSingleValue(singleValue):
    """Parse single value from readParameters function.

    Parameters
    ----------
    singleValue: int, float, or string
        Single value from readParameters function.

    Returns
    -------
    singleValueParsed: int, float, or string
        Parsed value from input.

    """
    
    #if it is not int then it is a float or string
    try:
        singleValueParsed = int(singleValue)
    except ValueError:
        #if it is not a float then it is a string
        try:
            singleValueParsed = float(singleValue)
        except ValueError:
            singleValueParsed = singleValue.rstrip('\n')
    
    return singleValueParsed

def checkDataImplementation(methodData):
    """Check for unexpected and not implemented data.

    Parameters
    ----------
    methodData: dict
        Parameter dictionary from method file.

    """
    if  not(methodData["Method"] == "FLASH" or methodData["Method"] == "FISP" or methodData["Method"] =="GEFC") or methodData["PVM_SpatDimEnum"] != "3D":
        print ('ERROR: Recon only implemented for FLASH/FISP 3D method');
        sys.exit(1)
    if methodData["PVM_NSPacks"] != 1:
        print ('ERROR: Recon only implemented 1 package');
        sys.exit(1)
    if methodData["PVM_NRepetitions"] != 1:
        print ('ERROR: Recon only implemented 1 repetition');
        sys.exit(1)
    if methodData["PVM_EncPpiAccel1"] != 1 or methodData["PVM_EncNReceivers"] != 1 or\
        methodData["PVM_EncZfAccel1"] != 1 or methodData["PVM_EncZfAccel2"] != 1:
        print ('ERROR: Recon for parallel acquisition not implemented');
        sys.exit(1)

def prepareData(rawComplexData, methodData):
    """Prepare raw data with a series of processing functions.

    Parameters
    ----------
    rawComplexData: complex array
        RAW data from Bruker files.
    methodData: dict
        Parameter dictionary from method file.
    """

    dim = methodData["PVM_EncMatrix"]
    EncPftAccel1 = methodData["PVM_EncPftAccel1"]
    EncSteps1 = methodData["PVM_EncSteps1"]
    EncSteps2 = methodData["PVM_EncSteps2"]
    SPackArrPhase1Offset = methodData["PVM_SPackArrPhase1Offset"]
    SPackArrSliceOffset = methodData["PVM_SPackArrSliceOffset"]
    Fov = methodData["PVM_Fov"]
    AntiAlias = methodData["PVM_AntiAlias"]
    SpatResol = methodData["PVM_SpatResol"]

    reshapedData = reshapeData(rawComplexData, dim)

    if EncPftAccel1 != 1:
        zerosData, dim = addZerosPartialPhaseAcq(reshapedData, EncPftAccel1, dim)
    else:
        zerosData = reshapedData
    
    reorderedData = reorderData(zerosData, dim, EncSteps1, EncSteps2)
    offsetData = applyFOVoffset(reorderedData,SPackArrPhase1Offset, SPackArrSliceOffset,Fov,AntiAlias)
    zeroFillData, dim, SpatResol, zero_fill = applyZeroFill(offsetData, dim, SpatResol)
    hanningData = applyHanningFilter(zeroFillData, dim, zero_fill)

    preparedComplexData = hanningData
    return preparedComplexData, SpatResol

def reshapeData(rawComplexData, dim):
    """Reshape raw complex data to dimensions from method data.

    Parameters
    ----------
    rawComplexData: complex array
        Unprocessed raw data in complex notation from fid directory.
    dim: array
        Dimensions from methods data.

    Returns
    -------
    reshapedData: complex array
        Reshaped array with method dimensions.
    """

    dim0 = dim[0]
    dim0_mod_128 = dim0%120

    if dim0_mod_128 != 0: #Bruker sets readout point to a multiple of 128
        dim0 = (int(dim0/128+1))*128

    try: # order="F" parameter for Fortran style order as by Bruker conventions
        rawComplexData = rawComplexData.reshape(dim0, dim[1], dim[2], order='F')
    except:
        print('ERROR: k-space data reshape failed (dimension problem)')
        sys.exit(1)

    if dim0 != dim[0]:
        reshapedData = rawComplexData[0:dim[0], :, :]
    else:
        reshapedData = rawComplexData

    return reshapedData

def addZerosPartialPhaseAcq(reshapedData, EncPftAccel1, dim):
    """Add zeros in case of partial phase acquisition.

    Parameters
    ----------
    reshapedData: complex array
    EncPftAccel1: int
    dim: array
        Image dimensions.

    Returns
    -------
    zerosData: array
        Data with zeros along partial acquisition.
    newDim: array
        New image dimension.
    """
    zeros_ = np.zeros(shape=(dim[0],int(dim[1]*(float(EncPftAccel)-1.)),dim[2]))
    zerosData = np.append(reshapedData, zeros_, axis=1)
    newDim = zerosData.shape

    return zerosData, newDim

def reorderData(zerosData, dim, EncSteps1, EncSteps2):
    """Reorder data depending on order of encoding direction 1 and 2.
    
    Parameters
    ----------
    zerosData: complex array
        Data from partial phase acquition step or just with reshaping.
    dim: array
        Data dimension.
    EncSteps1: array
        Encoding steps in direction 1.
    EncSteps2: array
        Encoding steps in direction 2.

    Returns
    -------
    reorderedData: complex array
        Data reordered with encoding steps order.
    """

    FIDdata_tmp=np.empty(shape=(dim[0],dim[1],dim[2]),dtype=np.complex64)
    reorderedData=np.empty(shape=(dim[0],dim[1],dim[2]),dtype=np.complex64)

    orderEncDir1= EncSteps1+dim[1]/2
    for i in range(0,orderEncDir1.shape[0]): 
        FIDdata_tmp[:,int(orderEncDir1[i]),:]=zerosData[:,i,:]
    
    orderEncDir2=EncSteps2+dim[2]/2
    for i in range(0,orderEncDir2.shape[0]): 
        reorderedData[:,:,int(orderEncDir2[i])]=FIDdata_tmp[:,:,i]

    return reorderedData

def applyFOVoffset(reorderedData, SPackArrPhase1Offset, SPackArrSliceOffset,Fov,AntiAlias):
    """Fov offset with anti aliasing.

    Parameters
    ----------
    reorderedData: complex array
        k-space data after reordering processing.
    SPackArrPhase1Offset: float
        Phase offset from method file.
    SPackArrSliceOffset: float
        Slice offset from method file.
    Fov: list
        FOV dimensions from method file.
    AntiAlias: list
        Anti alias from method file.

    Returns
    -------
    offsetData: complex array
        Data with offset correction.
    """
    realFOV = Fov*AntiAlias

    phase_step1 = +2.*np.pi*float(SPackArrPhase1Offset)/float(realFOV[1])
    phase_step2 = -2.*np.pi*float(SPackArrSliceOffset)/float(realFOV[2])

    mag = np.abs(reorderedData[:,:,:])
    ph = np.angle(reorderedData[:,:,:])

    for i in range(0,reorderedData.shape[1]): 
        ph[:,i,:] -= float(i-int(reorderedData.shape[1]/2))*phase_step1
    for j in range(0,reorderedData.shape[2]): 
        ph[:,:,j] -= float(j-int(reorderedData.shape[2]/2))*phase_step2

    offsetData = mag * np.exp(1j*ph)

    return offsetData

def applyZeroFill(offsetData, dim, SpatResol):
    """Zero filling in k-space to reconstruct the data with increased resolution and interpolation with correct aspect ratio.

    Parameters
    ----------
    offsetData: complex array
        k-space data after FOV offset processing.
    dim: array
        Image dimension after processing steps.
    SpatResol: array
    

    Returns
    -------
    zeroFillData: complex array
        k-space data with zero filling with zero_fill proportions (size_of_dim/zero_fill)
    dim: array
        Data dimension after processing steps.
    SpatResol: array
        Image spatial resolution after processing steps.
    zero_fill: int
        Proportion of image dimensions for zero filling (default is 2 so the zero filling will be applied with half of dimension in each direction)

    """
    zero_fill=2
    SpatResol=SpatResol/zero_fill

    dim0Padding=int(dim[0]/zero_fill)
    dim1Padding=int(dim[1]/zero_fill)
    dim2Padding=int(dim[2]/zero_fill)


    zeroFillData = np.pad(offsetData, [(dim0Padding,dim0Padding), (dim1Padding,dim1Padding), (dim2Padding,dim2Padding)], mode='constant')
    dim=zeroFillData.shape

    return zeroFillData, dim, SpatResol, zero_fill

def applyHanningFilter(data, dim, zero_fill):
    """Hanning windowing for k-space data smoothing.

    Parameters
    ----------
    data: complex array
    dim: array
        Image dimension after processing steps.
    zero_fill:
        Zero fill order to reconstruct filter size in original image dimension.

    Returns
    -------
    hanningData: complex array
        Filtered k-space.
    """
    percentage = 10

    nz = np.asarray(np.nonzero(data))
    first_x=np.amin(nz[0,:]); last_x=np.amax(nz[0,:])
    first_y=np.amin(nz[1,:]); last_y=np.amax(nz[1,:])
    first_z=np.amin(nz[2,:]); last_z=np.amax(nz[2,:])

    npoints_x = int(float(dim[0]/zero_fill)*percentage/100.)
    npoints_y = int(float(dim[1]/zero_fill)*percentage/100.)
    npoints_z = int(float(dim[2]/zero_fill)*percentage/100.)

    hanning_x = np.zeros(shape=(dim[0]),dtype=np.float32)
    hanning_y = np.zeros(shape=(dim[1]),dtype=np.float32)
    hanning_z = np.zeros(shape=(dim[2]),dtype=np.float32)

    x_ = np.linspace (1./(npoints_x-1.)*np.pi/2.,(1.-1./(npoints_x-1))*np.pi/2.,num=npoints_x)
    hanning_x [first_x:first_x+npoints_x] = np.power(np.sin(x_),2)
    hanning_x [first_x+npoints_x:last_x-npoints_x+1] = 1
    x_ = x_[::-1]
    hanning_x[last_x-npoints_x+1:last_x+1] = np.power(np.sin(x_),2)

    y_ = np.linspace (1./(npoints_y-1.)*np.pi/2.,(1.-1./(npoints_y-1))*np.pi/2.,num=npoints_y)
    hanning_y [first_y:first_y+npoints_y] = np.power(np.sin(y_),2)
    hanning_y [first_y+npoints_y:last_y-npoints_y+1] = 1
    y_ = y_[::-1]
    hanning_y[last_y-npoints_y+1:last_y+1] = np.power(np.sin(y_),2)

    z_ = np.linspace (1./(npoints_z-1.)*np.pi/2.,(1.-1./(npoints_z-1))*np.pi/2.,num=npoints_z)
    hanning_z [first_z:first_z+npoints_z] = np.power(np.sin(z_),2)
    hanning_z [first_z+npoints_z:last_z-npoints_z+1] = 1
    z_ = z_[::-1]
    hanning_z[last_z-npoints_z+1:last_z+1] = np.power(np.sin(z_),2)

    hanningData = data
    hanningData *= hanning_x[:, None, None]
    hanningData *= hanning_y[None, :, None]
    hanningData *= hanning_z[None, None, :]
    return hanningData

def applyFFT(data):
    """Function to apply fft in 3D k-space.

    Parameters
    ----------
    data: complex array
        3D k-space from RAW data files.

    Returns
    -------
    transfData: complex array
        Transformed data from frequency to spatial domain.
    """

    shiftedData = np.fft.fftshift(data, axes=(0,1,2))
    transfData = shiftedData
    for k in range(0,data.shape[1]):
        transfData[:,k,:] = np.fft.fft(shiftedData[:,k,:], axis=(0))
    for i in range(0,data.shape[0]):
        transfData[i,:,:] = np.fft.fft(transfData[i,:,:], axis=(0))
    for i in range(0,data.shape[0]):
        transfData[i,:,:] = np.fft.fft(transfData[i,:,:], axis=(1))
    transfData = np.fft.fftshift(transfData, axes=(0,1,2))
    return transfData

def calculateMagnitude(spatialDomainData, acqpData, methodData):
    """Calculates magnitude image from processed spatial domain data.

    Parameters
    ----------
    spatialDomainData: complex array
        Processed data in spatial domain (post fft).
    acqpData: dict
        Parameter dictionary from acqp file.
    methodData: dict
        Parameter dictionary from method file.

    Returns
    -------
    magnitudeData: array
        Magnitude image ready to be saved in NIFTI format.
    """
    ReceiverGain = acqpData["RG"] # RG is a simple attenuation FACTOR, NOT in dezibel (dB) unit 
    n_Averages = methodData["PVM_NAverages"]

    magnitudeData = np.abs(spatialDomainData)/RG_to_voltage(ReceiverGain)/n_Averages; 
    max_ABS = np.amax(magnitudeData);
    magnitudeData *= 32767./max_ABS
    magnitudeData = magnitudeData.astype(np.int16)

    return magnitudeData

def calculatePhase(spatialDomainData):
    """Calculates phase image from processed spatial domain data.

    Parameters
    ----------
    spatialDomainData: complex array
        Processed data in spatial domain (post fft).

    Returns
    -------
    phaseData: array
        Phase image ready to be saved in NIFTI format.
    """

    phaseData = np.angle(spatialDomainData)
    max_PH = np.pi;
    phaseData *= 32767./max_PH
    phaseData = phaseData.astype(np.int16)
    phaseData [0,0,0] = 32767 
    phaseData [1,1,1] = -32767

    return phaseData 

def RG_to_voltage(RG):
    """Convert receiver gain to voltage.

    This comes from the Bruker provided conversion list below
    
     Receiver,   Gain Equivalent Voltage Gain [dB],        V_out/V_in
                 20*log (V_out/V_in)
    
     2050,       78,                                       7943.282347
     1820,       77,                                       7079.457844
     1620,       76,                                       6309.573445
     1440,       75,                                       5623.413252
     1290,       74,                                       5011.872336
     1150,       73,                                       4466.835922
     1030,       72,                                       3981.071706
     912,        71,                                       3548.133892
     812,        70,                                       3162.27766
     724,        69,                                       2818.382931
     645,        68,                                       2511.886432
     575,        67,                                       2238.721139
     512,        66,                                       1995.262315
     456,        65,                                       1778.27941
     406,        64,                                       1584.893192
     362,        63,                                       1412.537545
     322,        62,                                       1258.925412
     287,        61,                                       1122.018454
     256,        60,                                       1000
     228,        59,                                       891.2509381
     203,        58,                                       794.3282347
     181,        57,                                       707.9457844
     161,        56,                                       630.9573445
     144,        55,                                       562.3413252
     128,        54,                                       501.1872336
     114,        53,                                       446.6835922
     101,        52,                                       398.1071706
     90.5,       51,                                       354.8133892
     80.6,       50,                                       316.227766
     71.8,       49,                                       281.8382931
     64,         48,                                       251.1886432
     57,         47,                                       223.8721139
     50.8,       46,                                       199.5262315
     45.2,       45,                                       177.827941
     40.3,       44,                                       158.4893192
     36,         43,                                       141.2537545
     32,         42,                                       125.8925412
     28.5,       41,                                       112.2018454
     25.4,       40,                                       100
     22.6,       39,                                       89.12509381
     20.2,       38,                                       79.43282347
     18,         37,                                       70.79457844 
     16,         36,                                       63.09573445 
     14.2,       35,                                       56.23413252 
     12.7,       34,                                       50.11872336 
     11.3,       33,                                       44.66835922 
     10,         32,                                       39.81071706 
     9,          31,                                       35.48133892 
     8,          30,                                       31.6227766  
     7.12,       29,                                       28.18382931 
     6.35,       28,                                       25.11886432 
     5.6,        27,                                       22.38721139 
     5,          26,                                       19.95262315 
     4.5,        25,                                       17.7827941  
     4,          24,                                       15.84893192 
     3.56,       23,                                       14.12537545 
     3.2,        22,                                       12.58925412 
     2.8,        21,                                       11.22018454 
     2.56,       20,                                       10          
     2.25,       19,                                       8.912509381 
     2,          18,                                       7.943282347 
     1.78,       17,                                       7.079457844 
     1.6,        16,                                       6.309573445 
     1.4,        15,                                       5.623413252 
     1.28,       14,                                       5.011872336 
     1.12,       13,                                       4.466835922 
     1,          12,                                       3.981071706 
     0.89,       11,                                       3.548133892 
     0.8,        10,                                       3.16227766  
     0.7,        9,                                        2.818382931 
     0.64,       8,                                        2.511886432 
     0.56,       7,                                        2.238721139 
     0.5,        6,                                        1.995262315 
     0.44,       5,                                        1.77827941  
     0.4,        4,                                        1.584893192 
     0.35,       3,                                        1.412537545 
     0.32,       2,                                        1.258925412 
     0.28,       1,                                        1.122018454 
     0.25,       0,                                        1

    Parameters
    ----------
    RG: float
        Receiver gain value.

    Returns
    -------
    Voltage: float
        Equivalent voltage converted from Bruker conversion list.

    """
    
    Voltage = np.power(10,11.995/20.) * np.power(RG,19.936/20.)
    return Voltage

def saveNIFTI(path, outputName, originalFileName, suffix, data, SpatResol):
    """Creates header with method and acquisition files, then saves image in NIFTI format.
    
    Parameters
    ----------
    path: str
        Output file path.
    outputName: str
        Usually contains the type of acquisiton: FLASH, PSIF, ...
    originalFileName: str
        File name created during patient registration in Paravision 5.1.
    suffix: str
        Could be either '_MAGNT' or '_PHASE' depending if you are saving a magnitude or phase image.
    data: array
        Image data to be saved.
    SpatResol: array
        Image spatial resolution after processing steps. 

    """
    affineMatrix = np.eye(4)

    affineMatrix[0,0] = SpatResol[0]*1000
    affineMatrix[0,3] = -(data.shape[0]/2)*affineMatrix[0,0]
    affineMatrix[1,1] = SpatResol[1]*1000
    affineMatrix[1,3] = -(data.shape[1]/2)*affineMatrix[1,1]
    affineMatrix[2,2] = SpatResol[2]*1000
    affineMatrix[2,3] = -(data.shape[2]/2)*affineMatrix[2,2]

    NIFTIimg = nib.Nifti1Image(data, affineMatrix)
    NIFTIimg.header.set_slope_inter(np.amax(data)/32767.,0)
    NIFTIimg.header.set_xyzt_units(3, 8)
    NIFTIimg.set_sform(affineMatrix, code=0)
    NIFTIimg.set_qform(affineMatrix, code=1)

    nib.save(NIFTIimg, path+outputName+originalFileName+suffix+'.nii.gz')


if __name__ == '__main__':

    inputPath = '/home/solcia/Documents/phd/MRI data/Coral/13'
    outputPath = '/home/solcia/Documents/phd/MRI data/Coral'

    inputName = '/fid'
    outputName = '/FLASH3D_'

    _,acqpData = readParameters(inputPath+'/acqp') #acqp stands for acquisition parameters
    originalFileName, methodData = readParameters(inputPath+'/method') # methods contains basic scan parameters

    checkDataImplementation(methodData)

    rawComplexData = readRAW(inputPath, inputName)
    
    preparedComplexData, SpatResol = prepareData(rawComplexData, methodData)
   
    spatialDomainData = applyFFT(preparedComplexData)

    magnitudeData = calculateMagnitude(spatialDomainData, acqpData, methodData)
    
    phaseData = calculatePhase(spatialDomainData)

    saveNIFTI(outputPath, outputName, originalFileName, '_MAGNT', magnitudeData, SpatResol)
    saveNIFTI(outputPath, outputName, originalFileName, '_PHASE', phaseData, SpatResol)
