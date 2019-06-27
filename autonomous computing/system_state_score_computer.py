''' 
Apple Inc. 2019 , All rights reserved

DESCRIPTION:
-------------------------------------------------------
This file looks at the benchmark data and the predicted
data. It then compares the two graphs by doing a 
Discrete Fourier Transform. The two DFTs are then compared
through normalization and a single value between [-1,+1]
is generated. -1 means that the system is much cooler than the
benchmark (desired) and +1 means system is much hoter than 
the benchmark. 
'''
import math
import numpy as np
import ac_utils

############################################
# Compute the DFT of the incoming signal
# Params:
#   1. signal - An array containing the sampled
#               data
# Returns: An array containing the amplitudes
#          for each of the resulting frequency
#          components
############################################
def computeDFTAmplitude(signal):
    fourier = np.fft.fft(signal)
    
    amplitudes = np.zeros(shape=fourier.size)

    idx=0
    for fk in fourier:
        # We now get individual elements of the array. 
        # Each element is a complex number
        reFK = fk.real #the real part
        imFK = fk.imag #the imaginary part

        # We compute the amplitude of each element
        amplitudes[idx] = math.sqrt(reFK**2 + imFK**2)
        idx +=1
    return amplitudes

##########################################
# Assuming the amplitudes represent DC signal
# (the zeroth element) and AC signal amplitudes
# (the other entries) , we compute the equivalent
# power delivered (like in a power signal, replace
# AC with equivalent DC signal)
# Param:
#   1. amp - An array containing the amplitudes
# Returns : A float
##########################################
def computeEquivalentPower(amp):
    #
    # normalization is done by adding the first element
    # with all other element's RMS value
    # 
    norm = 0
    if amp.size < 1:
        return 0
    elif amp.size == 1:
        return amp[0]
    else:
        norm =amp[0]
        for idx in range(1,amp.size):
            norm += (amp[idx]/math.sqrt(2))
    return norm

##########################################
# Compute the normalized system state score
# for the given signal, based on historical
# data. Both input signal and historical data
# must be of the same size.
# Param:
#   1. signal - An array containing the samples whose
#               state score is to be computed
#   2. history- The historical data to be compared
#               against.
# Returns : A float between -1,+1. -100 on errors
##########################################
def computeNormalizedStateScore(signal,history):
    if(signal.size != history.size) :
        return -100
    
    rawScoreSignal = computeEquivalentPower( computeDFTAmplitude(signal) )
    rawScoreHistory= computeEquivalentPower( computeDFTAmplitude(history) )

    normalizedScore = -( rawScoreHistory - rawScoreSignal ) / rawScoreHistory
    if(normalizedScore > 1):
        normalizedScore = 1
    
    return normalizedScore
