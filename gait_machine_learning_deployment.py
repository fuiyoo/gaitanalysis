import numpy as np
import pickle
import streamlit as st
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import colors
import pywt
import nolds

# Add a title to the Streamlit application
st.title("Machine Learning Deployment")

loaded_model = pickle.load(open('STFT+CWT+DFAsvm_model.sav','rb'))

# Create a file uploader for the Excel file
uploaded_file = st.file_uploader("Choose an Excel file", type='xlsx')

def calcSTFT_norm(stride_interval, samplingFreq, window='hann', nperseg=64, noverlap=None, figsize=(9,5), cmap='magma', ylim_max=None):
    ##Calculating STFT
    f, t, Zxx = signal.stft(stride_interval, samplingFreq, window=window, nperseg=nperseg, noverlap=noverlap)
    Zxx_abs_squared = np.abs(Zxx)**2  # Square magnitude of STFT
    RMS = np.sqrt(np.mean(Zxx_abs_squared, axis=0))  # Calculate RMS along the frequency axis
    max_RMS = np.max(RMS)
    ##Plotting STFT
    fig = plt.figure(figsize=figsize)
    spec = plt.pcolormesh(t, f, np.abs(Zxx), 
                          norm=colors.PowerNorm(gamma=1./8.),
                          cmap=plt.get_cmap(cmap))
    cbar = plt.colorbar(spec)  # Set format for colorbar, if needed
    plt.title('STFT Magnitude')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    st.pyplot(fig)
    
    fig2 = plt.figure(figsize=figsize)
    plt.plot(t, RMS, color='blue')  # Customize marker and linestyle as needed
    plt.title('Instantaneous RMS')
    plt.xlabel('Time[sec]')
    plt.ylabel('RMS')
    plt.grid(True)
    st.pyplot(fig2)
    
    return max_RMS
    
def calcCWT_norm(stride_interval,time):
    # Perform continuous wavelet transform (CWT)
    wavelet = 'morl'  # Complex Morlet wavelet, but you can choose others
    scales = np.arange(1, 128)  # Range of scales, you can adjust this
    coefficients, frequencies = pywt.cwt(stride_interval, scales, wavelet)

    # Calculate the instantaneous RMS value
    rms = np.sqrt(np.mean(np.abs(coefficients) ** 2, axis=0))
    max_rms = np.max(rms)
    
    # Plot the CWT coefficients
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(coefficients), aspect='auto', extent=[0, len(time), frequencies[-1], frequencies[0]], cmap='jet')
    plt.colorbar(label='Magnitude')
    plt.title('Continuous Wavelet Transform')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    st.pyplot(plt.gcf())

    plt.figure(figsize=(10, 6))
    plt.plot(time, rms, color='blue')
    plt.xlabel('Time [sec]')
    plt.ylabel('Instantaneous RMS')
    plt.title('Instantaneous RMS from Continuous Wavelet Transform Coefficients')
    st.pyplot(plt.gcf())
    
    return max_rms

def calcDFA(stride_interval):
    # Calculate DFA
    dfa = nolds.dfa(stride_interval)
    return dfa

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write(df)
    
    time = df['X'].values
    stride_interval = df['Y'].values
    samplingFreq = 1 / np.mean(np.diff(df['X']))  # Assuming a sampling frequency of 1, adjust this as needed

    # Calculate maximum STFT RMS
    max_STFT_RMS = calcSTFT_norm(stride_interval, samplingFreq)
    st.write(f"Maximum STFT RMS: {max_STFT_RMS}")

    # Calculate maximum CWT RMS
    max_CWT_RMS = calcCWT_norm(stride_interval,time)
    st.write(f"Maximum CWT RMS: {max_CWT_RMS}")

    # Calculate DFA
    dfa = calcDFA(stride_interval)
    st.write(f"DFA: {dfa}")
    
    X_predict = [dfa,max_CWT_RMS,max_STFT_RMS]
    
    # Use the loaded model to predict Y_predict
    Y_predict = loaded_model.predict([X_predict])
    
    # Display the Y_predict
    if Y_predict == 0:
        st.write("The group is Old.")
    elif Y_predict == 1:
        st.write("The group is Young.")
