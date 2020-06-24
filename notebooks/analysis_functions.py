import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import medfilt

def plot_instrument_data(file_path_instrument_data: str, start_deform: int, end_deform: int, 
                         load_conversion: int = 25, temp_conversion: int = 150, position_conversion = 0.5): 
    
    """ Plot load, temperature and position data, from instrument data recorded for each diffraction pattern image.
    The instrument data is in the form of an analogue voltage signal, with a conversion factor used to calculate 
    the Newtons, Degree Celsius or Millimetre values to plot.
    
    :param file_path_instrument_data: input file path string.
    :param start_deform: value defining the start point of deformation.
    :param end_deform: value defining the end point of deformation.
    :param load_conversion: conversion for load in Newton / Volt (default is 25).
    :param temp_conversion: conversion for temperature in Degree Celsius / Volt (default is 150).
    :param position_conversion: conversion for position in Millimetres / Volt  (default is 0.5).
    
    :return: length of the instrument data (equivalent to the maximum frame).
    """ 
    
    instrument_data = np.loadtxt(file_path_instrument_data, skiprows=6)
    max_frame=len(instrument_data)

    load = instrument_data[:, 1] * load_conversion
    temperature = instrument_data[:, 2] * temp_conversion
    position = instrument_data[:, 3] * position_conversion

    plt.plot(instrument_data[:, 0], temperature)
    plt.xlabel("Frame_number")
    plt.ylabel("Temperature (°C)")
    plt.show()

    plt.plot(instrument_data[:, 0], position)
    plt.xlabel("Frame_number")
    plt.ylabel("ETMT Gauge Position")
    plt.show()

    plt.plot(instrument_data[start_deform:end_deform, 0], load[start_deform:end_deform])
    plt.xlabel("Frame_number")
    plt.ylabel("Load (N)")
    plt.show()
    
    return max_frame

def plot_ETMT_data(file_path_etmt_data: str, number_deform_frames: int, first_point: int, last_point: int,
                  acquisition_freq_sxrd: int = 10, acquisition_freq_etmt: int = 50, filter_data: int = 0):
    """ Plot true stress, true stress versus true strain, and true stress versus true strain at an adjusted frequency
    from data recorded at the ETMT.
    
    :param file_path_etmt_data: input file path string.
    :param first_point: value defining the start point of deformation.
    :param last_point: value defining the end point of deformation.
    :param acquisition_freq_sxrd: acquisition frequency of the SXRD detector in Hz (default is 10 Hz).
    :param acquisition_freq_etmt: cacquisition frequency of the ETMT data recorder in Hz (default is 50 Hz).
    :param filter_data: size of the median filter window to filter the data with medfilt function (default is ).
    
    :return: NumPy arrays for the true stress and true strain.
    """

    etmt_data = np.loadtxt(file_path_etmt_data, delimiter=',', skiprows=1)

    acquisition_frequency_ratio = int(acquisition_freq_etmt / acquisition_freq_sxrd)
    print(acquisition_frequency_ratio)
    end_deform_point = int(acquisition_frequency_ratio * number_deform_frames)
    
    data_load = etmt_data[first_point-1:last_point, 0]
    data_strain = etmt_data[first_point-1:last_point, 3]
    data_stress = etmt_data[first_point-1:last_point, 4]

    # some filtering of the data is needed due to fluctuations in the ETMT data
    filter_strain = medfilt(data_strain,filter_data)
    filter_stress = medfilt(data_stress,filter_data)
    
    # plot the true stress for each point in the ETMT data
    plt.plot(filter_stress[0:end_deform_point],'-o', color='blue',markersize=10)
    plt.title('ETMT Data')
    plt.ylabel('True Stress, ${\sigma}$ (MPa)')
    plt.xlabel('ETMT Data Number')
    plt.show()
    
    # plot the true stress versus true strain at the ETMT acquisition frequency
    plt.plot(filter_strain[0:end_deform_point],filter_stress[0:end_deform_point],'-o',color='blue', markersize=10)
    plt.title('ETMT Data')
    plt.ylabel('True Stress, ${\sigma}$ (MPa)')
    plt.xlabel('True Strain, ${\epsilon}$')
    plt.show()
    
    # plot the true stress versus true strain, with each point representing a diffraction pattern image
    true_strain=np.empty([number_deform_frames,])
    true_stress=np.empty([number_deform_frames,])
    for image in range(0, number_deform_frames):
        true_strain[image]=filter_strain[image*acquisition_frequency_ratio]
        true_stress[image]=filter_stress[image*acquisition_frequency_ratio]  
    
    plt.plot(true_strain, true_stress, 'o',color='blue', markersize=10)
    plt.title('ETMT Data at Adjusted Frequency')
    plt.ylabel('True Stress, $\sigma$ (MPa)')
    plt.xlabel('True Strain, $\epsilon$')
    plt.show()
    
    return (true_stress, true_strain)

def colour_range(N: int = 9, colour_map: str = 'viridis'):
    """ Return a range of hex colour codes for a particular colour map.
    
    :param N: number of desired colour codes (default is 9).
    :param colour_map: type of colour map (default is 'viridis').
    :return: list of hex colour codes
    """
    base = plt.cm.get_cmap(colour_map)
    colour_list = base(np.linspace(0, 1, N))
    colour_hex_list=[]
    for i in range (N-1, -1, -1):
         colour_hex_list.append(colors.rgb2hex(colour_list[i]))
    
    return colour_hex_list

def find_nearest(array, value):
    """ Find the nearest value in an array and return it's index.
    
    :param array: NumPy array of values.
    :param value: value to look for in the array 
    :return: index of nearest value in array.
    """ 
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    print('value to match =', value, 'value found in array =', array[idx], 'index of array =', idx)
    
    return idx

def x_ray_wavelength(x_ray_energy: float = 89.07) -> float:
    """ Calculate X-ray wavelength (in metres) from X-ray energy (in keV) using Planck's equation.
    
    :param x_ray_energy: X-ray energy in keV (default is 89.07 keV).
    :return: X-ray wavelength in metres.
    """ 
    c = 2.99792458e8
    h = 6.62607004e-34
    e = 1.6021766208e-19
    x_ray_wavelength=(h * c) / (x_ray_energy * 1e3 * e)
    
    return x_ray_wavelength

def calc_dspacing(two_theta: float, x_ray_energy: float = 89.07) -> float:
    """ Calculate d-spacing from 2-theta values using Bragg's law.
    
    :param two_theta: 2-theta value in degrees.
    :param x_ray_energy: X-ray energy in keV (default is 89.07 keV).
    :return: d-spacing in metres.
    """ 
    c = 2.99792458e8
    h = 6.62607004e-34
    e = 1.6021766208e-19
    x_ray_wavelength = (h * c) / (x_ray_energy * 1e3 * e)
    dspacing = x_ray_wavelength / (2 * np.sin(np.array(two_theta) * np.pi / 360))
    
    return dspacing

def calc_strain(two_theta: np.ndarray, zero_range: int = 1) -> np.ndarray:
    """ Calculate strain from 2-theta values.
    
    :param two_theta: 2-theta value in degrees.
    :param zero_range: Integer used to define a range of points to calculate an average initial 2-theta value
                       (default is 1).
    :return: NumPy array of strain values in degrees.
    """ 
    two_theta = 0.5 * (np.array(two_theta)) * np.pi / 180.0
    two_theta_0 = np.mean(two_theta[0:zero_range])
    print(two_theta_0)
    strain = - (two_theta - two_theta_0) / np.tan(two_theta)
    
    return strain

def relative_amplitude(amplitude: np.ndarray) -> np.ndarray:
    """ Divide an array of amplitude values by the first value in the array.
    
    :param amplitude: NumPy array of amplitude float values.
    :return: NumPy array of float values.
    """ 
    relative_amplitude = np.array(amplitude) / amplitude[0]
    
    return relative_amplitude