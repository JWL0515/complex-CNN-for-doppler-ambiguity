# Complex CNN for Doppler Ambiguity
This repo is based on my Diplom thesis. The main Konzept is to solve **classification probelm**. 

Two programming languages are used: **Python** and **Matlab**.

## Introduction
The goal is solving the doppler ambiguity problem. That means using complex range-Doppler maps that contain complex numbers to train CNN models. To achieve this, we need CNN which is applicable to complex numbers. The code in order complexPyTorch is based on the open source **[complexPyTorch](https://github.com/wavefrontshaping/complexPyTorch)**. 

## Dataset
Sorry, i can not upload example dataset here. The example dataset is already too big. 
Example Dataset is on **[DOPPLER AMBIGUITY DATASET](https://ieee-dataport.org/documents/doppler-ambiguity-dataset)**. All data is generated using Matlab to simulate a 77GHz FMCW millimeter-wave radar sensing in the road scenario. 

### Input

The input is called **range-dopller-map** with one target:

![image](https://user-images.githubusercontent.com/123400810/220657220-25804278-aab1-4522-89fc-795c18d1685f.png)

This is the processed data with python. The input is **complex numbers**. In this picture, the dimension of input is **50x64**. The dimension of **raw data** is **1024x64**.

### label

The label is called **Factor 𝐹**. It is based on this formel:

                                            𝑣_𝑐𝑜𝑟𝑟𝑒𝑐𝑡=𝑣_𝑑𝑒𝑡+2𝐹𝑣_𝑚𝑎𝑥

The **𝑣_𝑐𝑜𝑟𝑟𝑒𝑐𝑡**, **𝑣_𝑑𝑒𝑡** are the ground truth velocity of target and the measured velocity of target.

The **𝑣_𝑚𝑎𝑥** is the maximal measurable velocity of radar. 

The **𝑣_𝑐𝑜𝑟𝑟𝑒𝑐𝑡**, **𝑣_𝑑𝑒𝑡**, **𝑣_𝑚𝑎𝑥** can be collected while simulating with Matlab.

If you are interessed in this formel, i highly recommend you to read **[Doppler disambiguation in MIMO FMCW radars with binary phase modulation](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rsn2.12063)**.

## Matlab
Matlab is for dataset generation.

The code is based on **[Radar Signal Simulation and Processing for Automated Driving](https://ww2.mathworks.cn/help/radar/ug/radar-signal-simulation-and-processing-for-automated-driving.html)**. Used toolboxs are: **Automated Driving Toolbox** and **Radar Toolbox**.

Example driving scenario for 2 targets:

![image](https://user-images.githubusercontent.com/123400810/221910056-bf10b0d0-3551-4af4-8b2e-f7493e745347.png)


### load_radar_setting.m
You can set radar parameters here.

#### important parameters
- **Two Modulations**: **TDM** and **BPM**.
- **sender number** 
- **receiver number** 

### load_scenario_setting.m
You can set drive scenario parameters here. 

The **ego-car** does the **uniform motion**. And **non-ego cars** do the **uniform acceleration motion**.

#### important parameters
- **frame_rate** and **running_time**
- **velocity of ego car** and **maxiaml velocity of non-ego cars**
- **maxiaml acceleration of non-ego cars**
- **RCS of non-ego cars**
- **appear probability of non-ego cars**
- **EXTRA**: **generate data within specific velocity range**

### radar_signal_simulation.m
The code is based on **[Radar Signal Simulation and Processing for Automated Driving](https://ww2.mathworks.cn/help/radar/ug/radar-signal-simulation-and-processing-for-automated-driving.html)**.

#### important parameters
- **display_scenario**: whether show video display
- **save_Xcubes**: whether save Xcubes
- **detect_object**:whether use CFAR to detect, 'no': no display, because no detect.
- **deletle_record**: whether delete record if there is error 

### helperAutoDrivingRadarSigProc.m
It is almost like the original one from Matlab example.

### run_loop.m
run 'radar_signal_simulation.m' n times to collecte datas.

## Python
Python is for data processing and model training and testing.

### Added functions in complexPyTorch

The orginal is provided by **[wavefrontshaping](https://github.com/wavefrontshaping)**. More details about complexPyTorch is **[here](https://github.com/wavefrontshaping/complexPyTorch)**

Added functions by me:
- in **complexFunctions.py**:
  - **complex_dropout3d**
  - **complex_avg_pool3d**
  - **_retrieve_elements_from_indices_3D**
  - **complex_max_pool3d**
  - **complex_leaky_relu**

- in **complexLayers.py**:
  - **ComplexConv3d**
  - **NaiveComplexBatchNorm3d**
  - **ComplexDropout3d**
  - **ComplexAvgPool3d**
  - **ComplexMaxPool3d**

### prepare_dataset.py
This script provides various functions to preprocess the raw dataset.

At the begin it is function **cfar_rect**. cfar means **[Constant false alarm rate](https://en.wikipedia.org/wiki/Constant_false_alarm_rate)**. With this function the target can be detcetd. And the Area is called **Region of Interest (ROI)**. The picture below is an example for **two targets**:

![image](https://user-images.githubusercontent.com/123400810/220660636-4098842a-a03d-4cf5-9d3f-64a8e11ed4d4.png)

After using cfar_rect we will get two ROIs, because we have two targets. 

cfar is basis for doppler ambiguity problem solving. After using cfar, we will get multiple single ROI. This make the Algorithm keeping simple. If we do not use cfar, we will get the problems below: 

- we have to train model with data in dimension 50x64. This make model very big compare with data after cfar in like 9x7.
- assume we have **2** targes, and each target have **4** possible velocty. Then we will have **1**6 possible combinations. But if we use cfar, we will get **2 single ROI**. This means we acctually have exact only 4 possible velocty for each target. This make preparing dataset also much easily.

After function cfar_rect, it is the main body **class PrepareDataset** for processing dataset.

The main functions for processing dataset:

- **convert_mat_dataset**: convert mat data to label.csv, Xbf.npy, Xcube.npy
- **process_decimate**: decimate Xbf.npy. original:1024x64. After decimating: 50x64 (This is depend on your radar setting).
- **process_normalize**: do normalization on whole dataset
- **process_doppler_vector**: get doppler vector datase
- **process_multiple_frames**: combined serveal .npy. Function is like [HMDB51](https://pytorch.org/vision/main/generated/torchvision.datasets.HMDB51.html#torchvision.datasets.HMDB51) in PyTorch.
- **process_frame_minus_frame**: do frame2-frame1, frame3-frame2...
- **process_cfar_rect_single_target**: get CFAR-ROI dataset for one target
- **process_cfar_rect_multiple_targets**: get CFAR-ROI dataset for multiple target
- **balance_label_factor**: balace dataset based on factor
- **balance_label_velocity**: balance dataset based on velocity range

#### How to use:
Details are in script. Here is only the quick review.

create one obejct firstly:

                                            preparedata = PrepareDataset()
                    
Then you can use function now:

                                            preparedata.process_normalize()

### baseline.py	
This script shows the structure of CNN and the training- and testing phase. It provided only a idea, the reason is:

The purpose of this thsis is to explore the feasibility and try to make the parameters of the model as few as possible, the structure of the model is very simple and parameters are also not 'perfect'. 

## Models
Some trained models are loaded. 

## Results

Only the main results will be showed. All results are tested with test dataset. And Dataset are generater with **TDM**.

### One Target - Dataset with Dimension 1x64 (Doppler-Vector) for 1D CNN
![image](https://user-images.githubusercontent.com/123400810/220673252-0ec8521d-ae26-4aee-9e06-97db30f4979c.png)

### One Target - Dataset with Dimension 50x64 for 2D and 3D CNN
![models_comparison](https://user-images.githubusercontent.com/123400810/220674490-6c156cd5-bcd7-4844-9ffd-29fb1743854b.png)

**Conv3D_fna** means traning CNN 3D using dataset with n frames.

**Conv2D_fna** means traning CNN 2D using dataset with n frame(s).

**Conv2D_delta1a** means traning CNN 2D using dataset with 1 frame (this frame = frame2 - frame1).

The number near the color circle is the total params. x-achse is the total mult-adds (M). And y is the accuracy on test dataset.

### One Target - Dataset with Dimension 7x7 for 2D and 3D CNN
![models_comparison](https://user-images.githubusercontent.com/123400810/220674578-85354d86-9337-4a5c-bf76-0eac62165a8c.png)

**Conv3D_fna** means traning CNN 3D using dataset with n frames.

**Conv2D_fna** means traning CNN 2D using dataset with n frame(s).

**Conv2D_delta1a** means traning CNN 2D using dataset with 1 frame (this frame = frame2 - frame1).

The number near the color circle is the total params. x-achse is the total mult-adds (M). And y is the accuracy on test dataset.

### Two Targets - Dataset with Dimension 9x7 for 2D CNN
The model for two targets is exactly the model for 1 target with Dataset with Dimension 9x7.

Accuracy is 77.5%.

## Summary

- Using Deep Learning can solve doppler ambiguity problem for one target and multiple target.
- 2D CNN with 1 frame has the best performence (accuracy, computation and parameters). 
- (Model with 2D-CNN, 1 frame and Dimension 9x7 has the best performence. This is based on experiments)

## Futher Work

- collect more data for train and find better parameters for models
- collect real worl data and use them to test
- imporve cfar
