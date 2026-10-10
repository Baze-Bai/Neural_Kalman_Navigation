from math import atan2, pi, sqrt, atan2, sin, cos, radians, ceil
import math
import numpy as np
from scipy import interpolate
from traj_utils import *
from tqdm import tqdm
import tensorflow as tf


ACCELEROMETER_NOISE_VARIANCE = 0.00615490459 # 噪声误差 0.00615490459 #(m/s^2)^2, 0.0011858 
GYROSCOPE_NOISE_VARIANCE = 0.0000030462#0.0000030462 #(rad/s)^2 0.0000030462
MAGNETOMETER_NOISE_VARIANCE = 0.36 #0.36 #ut^2 0.09 
GYROSCOPE_ARW = [3.09, 2.7, 5.4] #[3.09, 2.7, 5.4] #deg/sqrt(hr) [11.28, 12.85,13.45]
GYROSCOPE_BI = [88.91,78.07,211.4] #[88.91,78.07,211.4] #deg/hr [45.9, 54.68, 52.38]

GPS_VELOCITY_NOISE_VARIANCE = 0.0025 #GPS速度误差
GPS_POSITION_NOISE_VARIANCE = 1.5**2 #GPS位置误差

def kalman_predict(X, P, Q, A, B, G,T): #卡尔曼预测函数，对状态估计与误差矩阵进行更新
    X = A@X + B@T
    P = A@P@np.transpose(A) + G@Q@np.transpose(G)
    return X,P 

# 迭代卡尔曼更新
def kalman_update(X, P, z, R, H, max_iter=10, tol=1e-5):
    
    """
    Iterative EKF update function.
    
    Args:
    X: State estimate (prior).
    P: Error covariance matrix (prior).
    z: Measurement vector.
    R: Measurement noise covariance.
    H: Measurement matrix.
    max_iter: Maximum number of iterations for IEKF.
    tol: Tolerance for convergence.
    
    Returns:
    Updated state X and covariance matrix P.
    """
    
    X_prev = X.copy()
    # 迭代进行卡尔曼更新
    for _ in range(max_iter):
        # Compute the Kalman gain
        K = (P @ np.transpose(H)) @ np.linalg.inv(H @ P @ np.transpose(H) + R)
        
        # Update the state estimate
        X_new = X_prev + K @ (z - H @ X_prev) #更新状态估计
        
        # Update the covariance matrix
        P_new = P - K @ H @ P
        
        # Check for convergence (based on state estimate change)
        if np.linalg.norm(X_new - X_prev) < tol:
            break
        
        X_prev = X_new #为下一轮更新状态估计做准备

    return X_new, P_new







'''
inputs:
net_inp_mat: Windowed training data for imu, equals a q(特征数量) X window_size X n_channels matrix (q is 10 when acc,gyr,mag and physics used)
GT_vel_x, GT_vel_y: ground truth velocities (to generate synthetic GPS) #真实坐标值
size_of_each: index of new trajectory in windowed files 
x0_list, y0_list: initial coordinates for each trajectory #每一个文件的初始坐标值
window_size and stride: input window size and stride for training set #窗口大小和窗口步长
file_idx: index of the trajectory to be considered #引用文件的序号
decimation factor: by what factor to downsample GPS compared to  IMU stride? (e.g. if stride is 20 for 100 Hz sampling rate and window size of 100, the network runs at 4 Hz. A decimation factor of 4 would equate to 1s GPS interval
my_model: the neural network in keras, loaded using model.load(MODEL_NAME.h5) #训练好的模型

outputs: 
fused_pos_x, fused_pos_y: GPS+IMU position #融合预测值
GPS_x, GPS_y: GPS position (for debugging) #根据位置真值数据所造出来的GPS数据

'''

def neural_ekf_gnss_imu(net_inp_mat, GT_vel_x,GT_vel_y, size_of_each,
                x0_list, y0_list,file_idx,window_size,stride,
                gps_decimation_factor,
                my_model):
    
    fused_pos_x = [] #存储最终预测值，将imu数据与GPS数据混合预测
    fused_pos_y = [] 
    
    dt = stride/(window_size-stride) #定义离散时间
    ########################################
    #using emulated GPS here, you can input original GPS data if available
    Gvx_gps, Gvy_gps, Gvx_vel, Gvy_vel, _ = gen_GPS_values_all_traj(GT_vel_x,GT_vel_y, #根据地面速度真值数据生成仿真GPS数据
                          size_of_each,x0_list,y0_list,window_size,stride,gps_decimation_factor,
                          GPS_POSITION_NOISE_VARIANCE, GPS_VELOCITY_NOISE_VARIANCE)
    
    GPS_x = Gvx_gps[file_idx] #获取序号对应文件的GPS位置数据
    GPS_y = Gvy_gps[file_idx]
    GPS_vel_x = Gvx_vel[file_idx] #获取序号对应文件的GPS速度数据
    GPS_vel_y = Gvy_vel[file_idx]
    ########################################
    if (file_idx == 0): #根据文件序号划分输入特征数据
        cur_inp = net_inp_mat[0:size_of_each[0],:,:] #根据文件序号判断每个文件所对应的特征长度
    elif (file_idx == 1):
        cur_inp = net_inp_mat[np.sum(size_of_each[0]):np.sum(size_of_each[0:file_idx+1]),:,:]

    else:
        cur_inp = net_inp_mat[np.sum(size_of_each[0:file_idx]):np.sum(size_of_each[0:file_idx+1]),:,:]
        
    X = np.array([x0_list[file_idx],y0_list[file_idx],0.0,0.0]).reshape(4,1) #初始化状态变量
    A = np.array(((1.0,0.0,0.0,0.0),(0.0,1.0,0.0,0.0),(0.0,0.0,0.0,0.0),(0.0,0.0,0.0,0.0))) 
    H = np.identity(4) #生成单位矩阵
    B = np.array(((dt, 0.0),(0.0,dt),(1.0,0),(0.0,1.0))) 
    R = np.identity(4) #观测协方差矩阵
    R[0,0] = GPS_POSITION_NOISE_VARIANCE
    R[1,1] = GPS_POSITION_NOISE_VARIANCE
    R[2,2] = GPS_VELOCITY_NOISE_VARIANCE
    R[3,3] = GPS_VELOCITY_NOISE_VARIANCE
    #R为对角矩阵，对角线上为误差变量
    P = 1e-5*np.zeros((4,4)) 
    gps_counter = 1
    for i in tqdm(range(cur_inp.shape[0])):
        image = tf.cast(cur_inp[i,:,:].reshape(1,cur_inp.shape[1],cur_inp.shape[2]), tf.float32)
        with tf.GradientTape(persistent=True) as t:
            t.watch(image)
            vx_jacob = my_model(image)[0][0]
            vy_jacob = my_model(image)[1][0]
            pred = my_model(image)
            
        my_grad1 = np.array(t.gradient(vx_jacob, image)).reshape(cur_inp.shape[1],cur_inp.shape[2])
        my_grad2 = np.array(t.gradient(vy_jacob, image)).reshape(cur_inp.shape[1],cur_inp.shape[2])
            
                    
        T = np.array(pred).flatten().reshape(2,1) #包含神经网络预测结果
        
        G = np.zeros((4,10)) 
        for j in range(10):
            G[0,j] = dt*np.sum(np.abs(my_grad1[:,j]))/np.sum(np.abs(my_grad1))
            G[1,j] = dt*np.sum(np.abs(my_grad2[:,j]))/np.sum(np.abs(my_grad2))
            G[2,j] = np.sum(np.abs(my_grad1[:,j]))/np.sum(np.abs(my_grad1))
            G[3,j] = np.sum(np.abs(my_grad2[:,j]))/np.sum(np.abs(my_grad2)) #利用特征对输出的梯度估计特征对状态更新的贡献

        
        Q = np.diag((ACCELEROMETER_NOISE_VARIANCE,
                     ACCELEROMETER_NOISE_VARIANCE,
                     ACCELEROMETER_NOISE_VARIANCE,
                     ((np.deg2rad(GYROSCOPE_ARW[0])/60.0)*sqrt(dt))**2+(np.deg2rad(GYROSCOPE_BI[0])/3600.0)**2+GYROSCOPE_NOISE_VARIANCE,     
                     ((np.deg2rad(GYROSCOPE_ARW[1])/60.0)*sqrt(dt))**2+(np.deg2rad(GYROSCOPE_BI[1])/3600.0)**2+GYROSCOPE_NOISE_VARIANCE,                      
                     ((np.deg2rad(GYROSCOPE_ARW[2])/60.0)*sqrt(dt))**2+(np.deg2rad(GYROSCOPE_BI[2])/3600.0)**2+GYROSCOPE_NOISE_VARIANCE,                      
             MAGNETOMETER_NOISE_VARIANCE,
             MAGNETOMETER_NOISE_VARIANCE,
             MAGNETOMETER_NOISE_VARIANCE,
             3*ACCELEROMETER_NOISE_VARIANCE               
            )) #传感器的协方差矩阵

        X, P = kalman_predict(X, P, Q, A, B, G,T) #更新X状态估计与误差矩阵P，X状态估计即估计x轴位置与y轴位置
        
        if(i%gps_decimation_factor == 0 and gps_counter < len(GPS_x)): #每隔gps_decimation_factor根据GPS信号更新一次x轴与y轴的定位
            z = np.array((GPS_x[gps_counter],GPS_y[gps_counter],GPS_vel_x[gps_counter],GPS_vel_y[gps_counter])).reshape(4,1)
            X, P = kalman_update(X,P,z,R,H) #通过卡尔曼更新函数得到更新后的X与P
            gps_counter+=1 
        
        fused_pos_x.append(X[0,0])
        fused_pos_y.append(X[1,0])
    
    return fused_pos_x, fused_pos_y, GPS_x, GPS_y #返回混合预测结果与GPS位置数据