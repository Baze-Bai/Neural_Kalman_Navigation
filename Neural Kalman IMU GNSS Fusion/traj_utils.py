from math import atan2, pi, sqrt, atan2, sin, cos, radians, ceil
import math
import numpy as np
from scipy import interpolate


'''
generate ATE and RTE values

inputs:
Gvx, Gvy: Ground truth velocity (in m/s)
Pvx, Pvy: neural network predicted velocity (in m/s) #神经网络的预测速度
sampling_rate: data sampling rate #取样速率
window_size and stride: input window size and stride for training set
length: how many samples? 

outputs:
ate,rte: ATE and RTE (in m) 
at_all, rt_all: Trajectory lengths (full, and 60 seconds) (in m) 

'''
def Cal_TE(Gvx, Gvy, Pvx, Pvy, sampling_rate=100,window_size=200,stride=10,length=None):
    
    if length==None:
        length = len(Pvx)
        
    distance = []
    
    for i in range(length):
        d = ((Gvx[i]-Pvx[i])*(Gvx[i]-Pvx[i])) + ((Gvy[i]-Pvy[i])*(Gvy[i]-Pvy[i])) #计算轨迹偏差的平方
        d = math.sqrt(d)
        distance.append(d) #将轨迹的偏差集合到一个列表中
    
    mean_distance = sum(distance)/len(distance) #求总体轨迹平均偏差值即ATE
    ate = mean_distance
    at_all = distance #各点偏差的列表
    
    n_windows_one_min= int(((sampling_rate*60)-window_size)/stride)
    distance = []
    if(n_windows_one_min < length):
        for i in range(n_windows_one_min):
            d = ((Gvx[i]-Pvx[i])*(Gvx[i]-Pvx[i])) + ((Gvy[i]-Pvy[i])*(Gvy[i]-Pvy[i]))
            d = math.sqrt(d)
            distance.append(d) #distance容纳一分钟内所有的相对偏差
        rte = sum(distance)/len(distance) #求平均值得到ATE
    else:
        rte=ate*(n_windows_one_min/length) #根据比率求得RTE
    
    rt_all = distance
    return ate, rte, at_all, rt_all #返回平均的ATE、RTE，轨迹总的ATE、RTE


'''Calculate Trajectory Length

inputs:
Gvx, Gvy: Velocity
length: how many samples?

outputs:
sum_distance: trajectory length in meters

'''
def Cal_len_meters(Gvx, Gvy, length=None):
    if length==None:
        length = len(Gvx)
        
    distance = []
    
    for i in range(1, length):
        d = ((Gvx[i]-Gvx[i-1])*(Gvx[i]-Gvx[i-1])) + ((Gvy[i]-Gvy[i-1])*(Gvy[i]-Gvy[i-1]))
        d = math.sqrt(d) #得到每个前后点之间的距离
        distance.append(d) #将每个前后点之间的距离储存在distance中
    
    sum_distance = sum(distance) #计算轨迹的总距离
    
    return sum_distance  


'''
generate trajectory from ground truth velocity for specific file

inputs:
GT_vel_x, GT_vel_y: Ground truth velocity (in m/s) #所有速度真值数据
size_of_each: index of new trajectory in windowed files #表示每个轨迹所包含的窗口数目
x0_list, y0_list: initial coordinates for each trajectory #初始点坐标 
window_size and stride: input window size and stride for training set 
file_idx: index of the file to be considered #文件序号

outputs:
Gvx, Gvy: x and y position (in m)

'''
def GT_pos_generator(GT_vel_x, GT_vel_y, size_of_each, 
                   x0_list, y0_list, window_size, stride,file_idx):  
    if (file_idx == 0):
        x_vel_test_sel = GT_vel_x[0:size_of_each[0]] #将真实坐标值作为测试数据集
        y_vel_test_sel = GT_vel_y[0:size_of_each[0]]
    elif (file_idx == 1):
        x_vel_test_sel = GT_vel_x[np.sum(size_of_each[0]):np.sum(size_of_each[0:file_idx+1])]
        y_vel_test_sel = GT_vel_y[np.sum(size_of_each[0]):np.sum(size_of_each[0:file_idx+1])]

    else:
        x_vel_test_sel = GT_vel_x[np.sum(size_of_each[0:file_idx]):np.sum(size_of_each[0:file_idx+1])] #根据文件序号区分对应的速度真值数据
        y_vel_test_sel = GT_vel_y[np.sum(size_of_each[0:file_idx]):np.sum(size_of_each[0:file_idx+1])]

    pointx = []
    pointy = []
    Lx =  x0_list[file_idx]
    Ly = y0_list[file_idx]
    for j in range(len(x_vel_test_sel)):
        Lx = Lx + (x_vel_test_sel[j]/(((window_size-stride)/stride)))
        Ly = Ly + (y_vel_test_sel[j]/(((window_size-stride)/stride)))    
        pointx.append(Lx)
        pointy.append(Ly)   
    Gvx = pointx
    Gvy = pointy
    
    return Gvx, Gvy #返回真值数据点


'''
generate synthetic GPS values for specific file from ground truth velocities #针对特定文件生成仿真GPS数据

inputs:
GT_vel_x, GT_vel_y: Ground truth velocity (in m/s)
size_of_each: index of new trajectory in windowed files 
x0_list, y0_list: initial coordinates for each trajectory 
window_size and stride: input window size and stride for training set
file_idx: index of the file to be considered
decimation factor: by what factor to downsample GPS compared to  IMU stride? (e.g. if stride is 20 for 100 Hz sampling rate and window size of 100, the network runs at 4 Hz. A decimation factor of 4 would equate to 1s GPS interval
pos_noise_var, velocity_noise_var: GPS position and velocity noise variance, in m^2 and (m/s)^2

outputs:
Gvx, Gvy: x and y position (in m)
GPS_vel_x, GPS_vel_y: x and y velocities from GPS position (in m/s)

'''

def gen_GPS_values(GT_vel_x, GT_vel_y, size_of_each, 
                   x0_list, y0_list, window_size, stride,  file_idx,
                  decimation_factor, pos_noise_var, velocity_noise_var):

    if (file_idx == 0):
        x_vel_test_sel = GT_vel_x[0:size_of_each[0]]
        y_vel_test_sel = GT_vel_y[0:size_of_each[0]]
    elif (file_idx == 1):
        x_vel_test_sel = GT_vel_x[np.sum(size_of_each[0]):np.sum(size_of_each[0:file_idx+1])]
        y_vel_test_sel = GT_vel_y[np.sum(size_of_each[0]):np.sum(size_of_each[0:file_idx+1])]

    else:
        x_vel_test_sel = GT_vel_x[np.sum(size_of_each[0:file_idx]):np.sum(size_of_each[0:file_idx+1])]
        y_vel_test_sel = GT_vel_y[np.sum(size_of_each[0:file_idx]):np.sum(size_of_each[0:file_idx+1])]

    pointx = []
    pointy = []
    Lx =  x0_list[file_idx]
    Ly = y0_list[file_idx]
    for j in range(len(x_vel_test_sel)): 
        Lx = Lx + (x_vel_test_sel[j]/(((window_size-stride)/stride))) 
        Ly = Ly + (y_vel_test_sel[j]/(((window_size-stride)/stride)))    
        pointx.append(Lx)
        pointy.append(Ly)   
    Gvx = pointx #得到GPS点位置数据值
    Gvy = pointy
    GPS_vel_x = np.insert(np.ediff1d(Gvx),0,0.0)
    GPS_vel_y = np.insert(np.ediff1d(Gvy),0,0.0) #通过一阶差分得到速度值
    if(decimation_factor!=0): #降采样，以decimation_factor为间隔
        Gvx = Gvx[0::decimation_factor]
        Gvy = Gvy[0::decimation_factor]
        GPS_vel_x = GPS_vel_x[0::decimation_factor]
        GPS_vel_y = GPS_vel_y[0::decimation_factor]        
    if(pos_noise_var != 0 and velocity_noise_var!=0): #添加噪声，生成仿真gps数据
        Gvx = Gvx + np.random.normal(0,sqrt(pos_noise_var),len(Gvx))
        Gvy = Gvy + np.random.normal(0,sqrt(pos_noise_var),len(Gvy)) #添加位置数据噪声
        GPS_vel_x = GPS_vel_x + np.random.normal(0,sqrt(velocity_noise_var),len(GPS_vel_x)) #添加速度数据噪声
        GPS_vel_y = GPS_vel_y + np.random.normal(0,sqrt(velocity_noise_var),len(GPS_vel_y))    
    
    return Gvx, Gvy, GPS_vel_x, GPS_vel_y


'''
generate  synthetic GPS values for  all files

inputs:
GT_vel_x, GT_vel_y: Ground truth velocity (in m/s)
size_of_each: index of new trajectory in windowed files 
x0_list, y0_list: initial coordinates for each trajectory 
window_size and stride: input window size and stride for training set
decimation factor: by what factor to downsample GPS compared to  IMU stride? (e.g. if stride is 20 for 100 Hz sampling rate and window size of 100, the network runs at 4 Hz. A decimation factor of 4 would equate to 1s GPS interval
pos_noise_var, velocity_noise_var: GPS position and velocity noise variance, in m^2 and (m/s)^2

outputs:
Gvx_list, Gvy_list: x and y position (in m)
Gvx_vel_list, Gvy_vel_list: x and y velocities from GPS position (in m/s)
size_of_each_GPS_list: index of new trajectory in the output velocities and positions

'''
def gen_GPS_values_all_traj(GT_vel_x, GT_vel_y, size_of_each,  #生成所有文件的仿真GPS数据
                   x0_list, y0_list, window_size, stride,
                  decimation_factor, pos_noise_var,velocity_noise_var):
    Gvx_list = []
    Gvy_list = []
    Gvx_vel_list = []
    Gvy_vel_list = []
    size_of_each_GPS_list = []
    
    for i in range(len(size_of_each)):
        Gvx, Gvy, GPS_vel_x, GPS_vel_y = gen_GPS_values(GT_vel_x,GT_vel_y,
                          size_of_each,x0_list,y0_list,window_size,stride,i,decimation_factor,pos_noise_var,velocity_noise_var)
        
        Gvx_list.append(Gvx)
        Gvy_list.append(Gvy)
        Gvx_vel_list.append(GPS_vel_x)
        Gvy_vel_list.append(GPS_vel_y)
        size_of_each_GPS_list.append(len(Gvx))
    return Gvx_list, Gvy_list,Gvx_vel_list, Gvy_vel_list, size_of_each_GPS_list


'''
generate trajectory from neural network predicted velocity for specific file


inputs:
net_inp_mat: The processed and windowed NN input (e.g. the test set matrix)
size_of_each: index of new trajectory in windowed files 
x0_list, y0_list: initial coordinates for each trajectory 
window_size and stride: input window size and stride for training set
file_idx: index of the file to be considered
my_model: the neural network in keras, loaded using model.load(MODEL_NAME.h5)

outputs:
Pvx, Pvy: predicted x and y position (in m)

'''
# 根据神经网络预测的速度生成轨迹
def model_pos_generator(net_inp_mat, size_of_each, 
                   x0_list, y0_list, window_size, stride,file_idx,my_model):
    
    if (file_idx == 0):
        cur_inp = net_inp_mat[0:size_of_each[0],:,:]
    elif (file_idx == 1):
        cur_inp = net_inp_mat[np.sum(size_of_each[0]):np.sum(size_of_each[0:file_idx+1]),:,:]

    else:
        cur_inp = net_inp_mat[np.sum(size_of_each[0:file_idx]):np.sum(size_of_each[0:file_idx+1]),:,:]    
    
    y_pred = my_model.predict(cur_inp)
    
    pointx = []
    pointy = []
    Lx =  x0_list[file_idx]
    Ly = y0_list[file_idx]
    for j in range(len(cur_inp)):
        Lx = Lx + (y_pred[0][j][0]/(((window_size-stride)/stride)))
        Ly = Ly + (y_pred[1][j][0]/(((window_size-stride)/stride)))
        pointx.append(Lx)
        pointy.append(Ly)
    Pvx = pointx
    Pvy = pointy   
    
    return Pvx, Pvy


'''
resample gps to match ground truth sampling rate, but with nearest interpolation

inputs:
x_gps, y_gps: GPs position (x,y) or GPS velocity(x,y)
gt_x: either x or y ground truth velocities

outputs:
gps_x_res, gps_y_res: resampled GPS position or velocity

'''
def resample_GPS(x_gps,y_gps,gt_x):
    my_x = np.linspace(0,len(gt_x),len(x_gps))
    f1 = interpolate.interp1d(my_x, x_gps,kind='nearest')
    f2 = interpolate.interp1d(my_x, y_gps,kind='nearest')

    gps_x_res = f1(np.linspace(0,len(gt_x),len(gt_x))) 
    gps_y_res = f2(np.linspace(0,len(gt_x),len(gt_x)))  
    
    return gps_x_res,gps_y_res