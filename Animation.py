import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# We proceed Load the data
df = pd.read_csv('Lets_Train.csv')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xdata, ydata, zdata = [], [], []
ln1, = plt.plot([], [], [], 'ro')
ln2, = plt.plot([], [], [], 'ro')
ln3, = plt.plot([], [], [], 'ro')
ln4, = plt.plot([], [], [], 'ro')
ln5, = plt.plot([], [], [], 'ro')
ln6, = plt.plot([], [], [], 'ro')
ln7, = plt.plot([], [], [], 'ro')
ln8, = plt.plot([], [], [], 'ro')
ln9, = plt.plot([], [], [], 'ro')
ln10, = plt.plot([], [], [], 'ro')
ln11, = plt.plot([], [], [], 'ro')
ln12, = plt.plot([], [], [], 'ro')
ln13, = plt.plot([], [], [], 'ro')
ln14, = plt.plot([], [], [], 'ro')
ln15, = plt.plot([], [], [], 'ro')
ln16, = plt.plot([], [], [], 'ro')
ln17, = plt.plot([], [], [], 'ro')
ln18, = plt.plot([], [], [], 'ro')
ln19, = plt.plot([], [], [], 'ro')
ln20, = plt.plot([], [], [], 'ro')
ln21, = plt.plot([], [], [], 'ro')
ln22, = plt.plot([], [], [], 'ro')
ln23, = plt.plot([], [], [], 'ro')
ln24, = plt.plot([], [], [], 'ro')
ln25, = plt.plot([], [], [], 'ro')
ln26, = plt.plot([], [], [], 'ro')
ln27, = plt.plot([], [], [], 'ro')
ln28, = plt.plot([], [], [], 'ro')
ln29, = plt.plot([], [], [], 'ro')
ln30, = plt.plot([], [], [], 'ro')
ln31, = plt.plot([], [], [], 'ro')
ln32, = plt.plot([], [], [], 'ro')

def init():

    x_columns = [f'X_{i}' for i in range(0, 33)]  
    y_columns = [f'Y_{i}' for i in range(0, 33)]  
    z_columns = [f'Z_{i}' for i in range(0, 33)]

    ax.set_xlim(df[x_columns].min().min(), df[x_columns].max().max())
    ax.set_ylim(df[y_columns].min().min(), df[y_columns].max().max())
    ax.set_zlim(df[z_columns].min().min(), df[z_columns].max().max())
    return ln1, ln2, ln3, ln4, ln5, ln6, ln7, ln8, ln9, ln10, ln11, ln12, ln13, ln14, ln15, ln16, ln17, ln18, ln19, ln20, ln21, ln22, ln23, ln24, ln25, ln26, ln27, ln28, ln29, ln30, ln31, ln32

def update(frame):

    # Update data fot Dot 1
    x1 = [df.loc[frame, 'X_0']]
    y1 = [df.loc[frame, 'Y_0']]
    z1 = [df.loc[frame, 'Z_0']]
    ln1.set_data(x1, y1)
    ln1.set_3d_properties(z1)

    # Update data fot Dot 2
    x2 = [df.loc[frame, 'X_1']]
    y2 = [df.loc[frame, 'Y_1']]
    z2 = [df.loc[frame, 'Z_1']]
    ln2.set_data(x2, y2)
    ln2.set_3d_properties(z2)

    # Update data fot Dot 3
    x3 = [df.loc[frame, 'X_2']]
    y3 = [df.loc[frame, 'Y_2']]
    z3 = [df.loc[frame, 'Z_2']]
    ln3.set_data(x3, y3)
    ln3.set_3d_properties(z3)

    # Update data fot Dot 4
    x4 = [df.loc[frame, 'X_3']]
    y4 = [df.loc[frame, 'Y_3']]
    z4 = [df.loc[frame, 'Z_3']]
    ln4.set_data(x4, y4)
    ln4.set_3d_properties(z4)

    # Update data fot Dot 5
    x5 = [df.loc[frame, 'X_4']]
    y5 = [df.loc[frame, 'Y_4']]
    z5 = [df.loc[frame, 'Z_4']]
    ln5.set_data(x5, y5)
    ln5.set_3d_properties(z5)

    # Update data fot Dot 6
    x6 = [df.loc[frame, 'X_5']]
    y6 = [df.loc[frame, 'Y_5']]
    z6 = [df.loc[frame, 'Z_5']]
    ln6.set_data(x6, y6)
    ln6.set_3d_properties(z6)

    # Update data fot Dot 7
    x7 = [df.loc[frame, 'X_6']]
    y7 = [df.loc[frame, 'Y_6']]
    z7 = [df.loc[frame, 'Z_6']]
    ln7.set_data(x7, y7)
    ln7.set_3d_properties(z7)

    # Update data fot Dot 8
    x8 = [df.loc[frame, 'X_7']]
    y8 = [df.loc[frame, 'Y_7']]
    z8 = [df.loc[frame, 'Z_7']]
    ln8.set_data(x8, y8)
    ln8.set_3d_properties(z8)

    # Update data fot Dot 9
    x9 = [df.loc[frame, 'X_8']]
    y9 = [df.loc[frame, 'Y_8']]
    z9 = [df.loc[frame, 'Z_8']]
    ln9.set_data(x9, y9)
    ln9.set_3d_properties(z9)

    # Update data fot Dot 10
    x10 = [df.loc[frame, 'X_9']]
    y10 = [df.loc[frame, 'Y_9']]
    z10 = [df.loc[frame, 'Z_9']]
    ln10.set_data(x10, y10)
    ln10.set_3d_properties(z10)
    
    # Update data fot Dot 11
    x11 = [df.loc[frame, 'X_10']]
    y11 = [df.loc[frame, 'Y_10']]
    z11 = [df.loc[frame, 'Z_10']]
    ln11.set_data(x11, y11)
    ln11.set_3d_properties(z11)
    
    # Update data fot Dot 12
    x12 = [df.loc[frame, 'X_11']]
    y12 = [df.loc[frame, 'Y_11']]
    z12 = [df.loc[frame, 'Z_11']]
    ln12.set_data(x12, y12)
    ln12.set_3d_properties(z12)

    # Update data fot Dot 13
    x13 = [df.loc[frame, 'X_12']]
    y13 = [df.loc[frame, 'Y_12']]
    z13 = [df.loc[frame, 'Z_12']]
    ln13.set_data(x13, y13)
    ln13.set_3d_properties(z13)

    # Update data fot Dot 14
    x14 = [df.loc[frame, 'X_13']]
    y14 = [df.loc[frame, 'Y_13']]
    z14 = [df.loc[frame, 'Z_13']]
    ln14.set_data(x14, y14)
    ln14.set_3d_properties(z14)

    # Update data fot Dot 15
    x15 = [df.loc[frame, 'X_14']]
    y15 = [df.loc[frame, 'Y_14']]
    z15 = [df.loc[frame, 'Z_14']]
    ln15.set_data(x15, y15)
    ln15.set_3d_properties(z15)

    # Update data fot Dot 16
    x16 = [df.loc[frame, 'X_15']]
    y16 = [df.loc[frame, 'Y_15']]
    z16 = [df.loc[frame, 'Z_15']]
    ln16.set_data(x16, y16)
    ln16.set_3d_properties(z16)

    # Update data fot Dot 17
    x17 = [df.loc[frame, 'X_16']]
    y17 = [df.loc[frame, 'Y_16']]
    z17 = [df.loc[frame, 'Z_16']]
    ln17.set_data(x17, y17)
    ln17.set_3d_properties(z17)

    # Update data fot Dot 18
    x18 = [df.loc[frame, 'X_17']]
    y18 = [df.loc[frame, 'Y_17']]
    z18 = [df.loc[frame, 'Z_17']]
    ln18.set_data(x18, y18)
    ln18.set_3d_properties(z18)

    # Update data fot Dot 19
    x19 = [df.loc[frame, 'X_18']]
    y19 = [df.loc[frame, 'Y_18']]
    z19 = [df.loc[frame, 'Z_18']]
    ln19.set_data(x19, y19)
    ln19.set_3d_properties(z19)

    # Update data fot Dot 20
    x20 = [df.loc[frame, 'X_19']]
    y20 = [df.loc[frame, 'Y_19']]
    z20 = [df.loc[frame, 'Z_19']]
    ln20.set_data(x20, y20)
    ln20.set_3d_properties(z20)

    # Update data fot Dot 21
    x21 = [df.loc[frame, 'X_20']]
    y21 = [df.loc[frame, 'Y_20']]
    z21 = [df.loc[frame, 'Z_20']]
    ln21.set_data(x21, y21)
    ln21.set_3d_properties(z21)

    # Update data fot Dot 22
    x22 = [df.loc[frame, 'X_21']]
    y22 = [df.loc[frame, 'Y_21']]
    z22 = [df.loc[frame, 'Z_21']]
    ln22.set_data(x22, y22)
    ln22.set_3d_properties(z22)

    # Update data fot Dot 23
    x23 = [df.loc[frame, 'X_22']]
    y23 = [df.loc[frame, 'Y_22']]
    z23 = [df.loc[frame, 'Z_22']]
    ln23.set_data(x23, y23)
    ln23.set_3d_properties(z23)

    # Update data fot Dot 24
    x24 = [df.loc[frame, 'X_23']]
    y24 = [df.loc[frame, 'Y_23']]
    z24 = [df.loc[frame, 'Z_23']]
    ln24.set_data(x24, y24)
    ln24.set_3d_properties(z24)

    # Update data fot Dot 25
    x25 = [df.loc[frame, 'X_24']]
    y25 = [df.loc[frame, 'Y_24']]
    z25 = [df.loc[frame, 'Z_24']]
    ln25.set_data(x25, y25)
    ln25.set_3d_properties(z25)

    # Update data fot Dot 26
    x26 = [df.loc[frame, 'X_25']]
    y26 = [df.loc[frame, 'Y_25']]
    z26 = [df.loc[frame, 'Z_25']]
    ln26.set_data(x26, y26)
    ln26.set_3d_properties(z26)

    # Update data fot Dot 27
    x27 = [df.loc[frame, 'X_26']]
    y27 = [df.loc[frame, 'Y_26']]
    z27 = [df.loc[frame, 'Z_26']]
    ln27.set_data(x27, y27)
    ln27.set_3d_properties(z27)

    # Update data fot Dot 28
    x28 = [df.loc[frame, 'X_27']]
    y28 = [df.loc[frame, 'Y_27']]
    z28 = [df.loc[frame, 'Z_27']]
    ln28.set_data(x28, y28)
    ln28.set_3d_properties(z28)

    # Update data fot Dot 29
    x29 = [df.loc[frame, 'X_28']]
    y29 = [df.loc[frame, 'Y_28']]
    z29 = [df.loc[frame, 'Z_28']]
    ln29.set_data(x29, y29)
    ln29.set_3d_properties(z29)

    # Update data fot Dot 30
    x30 = [df.loc[frame, 'X_29']]
    y30 = [df.loc[frame, 'Y_29']]
    z30 = [df.loc[frame, 'Z_29']]
    ln30.set_data(x30, y30)
    ln30.set_3d_properties(z30)

    # Update data fot Dot 31
    x31 = [df.loc[frame, 'X_30']]
    y31 = [df.loc[frame, 'Y_30']]
    z31 = [df.loc[frame, 'Z_30']]
    ln31.set_data(x31, y31)
    ln31.set_3d_properties(z31)

    # Update data fot Dot 32
    x32 = [df.loc[frame, 'X_31']]
    y32 = [df.loc[frame, 'Y_31']]
    z32 = [df.loc[frame, 'Z_31']]
    ln32.set_data(x32, y32)
    ln32.set_3d_properties(z32)
    
    return ln1, ln2, ln3, ln4, ln5, ln6, ln7, ln8, ln9, ln10, ln11, ln12, ln13, ln14, ln15, ln16, ln17, ln18, ln19, ln20, ln21, ln22, ln23, ln24, ln25, ln26, ln27, ln28, ln29, ln30, ln31, ln32


# Create the animation
ani = FuncAnimation(fig, update, frames=range(len(df)),
                    init_func=init, blit=False)  # For 3D plots, blit must be False

plt.show()




