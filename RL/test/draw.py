import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os


filelist=os.listdir('/home/jtangas/catkin_ws_garyT/src/RL_position_control_gazebo/RL/trained_model/10_12_diff_wheel_64_32_terminal_witout_v_max')
for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
    if not(fichier.endswith(".pth")):
        filelist.remove(fichier)
print(filelist)
print(filelist[5][-9:-4])
path = [[2,2],[2,3],[2,4]]
x,y = zip(*path)
plt.plot(x,y)
plt.savefig("{}_episodes.png".format(filelist[5][-9:-4]))
