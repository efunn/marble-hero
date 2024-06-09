import cv2
import numpy as np
import pandas as pd

screen_w = 16/9
screen_h = 1

full_screen = np.float32([[-screen_w,-screen_h],[-screen_w,screen_h],
    [screen_w,screen_h],[screen_w,-screen_h]])
perspective_screen = np.float32([[-0.25,0],[-0.75,1],
    [1.75,1],[1.25,0]]) # stable
# perspective_screen = np.float32([[-0.25,0],[-0.9,1],
#     [1.9,1],[1.25,0]]) # alt

trans_mat = cv2.getPerspectiveTransform(full_screen,perspective_screen)

# np.matmul(trans_mat, [x,y,1]) -> [ti*x', ti*y', ti]

x_grid_pts = 100
y_grid_pts = 60
x_pts = np.linspace(-screen_w,screen_w,x_grid_pts)
y_pts = np.linspace(-screen_h,screen_h,y_grid_pts)
grid_pts = np.meshgrid(x_pts,y_pts)
xy_pts = np.array(list(zip(*(pt.flat for pt in grid_pts))))
xy_pts_mat = np.ones((xy_pts.shape[0],xy_pts.shape[1]+1))
xy_pts_mat[:,:2] = xy_pts

res_mat = np.matmul(trans_mat,xy_pts_mat.T).T
for col in range(res_mat.shape[1]):
    res_mat[:,col] = res_mat[:,col]/res_mat[:,-1]
uv_pts = res_mat[:,:-1]

out_mat = np.ones((x_grid_pts*y_grid_pts,5))
out_mat[:,:2] = xy_pts
out_mat[:,2:4] = uv_pts

# debug_array = np.linspace(0,1,x_grid_pts*y_grid_pts)
# out_mat[:,-1] = debug_array[::-1]

df = pd.DataFrame(out_mat)
df.to_csv('perspective.data',index=False,sep='\t')
