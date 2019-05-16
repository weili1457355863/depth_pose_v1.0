#-*-coding:utf-8-*- 
"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 2019/5/14 下午3:53 
  description: prepare the data for dataset(resize,clean stationary pictures,generate depth,pose and image)
"""
from path import Path
import numpy as np
from scipy import misc
from collections import Counter #a dict subclass for counting hashable objects

from kitti_utils import read_calib_file,get_p_rect

class KittiRawLoader(object):
    """
    Obtain all train scenes' path, the format of directory
    root_dir
        2011_09_26
            2011_09_26_drive_0001_sync
                image_00
                oxts
                velodyne_points
            calib_cam_to_cam.txt
    """
    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 min_speed=2,
                 from_speed=True
                 ):
        self.dataset_dir = Path(dataset_dir) #root dir
        self.img_height=img_height
        self.img_width=img_width
        self.min_speed=min_speed
        self.from_speed=from_speed
        # test scenes
        cur_dir=Path(__file__).realpath().dirname()  # current directory of this file
        test_scene_file=cur_dir+'/test_scenes.txt'
        with open(test_scene_file,'r') as f:
            test_scenes=f.readlines()  # include \n
        self.test_scenes=[t[:-1] for t in test_scenes]  # remove \n
        #  train scenes dir(remove test scenes)
        self.scenes_dirs = []
        self.cam_ids=['02','03']
        self.dates=['2011_09_26','2011_09_28','2011_09_29','2011_09_30','2011_10_03']
        self.collect_train_scenes()

    # remove test scenes dir
    def collect_train_scenes(self):
        for date in self.dates:
            dir_scenes = (self.dataset_dir/date).dirs()  # return a list
            for dr in dir_scenes:
                if dr.name[:-5] not in self.test_scenes:  # remove _sync
                    self.scenes_dirs.append(dr)

    # get one scene parameters->2 scenes(02,03)
    def get_scene_parameters(self,dir_scene):
        # Read calibration files
        cam2cam = read_calib_file(dir_scene.parent/'calib_cam_to_cam.txt')
        # oxts data
        oxts = sorted((dir_scene/'oxts'/'data').files('*.txt'))  # Path(dir).files() files of dir
        train_scene_para = []
        for c in self.cam_ids:
            # cid: camera_id(02/03), dir: single scene directory, speed: camera relative speed, frame_id:0000000001,
            # rel_path: the name of relative transform directory
            scene_para = {'cid': c, 'dir': dir_scene, 'speed': [], 'frame_id': [],
                          'rel_path': dir_scene.name+'_'+c}
            for n, f in enumerate(oxts):
                metadata = np.genfromtxt(f)
                speed=metadata[8:11]  # forward, left, upward m/s
                scene_para['speed'].append(speed)
                scene_para['frame_id'].append('{:010d}'.format(n))  # the width is 10 ep0000000001
            sample = self.load_imgs(scene_para,0)  # Get the resize ratio of image sample[1]=zoom_x,sample[2]=zoom_y
            P_rect = get_p_rect(cam2cam,c,sample[1],sample[2])
            scene_para['P_rect']=P_rect
            scene_para['intrinsics']=P_rect[:,:3]
            train_scene_para.append(scene_para)
        return train_scene_para

    # load one picture of one scene and then resize
    def load_imgs(self,scene_data,index):
        img_flie = scene_data['dir']/'image_{}'.format(scene_data['cid'])/'data'/(scene_data['frame_id'][index]+'.png')
        # print('img_file:',img_flie)
        img=misc.imread(img_flie)
        zoom_x=self.img_width/img.shape[1]
        zoom_y=self.img_height/img.shape[0]
        # print('zoom_x:{}/{}={}'.format(self.img_width,img.shape[1],zoom_x))
        # print('zoom_y:{}/{}={}'.format(self.img_height,img.shape[0],zoom_y))
        img=misc.imresize(img,(self.img_height,self.img_width))
        return img, zoom_x, zoom_y

    def generate_depth_map(self, scene_para, index):
        def sub2ind(matrixSize, rowSub, colSub):
            m, n = matrixSize
            return rowSub * (n - 1) + colSub - 1
        # compute projection matrix velodyne->image plane
        R_cam2rect = np.eye(4)
        calib_dir = scene_para['dir'].parent
        cam2cam = read_calib_file(calib_dir / 'calib_cam_to_cam.txt')
        velo2cam =read_calib_file(calib_dir / 'calib_velo_to_cam.txt')
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
        P_rect = np.copy(scene_para['P_rect'])
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
        P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
        # load velodyne points and remove all behind image plane (approximation)
        velo_file_name = scene_para['dir'] / 'velodyne_points' / 'data' / '{}.bin'.format(
            scene_para['frame_id'][index])
        # each row of the velodyne data is forward:x, left:y, up:z, reflectance
        velo = np.fromfile(velo_file_name, dtype=np.float32).reshape(-1, 4)
        velo[:, 3] = 1
        velo = velo[velo[:, 0] >= 0, :]

        # project the points to the camera
        velo_pts_im = np.dot(P_velo2im, velo.T).T
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, -1:]

        # check if in bounds
        # use minus 1 to get the exact same value as KITTI matlab code
        velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
        velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1

        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:, 0] < self.img_width )
        val_inds = val_inds & (velo_pts_im[:, 1] < self.img_height )
        velo_pts_im = velo_pts_im[val_inds, :]

        # project to image
        depth = np.zeros((self.img_height , self.img_width )).astype(np.float32)
        depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth < 0] = 0
        return depth
    # get one scene sample including image and depth
    def get_scene_samples(self,scene_para):
        # construct one sample
        def construct_sample(scene_para,index,frame_id):
            sample={'img':self.load_imgs(scene_para,index)[0],'frame_id':frame_id}
            sample['depth']=self.generate_depth_map(scene_para,index)
            return sample
        # remove stationary img
        cum_speed=np.zeros(3)
        if self.from_speed:
            for i, speed in enumerate(scene_para['speed']):
                cum_speed+=speed  # Why use the accumulation of speed---if it has enough move, it is ok. eg: 72,75
                speed_mag=np.linalg.norm(cum_speed)
                if(speed_mag>=self.min_speed):
                    frame_id=scene_para['frame_id'][i]
                    yield construct_sample(scene_para,i,frame_id)  # yield for saving memory
                    cum_speed=0
        else:
            for i, frame_id in enumerate(scene_para['frame_id']):
                yield construct_sample(scene_para, i, frame_id)


# test
# data_loader = KittiRawLoader('/home/lw/data/test/kitti_raw_data')
# scene_dir=data_loader.scenes_dir[0]
# train_scene_para=data_loader.get_scene_parameters(scene_dir)
# scene_para=train_scene_para[0]
# cum_speed=np.zeros(3)
#
# for i, speed in enumerate(scene_para['speed']):
#     cum_speed+=speed
#     speed_mag=np.linalg.norm(cum_speed)
#     print(speed_mag)