#-*-coding:utf-8-*- 
"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 2019/5/15 上午10:03 
  description: Transform kitti raw data to ideal training formation
"""
import argparse
from tqdm import tqdm
from path import Path
import numpy as np
from scipy import misc
import time
from pebble import ProcessPool  # Allows to schedule jobs within a Pool of Processes.

from kitti_raw_data_loader import KittiRawLoader

parser=argparse.ArgumentParser()
parser.add_argument('dataset_dir',help='path to original data')
parser.add_argument('--img_width',type=int,default=416,help='the width of image')
parser.add_argument('--img_height',type=int,default=128,help='the height of image')
parser.add_argument('--min_speed',type=int,default=2,help='minus speed for cleaning data')
parser.add_argument('--num_threads',type=int,default=4,help='the number of threads to work')
parser.add_argument('--from_speed',action='store_true',help='whether clean data')
parser.add_argument('--dump_root',type=str,default='./dump',help='path to dump data')
args=parser.parse_args()


# dump one scene data including imgages and depth maps ->2 scenes(left and right)
def dump_new_scene(args,scene_dir):
    scene_para_list=data_loader.get_scene_parameters(scene_dir)
    for scene_para in scene_para_list:
        dump_path=args.dump_root/scene_para['rel_path']
        dump_path.mkdir_p()
        intrinsics=scene_para['intrinsics']
        np.savetxt(dump_path/'cam.txt',intrinsics)
        for sample in data_loader.get_scene_samples(scene_para):
            img,depth,frame_id=sample['img'],sample['depth'],sample['frame_id']
            img_dump_file=dump_path/'{}.jpg'.format(frame_id)
            misc.imsave(img_dump_file,img)
            depth_dump_file=dump_path/'{}.npy'.format(frame_id)
            np.save(depth_dump_file,depth)
        if len(dump_path.files('*.jpg'))<3:
            dump_path.rmtree()


def main():
    args.dump_root=Path(args.dump_root)
    args.dump_root.mkdir_p()
    # if data have been exist, remove it
    dirs=args.dump_root.dirs()
    if len(dirs)>0:
        print('Remove existing data')
        for dir in dirs:
            dir=Path(dir)
            dir.rmtree()
    global data_loader
    data_loader=KittiRawLoader(args.dataset_dir,
                               img_height=args.img_height,
                               img_width=args.img_width,
                               min_speed=args.min_speed,
                               from_speed=args.from_speed)
    n_scenes=len(data_loader.scenes_dirs)
    print('Found {} potential scenes'.format(n_scenes))

    # current_time=time.time()
    # for scene_dir in tqdm(data_loader.scenes_dirs):
    #     dump_new_scene(args,scene_dir)
    # end_time=time.time()
    # cost_time=end_time-current_time
    # print('Single threads cost time:{:.4f}s'.format(cost_time))
    # dirs=args.dump_root.dirs()
    # if len(dirs)>0:
    #     print('Remove existing data')
    #     for dir in dirs:
    #         dir=Path(dir)
    #         dir.rmtree()

    current_time = time.time()
    with ProcessPool(max_tasks=args.num_threads) as pool:
        tasks=pool.map(dump_new_scene,[args]*n_scenes,data_loader.scenes_dirs)
        try:
            for _ in tqdm(tasks.result(),total=n_scenes):
                pass
        except KeyboardInterrupt as e:
            tasks.cancel()
            raise e
    end_time = time.time()
    cost_time = end_time - current_time
    print('Multi threads cost time:{:.4f}s'.format(cost_time))
    print('Finish dump transformed data')

    # Generate train and validation files
    np.random.seed(8946)  # make sure the random is equal everytime
    scene_dirs=args.dump_root.dirs() # all scene directories
    scene_dirs_prefix=set([scene_dir.basename()[:-2] for scene_dir in scene_dirs])  # get all dir basename (combine left and right)
    with open(args.dump_root/'train.txt','w') as ft:
        with open(args.dump_root/'val.txt','w') as fv:
            for pr in scene_dirs_prefix:
                correspond_dirs=args.dump_root.dirs('{}*'.format(pr))  # if basename is same, dump into the same txt
                if np.random.random() < 0.3:
                    for s in correspond_dirs:
                        fv.write('{}\n'.format(s.name))
                else:
                    for s in correspond_dirs:
                        ft.write('{}\n'.format(s.name))


if __name__=='__main__':
    main()
