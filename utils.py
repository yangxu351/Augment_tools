import os
import shutil

def make_dir_if_not_exist(mf, rm=False):
    if not os.path.exists(mf):
        os.makedirs(mf)
    else:
        if rm: # remove folder if exists
            shutil.rmtree(mf)
            os.mkdir(mf)

