import numpy as np
import pandas as pd

def _iter_loadxyz(infile, skiprows=0, dtype=float):
    def iter_func():
        for _ in range(skiprows):
            next(infile)
        frame = None
        globalT = None
        for line in infile:
            line = line.strip().split()
            if len(line) == 1:
                try:
                    num_parts = int(line[0])
                except ValueError:
                    print("H")
                    print(line)


                if frame:
                    if len(frame) == num_parts:
                        yield globalT, frame
                    else:
                        print ("Frame {0} defect.".format(globalT))
                        yield globalT, [[None]*3]*num_parts
                frame = []
                try:
                    line = next(infile)
                    globalT = float(line.strip().split()[-1])
                except (IndexError, ValueError):
                    if globalT is None:
                        globalT = 0
                    else:
                        globalT += 1 ##None
                continue
            vec = []
            for item in line[1:]:
                vec.append(dtype(item))
            frame.append(vec)
        yield globalT, frame
    return iter_func()

def load_xyz(filename):
    data_xyz = _iter_loadxyz(filename)
    tsteps, frames = zip(*data_xyz)
    frames = np.array(frames)
    frames = frames.swapaxes(1,2).reshape(-1,frames.shape[1])
    idx = pd.MultiIndex.from_product([tsteps,[0,1,2]], names=['time', 'coordinate'])
    data = pd.DataFrame(frames, index=idx)
    return data

def convert_xyz_df_coord_index_to_column(data):
    df = data.stack().unstack(level=1)
    df.columns = ["X", "Y", "Z"]
    return df

