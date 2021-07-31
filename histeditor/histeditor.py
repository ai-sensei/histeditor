import numpy as np


def make_dataxy(data, data_len, pre_len, ydata=True):
    x, y = [], []
    for i in range(len(data) - (pre_len + data_len)):
        latest = i + data_len
        x.append(list(data[i:latest]))
        if ydata:
            y.append(data[latest + pre_len] / data[latest] - 1)

    x = np.array(x, dtype='float32')

    if not ydata:
        return x
    else:
        y = np.array(y, dtype='float32').reshape(len(y), 1)

        return x, y

def normalize(data):
    data_dim = len(data.shape)
    if data_dim == 2:
        dim1 = data.shape[0]
        mx = np.max(data, axis=1).reshape(dim1, 1)
        mn = np.min(data, axis=1).reshape(dim1, 1)

    else:
        dim1, dim2, dim3 = data.shape[0], data.shape[1], data.shape[2]
        mx = np.max(data.reshape(dim1, dim2 * dim3), axis=1).reshape(dim1, 1, 1)
        mn = np.min(data.reshape(dim1, dim2 * dim3), axis=1).reshape(dim1, 1, 1)

    return (data - mn) / (mx - mn) + 1


class HistEditor:
    def __init__(self, hist, minutes, data_min, pre_min):
        self.origin = hist
        self.minutes = minutes
        self.data_min = data_min
        self.pre_min = pre_min

    def make_dataset1d(self, norm=False, dtype='float32'):
        hist = np.array(self.origin)[self.minutes - 1:][::self.minutes]
        data_len = int(self.data_min / self.minutes)
        pre_len = int(self.pre_min / self.minutes)

        data_x, data_y = make_dataxy(hist, data_len, pre_len)

        if norm:
            return normalize(data_x).astype(dtype), data_y.astype(dtype)
        else:
            return data_x.astype(dtype), data_y.astype(dtype)

    def make_dataset1d_2(self, norm=False, dtype='float32'):
        hist = np.array(self.origin)[self.minutes - 1:][::self.minutes]
        data_len = int(self.data_min / self.minutes)
        pre_len = int(self.pre_min / self.minutes)

        data_x = make_dataxy(hist, data_len, pre_len, ydata=False)
        data_y = make_dataxy(hist[data_len:], pre_len, 0, ydata=False)

        if norm:
            return normalize(data_x).astype(dtype), normalize(data_y).astype(dtype)
        else:
            return data_x.astype(dtype), data_y.astype(dtype)

    def make_dataset2d(self, norm=False, dtype='float32'):
        hist = self.origin
        data_len = int(self.data_min / self.minutes)
        pre_len = int(self.pre_min / self.minutes)

        dd = []
        for i in range(int(len(hist) / self.minutes)):
            dd.append(hist[i * self.minutes:i * self.minutes + self.minutes])
        dd = np.array(dd)

        st = dd[:, 0].reshape(len(dd), 1)
        ed = dd[:, -1].reshape(len(dd), 1)
        mx = np.max(dd, axis=1).reshape(len(dd), 1)
        mn = np.min(dd, axis=1).reshape(len(dd), 1)

        data_x = np.concatenate([mx, ed, st, mn], axis=1)

        rolled = np.roll(data_x[:, 1], -pre_len)
        data_y = (rolled / data_x[:, 1]).reshape(len(data_x), 1)[:-pre_len] - 1
        data_x = data_x[:-pre_len]

        ddd = []
        for i in range(len(data_x) - data_len):
            ddd.append(data_x[i:i + data_len])
        data_x = np.array(ddd)

        if norm:
            return normalize(data_x).astype(dtype), data_y[data_len:].astype(dtype)
        else:
            return data_x.astype(dtype), data_y[data_len:].astype(dtype)
