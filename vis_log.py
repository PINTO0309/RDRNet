import json
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    """単純移動平均を返す"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

log_file =  './20250303_152611.json'

res = []
decoder = json.JSONDecoder()
with open(log_file, 'r') as f:
    line = f.readline()
    while line:
        res.append(decoder.raw_decode(line))
        line = f.readline()

x_epoch_data = []
loss = []
loss_context = []
loss_spatial = []
acc_seg = []

x_epoch_data_2 = []
mIou = []
mAcc = []

for i in range(len(res)-1):
    if 'lr' in res[i+1][0]:
        x_epoch_data.append(res[i+1][0]['iter'])
        loss.append(res[i+1][0]['loss'])
        loss_context.append(res[i+1][0]['decode.loss_context'])
        loss_spatial.append(res[i+1][0]['decode.loss_spatial'])
        acc_seg.append(res[i+1][0]['decode.acc_seg'])

    elif 'aAcc' in res[i+1][0]:
        x_epoch_data_2.append(res[i][0]['iter'])
        mIou.append(res[i+1][0]['mIoU'])
        mAcc.append(res[i+1][0]['mAcc'])

    else:
        pass

# ====== 移動平均線の計算 ======
window_size = 100
ma_loss         = moving_average(loss,         window_size)
ma_loss_context = moving_average(loss_context, window_size)
ma_loss_spatial = moving_average(loss_spatial, window_size)
ma_acc_seg      = moving_average(acc_seg,      window_size)
x_epoch_data_ma = x_epoch_data[window_size-1:]

window_size = 5
ma_mIou = moving_average(mIou, window_size)
ma_mAcc = moving_average(mAcc, window_size)
x_epoch_data_2_ma = x_epoch_data_2[window_size-1:]

# ====== グラフ描画 ======
# fig = plt.figure(figsize=(14, 5))
fig = plt.figure(figsize=(18, 8))

# ----------------------
# 左: loss/acc_seg
# ----------------------
ax1 = fig.add_subplot(1, 2, 1)
ax2 = ax1.twinx()

# 元データ
line_loss,         = ax2.plot(x_epoch_data, loss, label='loss', color='red')
line_loss_context, = ax2.plot(x_epoch_data, loss_context, label='loss_context', color='blue')
line_loss_spatial, = ax2.plot(x_epoch_data, loss_spatial, label='loss_spatial', color='green')
line_acc_seg,      = ax1.plot(x_epoch_data, acc_seg, label='acc_seg', color='yellow')

# 移動平均データ（オリジナルと区別するため色を変える）
line_ma_loss,         = ax2.plot(x_epoch_data_ma, ma_loss, label='loss(MA)', color='purple')#, linestyle='--')
line_ma_loss_context, = ax2.plot(x_epoch_data_ma, ma_loss_context, label='loss_context(MA)', color='cyan')#, linestyle='--')
line_ma_loss_spatial, = ax2.plot(x_epoch_data_ma, ma_loss_spatial, label='loss_spatial(MA)', color='orange')#, linestyle='--')
line_ma_acc_seg,      = ax1.plot(x_epoch_data_ma, ma_acc_seg, label='acc_seg(MA)', color='black')#, linestyle='--')

ax1.set_title("loss/acc_seg")
ax1.set_xlabel('epoch')
ax1.set_ylabel('acc_seg')
ax2.set_ylabel('loss')
ax1.set_ylim(0, 100)

# 凡例をまとめて表示
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', bbox_to_anchor=(-0.34, 1))

# ----------------------
# 右: mIoU/mAcc
# ----------------------
ax3 = fig.add_subplot(1, 2, 2)
line_mIou, = ax3.plot(x_epoch_data_2, mIou, label='mIoU', color='blue')
line_mAcc, = ax3.plot(x_epoch_data_2, mAcc, label='mAcc', color='green')

# 移動平均データ（別の色）
line_ma_mIou, = ax3.plot(x_epoch_data_2_ma, ma_mIou, label='mIoU(MA)', color='magenta', linestyle='--')
line_ma_mAcc, = ax3.plot(x_epoch_data_2_ma, ma_mAcc, label='mAcc(MA)', color='black', linestyle='--')

ax3.set_title("mIoU/mAcc")
ax3.set_xlabel('epoch')
ax3.set_ylabel('mIoU/mAcc')
ax3.set_ylim(0, 100)
ax3.legend(loc='upper left')

plt.show()
