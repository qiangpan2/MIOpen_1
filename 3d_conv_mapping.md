# MIOpenDriver 3D Convolution 参数详解

## 命令行参数解释

```bash
MIOpenDriver convfp16 -n 1 -c 16 --in_d 5 -H 104 -W 60 -k 16 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F 1 -t 1
```

这是 MIOpen 库中的一个测试驱动命令，用于测试 3D 卷积操作（`convfp16` 表示使用半精度浮点数）。

## 各参数详细解释

| 参数 | 含义 |
|------|------|
| `MIOpenDriver` | MIOpen 库的测试驱动程序 |
| `convfp16` | 表示执行 3D 卷积操作，使用半精度浮点数（FP16） |
| `-n 1` | 输入 batch 大小为 1 |
| `-c 16` | 输入通道数为 16 |
| `--in_d 5` | 输入数据的深度（D）为 5 |
| `-H 104` | 输入数据的高度（H）为 104 |
| `-W 60` | 输入数据的宽度（W）为 60 |
| `-k 16` | 输出通道数（滤波器数量）为 16 |
| `--fil_d 1` | 滤波器的深度（D）为 1 |
| `-y 1` | 滤波器的高度（Y）为 1 |
| `-x 1` | 滤波器的宽度（X）为 1 |
| `--pad_d 0` | 深度方向的填充（padding）为 0 |
| `-p 0` | 高度方向的填充（padding）为 0 |
| `-q 0` | 宽度方向的填充（padding）为 0 |
| `--conv_stride_d 1` | 深度方向的卷积步长为 1 |
| `-u 1` | 高度方向的卷积步长为 1 |
| `-v 1` | 宽度方向的卷积步长为 1 |
| `--dilation_d 1` | 深度方向的扩张（dilation）为 1 |
| `-l 1` | 高度方向的扩张（dilation）为 1 |
| `-j 1` | 宽度方向的扩张（dilation）为 1 |
| `--spatial_dim 3` | 空间维度为 3（表示 3D 卷积） |
| `--in_layout NDHWC` | 输入数据的内存布局为 NDHWC（N: batch, D: depth, H: height, W: width, C: channels） |
| `--fil_layout NDHWC` | 滤波器的内存布局为 NDHWC |
| `--out_layout NDHWC` | 输出数据的内存布局为 NDHWC |
| `-m conv` | 运算模式为卷积 |
| `-g 1` | 组卷积的组数为 1（即普通卷积） |
| `-F 1` | 前向传播模式 |
| `-t 1` | 测试模式（启用某些优化） |

## 总结

这是一个测试 3D 卷积操作的命令，使用 FP16 精度：
- 输入数据维度为 `1x5x104x60x16`（NxDxHxWxC）
- 滤波器维度为 `1x1x1x16x16`（DxYxXxKxC）
- 无填充，步长为 1，无扩张，3D 卷积

## 与 CK Tile 示例代码的参数对应关系

| MIOpenDriver 参数 | CK Tile 参数 | 值 |
|-------------------|--------------|----|
| `-n 1` | `--n 1` | 1 (batch size) |
| `-c 16` | `--c 16` | 16 (input channels) |
| `--in_d 5` | `--d 5` | 5 (input depth) |
| `-H 104` | `--h 104` | 104 (input height) |
| `-W 60` | `--w 60` | 60 (input width) |
| `-k 16` | `--k 16` | 16 (output channels) |
| `--fil_d 1` | `--z 1` | 1 (filter depth) |
| `-y 1` | `--y 1` | 1 (filter height) |
| `-x 1` | `--x 1` | 1 (filter width) |
| `--pad_d 0` | `--lpad_d 0 --rpad_d 0` | 0 (depth padding) |
| `-p 0` | `--lpad_h 0 --rpad_h 0` | 0 (height padding) |
| `-q 0` | `--lpad_w 0 --rpad_w 0` | 0 (width padding) |
| `--conv_stride_d 1` | `--stride_d 1` | 1 (depth stride) |
| `-u 1` | `--stride_h 1` | 1 (height stride) |
| `-v 1` | `--stride_w 1` | 1 (width stride) |
| `--dilation_d 1` | `--dilation_d 1` | 1 (depth dilation) |
| `-l 1` | `--dilation_h 1` | 1 (height dilation) |
| `-j 1` | `--dilation_w 1` | 1 (width dilation) |
| `-g 1` | `--g 1` | 1 (group count) |
| `--in_layout NDHWC` | `--in_layout NDHWGC` | NDHWC -> NDHWGC (注意布局差异) |
| `--fil_layout NDHWC` | `--wei_layout GKZYXC` | NDHWC -> GKZYXC (注意布局差异) |
| `--out_layout NDHWC` | `--out_layout NDHWGK` | NDHWC -> NDHWGK (注意布局差异) |

## 运行 CK Tile 示例代码

以下为一组use case的mapping 

```bash

./MIOpenDriver convfp16 -n 1 -c 16 --in_d 5 -H 104 -W 60 -k 16 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F 1 -t 1

./bin/tile_example_grouped_conv_fwd -n=1 -c=16 -d=5 -h=104 -w=60 -k=16 -z=1 -y=1 -x=1 -stride_d=1 -stride_h=1 -stride_w=1 -dilation_d=1 -dilation_h=1 -dilation_w=1 -lpad_d=0 -lpad_h=0 -lpad_w=0 -rpad_d=0 -rpad_h=0 -rpad_w=0 -g=1 -in_layout=NDHWGC -wei_layout=GKZYXC -out_layout=NDHWGK -prec=fp16 -v=1

```