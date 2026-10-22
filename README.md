# Image-Generation-Group-14

人脸匿名化 V2：以非身份条件生成图作为对照基线，使用冻结的预训练网络进行联合潜码优化。

**模型权重体积较大，本仓库不上传预训练权重。下载代码后，须先按下表补齐六个文件才能运行。** 本地完整模型包中的同名文件也可直接复制到对应位置。

## 1. 安装环境

已验证环境：Python 3.13、PyTorch 2.8.0、torchvision 0.23.0，Windows + NVIDIA GPU（6 GB 显存）。

```powershell
git clone https://github.com/toxcyww/Image-Generation-Group-14.git
cd Image-Generation-Group-14
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

## 2. 放置模型权重

所有保存路径均相对于仓库根目录；缺少的文件夹请自行创建。下载链接中的文件名可能与保存路径不同，请按下表保存。

| 权重 | 下载来源 | 保存路径 |
| --- | --- | --- |
| e4e FFHQ（含配套 StyleGAN2 生成器） | [e4e 权重镜像](https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/47855f8ea72b02294649385211a1ae1fd7dcb58e/pretrained_models/encoder4editing/e4e_ffhq_encode.pt) | `models/e4e_ffhq_encode.pt` |
| SimSwap ArcFace | [官方预训练模型说明](https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md#pretrained-model)，下载其中的 `arcface_checkpoint.tar` | `models/arcface_checkpoint.tar` |
| YuNet（2023mar） | [OpenCV Zoo 权重](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)，保存时重命名为 `yunet.onnx` | `models/yunet.onnx` |
| 3DDFA-V2 MobileNet | [官方权重](https://raw.githubusercontent.com/cleardusk/3DDFA_V2/master/weights/mb1_120x120.pth) | `third_party/tddfa/weights/mb1_120x120.pth` |
| LPIPS SqueezeNet 标定权重 | [官方权重](https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/082bb24f84c091ea94de2867d34c4544f68e0963/lpips/weights/v0.1/squeeze.pth) | `third_party/lpips/weights/v0.1/squeeze.pth` |
| SqueezeNet 1.1 主干 | [PyTorch 官方权重](https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth) | `torch_home/hub/checkpoints/squeezenet1_1-b8a52dc0.pth` |

程序不会自动下载这些权重。3DDFA 的小型归一化配置文件已随代码提供，无需另行下载。第三方代码及许可证见 [third_party/NOTICES.txt](third_party/NOTICES.txt)。

## 3. 运行

输入须为清晰、方形的人脸裁剪图；输出目录须使用尚不存在的新目录。

处理一张图片：

```powershell
python -B infer.py --input "人脸.png" --output "outputs/result_01" --device cuda
```

批量处理图片文件夹：

```powershell
python -B infer.py --input "人脸文件夹" --output "outputs/batch_01" --device cuda
```

支持 PNG、JPG、JPEG、BMP、WebP。批量输入的文件名去掉扩展名后须互不重复；结果按图片名称分文件夹保存。

每张图依次完成联合优化和非身份条件基线生成，各 300 步。所有网络权重冻结，仅优化潜码；联合优化仍从原图反演潜码开始。

运行结束后查看输出目录：

- `final.png`：联合优化后的最终图像。
- `nonid_baseline.png`：非身份条件基线。
- `target.png`：自动筛选的监督图。
- `comparison.jpg`：从左到右为原图、非身份条件基线、监督图、联合优化结果。
