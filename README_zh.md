# evaluation_metrics 使用说明（中文）

本文件说明当前目录下 8 个脚本如何使用、输入输出是什么、指标含义是什么。

> 说明
> - 这些脚本大多是“可直接运行的示例脚本”（不是封装好的 pip 包接口）。你需要把脚本里的占位路径（`Path_of_...`）替换成你自己的数据路径。
> - 绝大多数指标依赖 `torch` / `torchvision` / `torchmetrics`；`CLIP_Score_cal.py` 还会触发 CLIP 模型下载（首次运行会慢）。
> - 本目录里多处出现 “CMP Images” —— 指 *cubemap*（立方体六面图）格式；“ERP Images” 指 equirectangular panorama（等距矩形全景图）。

---

## 1. 环境与依赖

建议使用 Python 3.9+。

常用依赖（按脚本覆盖面）：

```bash
pip install torch torchvision torchmetrics lightning pillow numpy py360convert
```

如果你的输入是全景视频（`.mp4`），并希望“一键评测”，请额外确保：
- 已安装 `ffmpeg` 且可在命令行直接运行 `ffmpeg`（用于抽帧）

本目录提供了一个面向 `.mp4` 的总控脚本：[video_eval_runner.py](video_eval_runner.py)。

另外：
- [e2c_converter.py](e2c_converter.py) 会调用外部命令 `convert360`（不是 Python 包），需要你自己安装并确保命令在 `PATH` 里可用。
- [FAED_cal.py](FAED_cal.py) 依赖你本地的 FAED 预训练 checkpoint（`.ckpt`）以及工程里的 `modules.AutoEncoder`、`utils.WandbLightningModule`。如果你的仓库里没有这两个模块或 checkpoint，不会跑通。

---

## 2. 数据与目录结构约定

### 2.1 ERP（全景等距矩形）图像目录

用于：一般生成结果统计（如 DS）、或作为转换为 CMP 的输入。

```
/path/to/erp_images/
  0001.png
  0002.jpg
  ...
```

### 2.2 CMP（Cubemap 六面）图像目录

用于：FID/KID/OmniFID/FAED 这些脚本示例里默认处理 CMP。

如果只是“平面 FID/KID”，你也可以直接把任意图片放一个文件夹里；
但 **OmniFID** 明确要求 **六个子目录**（U/F/R/B/L/D）分别存六个面。

```
/path/to/cmp_images/
  U/
    0001_U.png
  F/
    0001_F.png
  R/
    0001_R.png
  B/
    0001_B.png
  L/
    0001_L.png
  D/
    0001_D.png
```

### 2.3 文本 prompt 文件

用于：CLIPScore。

`prompt_file` 是纯文本，每行一个 prompt。脚本按 `sorted()` 后的图片顺序与 prompt 逐行对齐。

```
A photo of a cat.
A panoramic view of a city at sunset.
...
```

重要：
- 图片数量必须与 prompt 行数一致，否则 `zip(images, prompts)` 只会计算最短的一侧，导致你误以为“全算完了”。

---

## 2.4 输入是 .mp4（视频）时怎么评测？

本目录的大多数指标脚本评测对象是“图像”。如果你的结果是全景视频（`.mp4`），通常做法是：
- 先抽帧：把每个视频按固定 FPS 采样成若干帧图像（ERP）
- 再对抽出的帧运行图像指标，并对所有帧/所有视频做平均汇总

为方便使用，本目录提供一键脚本：[video_eval_runner.py](video_eval_runner.py)。

推荐用法（默认按 `--fps` 抽帧，同时算 DS/CLIPScore/IS/FID/KID）：

```bash
python video_eval_runner.py --real_dir D:\\data\\gt_videos --fake_dir D:\\data\\gen_videos --out_dir D:\\data\\_eval_cache --fps 1 --prompt "你的文本提示"
```

如果视频很长、或者视频时长差异很大，可以改用“每个视频均匀采样 N 帧”（避免按 fps 抽出太多帧）：

```bash
python video_eval_runner.py --real_dir D:\\data\\gt_videos --fake_dir D:\\data\\gen_videos --out_dir D:\\data\\_eval_cache --num_frames_per_video 32 --prompt "你的文本提示"
```

说明：
- 设置 `--num_frames_per_video` 后，会按视频时长均匀取帧（用 `ffprobe` 读时长，再用 `ffmpeg -ss` 定位抽帧），并覆盖 `--fps`。
- `--max_frames_per_video` 仍然生效：最终会取 `min(num_frames_per_video, max_frames_per_video)`。

如果你希望“完全公平/最忠实”，并且每个视频帧数固定（例如你说的每个视频都是 81 帧），推荐直接抽取每一帧：

```bash
python video_eval_runner.py --real_dir D:\\data\\gt_videos --fake_dir D:\\data\\gen_videos --out_dir D:\\data\\_eval_cache --every_frame --prompt "你的文本提示"
```

说明：
- `--every_frame` 会覆盖 `--fps/--num_frames_per_video`，直接抽取全部帧（可以再配合 `--max_frames_per_video` 限制上限）。

如果你想“每隔 K 帧抽一帧”（例如每隔 5 帧抽一次）：

```bash
python video_eval_runner.py --real_dir D:\\data\\gt_videos --fake_dir D:\\data\\gen_videos --out_dir D:\\data\\_eval_cache --every_k_frames 5 --prompt "你的文本提示"
```

说明：
- `--every_k_frames 5` 会抽取帧序号 0,5,10,...（按解码后的帧序号）。
- 该模式会覆盖 `--fps/--num_frames_per_video`；如果你之前跑过别的抽帧模式，脚本会自动识别参数变化并重新抽帧。

说明：
- `--real_dir` 可选；如果不提供，就会跳过 FID/KID（因为缺少 GT 分布）。
- `--prompt` 或 `--prompts_file` 二选一（CLIPScore 需要文本）。
- 结果会写入 `out_dir/results.json`。

尺寸相关参数（可选）：
- `--inception_resize 299`：控制 FID/KID/IS（以及 `--omnifid` 的每个面）送入 Inception 的 resize 尺寸。你可以设成 `256` 或 `512`，但注意这会改变数值，和标准 FID（299）不一定可直接横比。
- `--clip_resize 224`：控制 CLIPScore 的 resize 尺寸（默认 224）。

关于 KID：
- KID 有一个限制：`subset_size` 必须小于样本数。若你抽帧太少（例如只有 1~2 帧），一键脚本会自动把 KID 的 `subset_size` 调小，仍不满足则跳过 KID，并把原因写入 `results.json`。
- 想稳定算 KID：提高 `--fps` 或设置更大的 `--max_frames_per_video` / `--num_frames_per_video`。

视频配对规则（fake vs GT）：
- 默认 `--pairing auto`：优先按文件名 stem 配对；如果 stem 不一致但两边视频数量相同，则自动按排序一一配对。
- 如需强制 stem 一致：加 `--pairing stem`
- 如需始终按排序配对：加 `--pairing sorted`

如果你还希望算 OmniFID（需要把每一帧 ERP 转成 CMP 六面并分面算 FID）：

```bash
python video_eval_runner.py --real_dir D:\\data\\gt_videos --fake_dir D:\\data\\gen_videos --out_dir D:\\data\\_eval_cache --fps 1 --prompt "你的文本提示" --omnifid --cmp_face_width 256
```

前置条件：
- 已安装 `convert360` 且命令可用（脚本会调用 `convert360 e2c ...`）
- 已安装 `py360convert`（用于把 dice 布局拆成 U/F/R/B/L/D 六面）

---

## 3. 逐脚本说明

下面按文件名解释其用途、输入输出与指标含义。你可以直接用 `python <script>.py` 方式运行。


### 3.1 CLIP 相关：CLIPScore

文件：[CLIP_Score_cal.py](CLIP_Score_cal.py)

**用途**
- 评估“图片与文本 prompt 的语义匹配程度”。

**输入**
- `image_folder`：图片文件夹（脚本会递归读取 `*.*`）。
- `prompt_file`：文本文件路径，每行一个 prompt。
- 预处理：
  - 图片 resize 到 `(224, 224)`
  - 强制 3 通道（灰度会扩展到 3 通道）
  - 转为 `float32`

**输出**
- 逐样本 CLIPScore（脚本内部 list 收集）
- 最终打印：`Average CLIP Score: xxxx`

**含义（如何解读）**
- 越大越好：表示图像和文字越匹配。
- 该分数依赖所用 CLIP 模型（脚本使用 `openai/clip-vit-large-patch14`），不同模型数值不可直接横比。

**运行方法**
1) 打开脚本，替换：
- `image_folder = "..."`
- `prompt_file = "..."`
2) 运行：

```bash
python CLIP_Score_cal.py
```


### 3.2 几何连续性：Discontinuity Score (DS)

文件：[Discontinuity_Score_cal.py](Discontinuity_Score_cal.py)

**用途**
- 评估全景图在左右边界拼接处的“接缝/断裂”程度。
- 该实现来源于 OmniFID 论文中的 discontinuity score 的非官方实现思路：把图像最左 3 列与最右 3 列拼起来，用 Scharr 边缘算子检测垂直缝。

**输入**
- `folder_path`：包含 RGB 图片的文件夹（脚本用 `os.listdir`，不递归）。
- 每张图片会被 `PIL.Image.open` 读取，再 `TF.to_tensor` 转为 `torch.Tensor`，形状 `(C,H,W)`。
- 约束：图像宽度至少 6 像素（因为要取左右各 3 列）。

**输出**
- 每张图：打印 `Processed xxx: DS = ...`
- 最终：打印平均 DS。

**含义（如何解读）**
- 越小越好：接缝越不明显、左右边界越连续。
- DS 是基于边缘响应的相对比值构造的，数值量纲不直观，推荐在同一数据集上做横向比较。

**运行方法**
1) 替换 `folder_path = "..."`
2) 运行：

```bash
python Discontinuity_Score_cal.py
```


### 3.3 感知分布距离：FID (Frechet Inception Distance)

文件：[FID_cal.py](FID_cal.py)

**用途**
- 衡量“生成图像分布”与“真实图像分布”的差异。
- 使用 InceptionV3 的 2048 维特征，计算两组特征的 Fréchet distance。

**输入**
- `real_folder`：真实图（GT）目录
- `fake_folder`：生成图目录
- 脚本会递归读取 `*.*`，并 resize 到 `(299, 299)`，强制 3 通道。

**输出**
- `FID: <value>`（`torch.Tensor` 标量）

**含义（如何解读）**
- 越小越好：生成分布越接近真实分布。
- 对样本量较敏感；尽量保证两边样本数量足够且相近。

**运行方法**
1) 替换 `real_folder`、`fake_folder`
2) 运行：

```bash
python FID_cal.py
```


### 3.4 多样性/质量：Inception Score (IS)

文件：[IS_cal.py](IS_cal.py)

**用途**
- 只使用生成图像本身，衡量“可被 Inception 模型高置信分类（质量）”与“类别分布多样（多样性）”。

**输入**
- `fake_folder`：生成图目录
- 递归读取 `*.*`，resize 到 `(299, 299)`，强制 3 通道。

**输出**
- `Inception Score: <...>`（torchmetrics 的 `InceptionScore.compute()` 通常返回一个 tuple/tensor，具体显示取决于版本）

**含义（如何解读）**
- 越大通常越好，但它不需要真实图对照；也因此可能被“投机”提高（例如生成某些易分类但不真实的图）。
- 更适合在同一任务/同一数据集上对比不同方法。

**运行方法**
1) 替换 `fake_folder`
2) 运行：

```bash
python IS_cal.py
```


### 3.5 分布距离：KID (Kernel Inception Distance)

文件：[KID_cal.py](KID_cal.py)

**用途**
- 类似 FID，但使用核方法（MMD）评估两组 Inception 特征分布差异。
- 有时在小样本场景比 FID 更稳定。

**输入**
- `real_folder`、`fake_folder`：真实与生成目录
- 递归读取 `*.*`，resize 到 `(299, 299)`，强制 3 通道。
- 关键参数：`subset_size=50`（KID 会做子集采样估计，因此会输出均值与方差）

**输出**
- `kid_mean, kid_std = kid.compute()`
- 脚本打印：
  - `KID Mean: ...`
  - `KID Standard Deviation: ...`

**含义（如何解读）**
- 越小越好。
- `std` 反映采样估计的不确定性；当样本较少时 std 往往更大。

**运行方法**
1) 替换 `real_folder`、`fake_folder`
2) 运行：

```bash
python KID_cal.py
```


### 3.6 全景几何保真：OmniFID

文件：[OmniFID_cal.py](OmniFID_cal.py)

**用途**
- 非官方实现：把全景图拆成 cubemap 六个面，分别算每个面的 FID，再按论文描述做组合，得到 OmniFID。

**输入**
- `real_folder`、`fake_folder`：都要求是 CMP 根目录，并且包含 6 个子目录：`U/F/R/B/L/D`。
- 每个子目录里放对应面的图片。
- 脚本内部会对每个 face 计算一个 FID（`feature=2048`），预处理 resize `(299,299)`。

**输出**
- `Individual FID Scores: {"U":..., "F":..., ...}`
- `Frontal FID: ...`（前向区域，定义为 F/R/B/L 的平均）
- `OmniFID: ...`（按 `(U + frontal + D)/3` 组合）

**含义（如何解读）**
- 越小越好。
- 相比直接在 ERP 上算 FID，这种做法更强调球面几何一致性（因为不同面对应不同视角区域）。

**运行方法**
1) 先确保你已经把 ERP 转成 CMP 六面目录结构（可用本目录的 e2c 脚本，见 3.8）。
2) 替换 `real_folder`、`fake_folder`
3) 运行：

```bash
python OmniFID_cal.py
```


### 3.7 AutoEncoder 特征距离：FAED

文件：[FAED_cal.py](FAED_cal.py)

**用途**
- 通过一个专门为全景训练的 AutoEncoder encoder 提取特征，然后用 FID 公式在特征空间中计算距离。
- 代码来自 PanFusion 的 FAED 思路（脚本开头已有来源说明）。

**输入**
- `real_folder`：GT CMP 图片目录
- `fake_folder`：生成 CMP 图片目录
- 预训练 checkpoint：
  - 代码里写死：
    ```python
    ckpt_path = os.path.join('Path_of_Folder_Containing_the_Pretrained_Checkpoint', 'faed.ckpt')
    ```
  - 你必须把路径改成你本地实际 checkpoint 的位置。
- 运行时的关键参数：`FrechetAutoEncoderDistance(pano_height=512)`

**输出**
- 打印：`FAED: <value>`（`torch.Tensor` 标量）

**含义（如何解读）**
- 越小越好：表示生成与真实在“FAED encoder 特征空间”的分布更接近。
- 与普通 FID 的差异在于：特征提取网络不是 Inception，而是面向全景的 AutoEncoder encoder。

**重要注意事项（很容易踩坑）**
- 本脚本引用了 `modules.AutoEncoder` 和 `utils.WandbLightningModule`。如果你当前 workspace 里没有这些模块，会直接 `ImportError`。
- `load_images_from_folder()` 使用 `torchvision.io.read_image`，读取到的是 `uint8 [0,255]` 张量；`get_activation()` 里会做 `(img/127.5)-1` 归一化到 `[-1,1]`。
- 该实现按 `pano_height` 推断特征维度 `num_features = pano_height * 4`，因此输入图像的分辨率/encoder 输出形状必须和训练时一致。

**运行方法**
1) 配好依赖与 checkpoint 路径
2) 替换 `real_folder`、`fake_folder`
3) 运行：

```bash
python FAED_cal.py
```


### 3.8 ERP → CMP：e2c_converter（转换工具，不是指标）

文件：[e2c_converter.py](e2c_converter.py)

**用途**
- 把 ERP（等距矩形全景）转换为 cubemap（dice 布局），再拆成六个面并按 `F/R/B/L/U/D` 六个子目录保存。
- 这通常是 OmniFID 的前置步骤。

**输入**
- `input_dir`：ERP 图片目录（`png/jpg/jpeg`）
- `output_dir`：输出 CMP 根目录
- `width`：每个面输出的分辨率（脚本传给 `convert360` 的 `--height/--width`）

**输出**
- `output_dir` 下生成六个子目录：`F/R/B/L/U/D`
- 每个输入 ERP 会生成：
  - 一个 cubemap dice 总图：`{base}_cubemap.png`
  - 六张面图：`{base}_{face}.png` 分别放入对应子目录

**依赖/外部要求**
- `pip install py360convert pillow numpy`
- 需要安装 `convert360` 并保证命令行可直接运行 `convert360 e2c ...`

**运行方法**
1) 替换：
- `input_directory = "..."`
- `output_directory = "..."`
2) 运行：

```bash
python e2c_converter.py
```

---

## 4. 常见问题（FAQ）

### Q1：为什么 CLIPScore 很慢/第一次会卡住？
- CLIP 模型可能会首次下载权重；确保网络可访问 HuggingFace/相关源，或提前缓存模型。

### Q2：为什么 FID/KID/IS 报 CUDA 或内存不足？
- 这些 torchmetrics 模块可能会在 GPU 上跑；你可以尝试：
  - 减小一次性加载的图片数量（当前脚本是全量 `torch.stack`）
  - 或者确保在 CPU 上运行（不显式 `.to('cuda')`）

### Q3：OmniFID 的目录结构必须是 U/F/R/B/L/D 吗？
- 当前脚本是按这 6 个目录名写死的，必须匹配。

### Q4：FAED 运行报错找不到 modules/utils？
- 这说明该脚本需要依赖“训练 FAED 的工程代码”。你要把对应代码加入 `PYTHONPATH`，或把缺失模块复制进当前工程，并提供 `faed.ckpt`。

---

## 5. 指标总结（越大越好 vs 越小越好）

- 越大越好：CLIPScore、Inception Score
- 越小越好：DS、FID、KID、OmniFID、FAED
