from PIL import Image
import glob

# 获取所有 iter*.png 文件，并按文件名排序
plot_path = '/nfs/share/home/zhaoxueyan/dataset_baseline/retrosoc_asic_cx55/workspace/output/dreamplace/result/retrosoc_asic/plot'
image_files = sorted(glob.glob(f'{plot_path}/iter*.png'))

# 统一缩放尺寸
# target_size = (800, 600)

# 打开并缩放所有图片
images = [Image.open(f) for f in image_files]

# 保存为 GIF
images[0].save(
    'output.gif',
    save_all=True,
    append_images=images[1:],
    duration=10,   # 每帧持续时间，单位毫秒
    loop=0          # 0 表示无限循环
)