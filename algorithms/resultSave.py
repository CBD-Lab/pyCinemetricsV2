import os
import csv
import matplotlib.pyplot as plt

# 输入图片数据, 可以讲数据绘制为图像, 和存到CSV
class Resultsave:
    def __init__(self, image_save_path):
        self.image_save_path = image_save_path

    # 其他方法保持不变...

    def diff_csv(self, diff, shot_len):
        try:
            # 检查 shot_len 是否为空
            if not shot_len:
                print("Warning: Empty shot_len data, cannot save CSV")
                return
                
            # 确保目录存在
            os.makedirs(self.image_save_path, exist_ok=True)
                
            # 确定文件路径
            shotlen_csv_path = os.path.join(self.image_save_path, "shotlength.csv")

            # 如果文件已经存在，则删除它
            if os.path.exists(shotlen_csv_path):
                os.remove(shotlen_csv_path)

            # 然后再创建新文件
            with open(shotlen_csv_path, "w+", newline='') as shotlen_csv:
                name2 = ['start', 'end', 'length']
                writer = csv.writer(shotlen_csv)
                writer.writerow(name2)
                for i in range(len(shot_len)):
                    writer.writerow(shot_len[i])
                    
            print(f"Shot length CSV saved to {shotlen_csv_path}")
        except Exception as e:
            print(f"Error in diff_csv: {e}")

    def plot_transnet_shotcut(self, shot_len):
        try:
            # 检查 shot_len 是否为空
            if not shot_len:
                print("Warning: Empty shot_len data, cannot plot")
                return
                
            shot_id = [i for i in range(len(shot_len))]
            shot_length = [shot_len[i][2] for i in range(len(shot_len))]
            
            # 清除之前的图形
            plt.clf()
            plt.close('all')
            
            # 创建一个黑底的图形
            fig = plt.figure(figsize=(8, 6))
            plt.style.use('dark_background')
            plt.bar(shot_id, shot_length, color='blue')
            plt.title('shot length', color="white")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.image_save_path), exist_ok=True)
            
            # 保存图像
            output_path = os.path.join(self.image_save_path, 'shotlength.png')
            plt.savefig(output_path)
            plt.close(fig)  # 确保图形被关闭
            
            print(f"Shot length plot saved to {output_path}")
        except Exception as e:
            print(f"Error in plot_transnet_shotcut: {e}")
            # 确保所有图形都被关闭
            plt.close('all')

    def color_csv(self, colors, colorsC):
        csv_file = open(os.path.join(self.image_save_path, "colors.csv"), "w+", newline='')
        name = ['FrameId']
        for i in range(colorsC):
            name.append("Color" + str(i))

        try:
            writer = csv.writer(csv_file)
            writer.writerow(name)
            # 遍历所有分镜帧
            for i in range(len(colors)):
                # datarow为照片帧号
                datarow = [colors[i][0][5:-4]]
                for j in range(colorsC):
                    datarow.append(colors[i][1][j])
                writer.writerow(datarow)
        finally:
            csv_file.close()

    def plot_scatter_3d(self, all_colors):
        plt.clf()  # 清除之前的图像
        plt.style.use('dark_background')  # 设置图表的背景为黑色

        movie_colors = []
        
        # 合并嵌套的颜色列表
        for i in all_colors:
            i = list(i)
            for j in i:
                movie_colors.append(list(j))

        # 用于存储 RGB 坐标和颜色
        x, y, z, dot_color = [], [], [], []

        # 分离 R, G, B，并转换为 [0, 1] 范围内的颜色
        for c in movie_colors:
            x.append(c[0])  # 红色通道
            y.append(c[1])  # 绿色通道
            z.append(c[2])  # 蓝色通道
            dot_color.append([c[0] / 255, c[1] / 255, c[2] / 255, 1])  # 标准化颜色并添加透明度

        # 创建一个 3D 图
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # 绘制 3D 散点图
        ax.scatter(x, y, z, c=dot_color)

        # 设置标题
        plt.title("3D Color Analysis")

        # 设置轴标签
        ax.set_xlabel('Red (R)')   # X轴表示红色通道
        ax.set_ylabel('Green (G)') # Y轴表示绿色通道
        ax.set_zlabel('Blue (B)')  # Z轴表示蓝色通道

        # 设置窗口标题
        fig.canvas.manager.set_window_title('Color Scatter')

        # 保存图像到指定路径
        plt.savefig(os.path.join(self.image_save_path, 'colors.png'))
