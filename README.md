# DNFM Server

## 功能说明

1. **技能坐标配置**
   - 将技能坐标配置为所有键位的像素坐标。
   - 使用手机截屏功能，获取屏幕对应像素位置，并将其配置为对应技能的坐标。

2. **技能映射**
   - 在 `hero` 文件夹下的 `.json` 文件中，将技能名称与 `skill` 文件夹内的技能对应。

3. **控制算法**
   - `run` 按钮：启动控制算法。
   - `stop` 按钮：停止控制算法。
   - `reset` 按钮：重启控制算法的线程。

4. **FPS 调整**
   - 在 `main.py` 文件的第26行找到 `fpsmax` 参数，调整其值到 5 到 30 之间。
   - 较低的 FPS 值会提高安全性，较高的 FPS 值会提高稳定性。

5. **英雄位置检测**
   - 如果英雄位置检测不准确，可以使用黑钻位置计算英雄位置。

6. **触控功能**
   - 仅支持点击屏幕，请先暂停算法
## 项目启动步骤

1. 确保正确配置技能对应的坐标和技能映射。
2. 调整 FPS 设置以适应你的需求。
3. 使用控制按钮运行或停止算法，必要时重启线程。

## 贡献

欢迎贡献新功能和优化代码，提交 PR 和 Issues 一起改进项目！

