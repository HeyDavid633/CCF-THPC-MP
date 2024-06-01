# CCF-THPC-MP Memo
> 为投稿 CCF-THPC 主题“混合精度”的代码准备

确定环境搭建的基本配置：
1. PyTorch或TensorFlow --- 具体到版本
2. Git仓库的相关构建 ✅
3. 开发的基本模式： 全部Docker化，测试环境统一纯净
4. 测试环境确定：4080laptop检验正确与否 --- 最终测试平台4090/A100

## 备注
+ TensorCore的使用与否：通过ncu进行确定性判断
