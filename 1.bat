# 1. 初始化本地仓库
git init

# 2. 【关键】先告诉 Git 忽略哪些文件，防止把 60MB 的大模型传上去把仓库撑爆
# 直接在终端执行这两行，创建一个 .gitignore 文件
echo "bin/" >> .gitignore
echo "models/" >> .gitignore
echo ".idea/" >> .gitignore

# 3. 把代码和 README 全部添加到暂存区
git add .

# 4. 提交第一次代码，写上专业的提交说明
git commit -m "feat: Initial commit of J-Llama Inference Engine"