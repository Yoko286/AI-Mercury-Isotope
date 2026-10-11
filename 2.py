import matplotlib.font_manager as fm

# 列出所有包含中文支持关键词的字体
fonts = [f.name for f in fm.fontManager.ttflist 
         if any(k in f.name for k in ['Hei', 'Song', 'Kai', 'YaHei', 'SimSun', 'SimHei', 'FangSong', 'Ming'])]

# 去重
fonts = sorted(set(fonts))
print("可用的中文字体：")
for f in fonts:
    print(f)