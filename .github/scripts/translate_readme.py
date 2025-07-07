from googletrans import Translator

# 创建翻译器
translator = Translator()

# 读取中文 README.md
with open("README.md", "r", encoding="utf-8") as f:
    text = f.read()

# 翻译成英文
result = translator.translate(text, src='zh-CN', dest='en')

# 写入英文 README_en.md
with open("README_en.md", "w", encoding="utf-8") as f:
    f.write(result.text)

print("✅ Translated README.md to README_en.md")
