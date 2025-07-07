from googletrans import Translator

translator = Translator()

with open("README.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

first_line = lines[0]

text_to_translate = "".join(lines[1:])

result = translator.translate(text_to_translate, src='zh-CN', dest='en')

with open("README_en.md", "w", encoding="utf-8") as f:
    f.write(first_line)        
    f.write(result.text)       
    
print("✅ Translated README.md to README_en.md (skipped first line)")
