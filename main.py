import os
from gradio_client import Client, handle_file

output_folder = "first"
output_folder = os.path.join(os.getcwd(), output_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

ref_audio_path = 'ref.ogg'

text_file = "text_file.txt"  # 替换为你的文本文件路径
# 读取文本文件并按行分割
with open(text_file, 'r', encoding='utf-8') as file:
    text_lines = file.readlines()

client = Client("http://localhost:9872/")

# 遍历每一行文本并生成相应的 WAV 文件
for index, text in enumerate(text_lines):
    text = text.strip()  # 去掉每行的换行符

    if text:  # 确保文本不为空
        # 调用 Gradio 的预测接口生成音频
        result = client.predict(
            text=text,
            text_lang="日文",
            ref_audio_path=handle_file(ref_audio_path),
            aux_ref_audio_paths=[],
            prompt_text="",
            prompt_lang="日文",
            top_k=5,
            top_p=1,
            temperature=1,
            text_split_method="凑四句一切",
            batch_size=20,
            speed_factor=1,
            ref_text_free=False,
            split_bucket=True,
            fragment_interval=0.3,
            seed=-1,
            keep_random=True,
            parallel_infer=True,
            repetition_penalty=1.35,
            api_name="/inference"
        )

        # 假设返回的结果是文件路径，保存到目标文件夹
        output_file_path = os.path.join(output_folder, f"result_{index+1}.wav")
        input_file_path = result[0]  # 获取文件路径

        with open(input_file_path, 'rb') as input_file:
            with open(output_file_path, 'wb') as output_file:
                output_file.write(input_file.read())

        print(f"File {index+1} saved to {output_file_path}")