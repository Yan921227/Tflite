import tensorflow as tf
import numpy as np

# 載入 TFLite 模型
interpreter = tf.lite.Interpreter(model_path="train/weights/best_saved_model/best_float32.tflite")
interpreter.allocate_tensors()

# 獲取輸入和輸出詳情
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=" * 50)
print("✅ TFLite 模型載入成功!")
print("=" * 50)
print("\n📥 輸入資訊:")
print(f"  形狀: {input_details[0]['shape']}")
print(f"  類型: {input_details[0]['dtype']}")
print(f"  名稱: {input_details[0]['name']}")

print("\n📤 輸出資訊:")
print(f"  形狀: {output_details[0]['shape']}")
print(f"  類型: {output_details[0]['dtype']}")
print(f"  名稱: {output_details[0]['name']}")

# 測試推理
print("\n🧪 測試推理...")
test_input = np.random.randn(*input_details[0]['shape']).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"✅ 推理成功! 輸出形狀: {output.shape}")
print("\n🎉 模型完全可用!")
