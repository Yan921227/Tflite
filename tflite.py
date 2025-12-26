from ultralytics import YOLO

# 載入模型
model = YOLO("train4\\weights\\best.pt")

# 嘗試轉換為 TFLite (使用 TensorFlow Lite 轉換)
try:
    model.export(format='tflite', imgsz=640)
    print("✅ TFLite 轉換完成！")
except Exception as e:
    print(f"❌ TFLite 轉換失敗: {e}")
    print("\n嘗試使用 saved_model 格式...")
    try:
        model.export(format='saved_model', imgsz=640)
        print("✅ SavedModel 轉換完成！可手動轉換為 TFLite")
    except Exception as e2:
        print(f"❌ SavedModel 也失敗: {e2}")