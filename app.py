from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from PIL import Image
import torch
from ultralytics import YOLO
import io
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BoundingBox(BaseModel):
    x_min: int = Field(..., ge=0)
    y_min: int = Field(..., ge=0)
    x_max: int = Field(..., ge=0)
    y_max: int = Field(..., ge=0)

class Detection(BaseModel):
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    detections: List[Detection]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

app = FastAPI(
    title="T-Bank Logo Detector API",
    description="REST API для детекции логотипа Т-Банка на изображениях",
    version="1.0.0"
)

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        logging.info("Загрузка модели YOLOv8...")
        model = YOLO("models/best.pt")
        
        # Прогрев модели (опционально)
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy_img, verbose=False)
        
        logging.info("✅ Модель успешно загружена и прогрета.")
        yield  # Здесь приложение "живет" и обрабатывает запросы
    finally:
        # Очистка ресурсов при завершении (опционально, но правильно)
        logging.info("Очистка ресурсов...")
        model = None  # или вызов model.clear() если поддерживается
        torch.cuda.empty_cache()
        logging.info("✅ Ресурсы освобождены.")


app = FastAPI(
    title="T-Bank Logo Detector API",
    description="Сервис детекции логотипа Т-Банка",
    version="1.0.0",
    lifespan=lifespan  # <-- Подключаем lifespan
)

SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

@app.post("/detect", response_model=DetectionResponse)
async def detect_logo(file: UploadFile = File(...)):

    if file.content_type not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат изображения. Поддерживаемые: {SUPPORTED_FORMATS}"
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Пустой файл")

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        results = model.predict(
            source=image_np,
            conf=0.25,
            iou=0.45,
            device="0" if torch.cuda.is_available() else "cpu",
            verbose=False
        )

        detections = []
        for box in results[0].boxes.xyxy.cpu().numpy():
            x_min, y_min, x_max, y_max = map(int, box[:4])

            if x_max <= x_min or y_max <= y_min:
                continue
            detections.append(
                Detection(
                    bbox=BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max
                    )
                )
            )

        logger.info(f"Обнаружено {len(detections)} логотипов на изображении {file.filename}")

        return DetectionResponse(detections=detections)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Ошибка при детекции логотипа"
        )