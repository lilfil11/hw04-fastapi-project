from fastapi import FastAPI, Path, UploadFile
from typing import Annotated
from datetime import datetime

import pickle
import numpy as np
import time

from sklearn.decomposition import PCA
from PIL import Image, ImageOps

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis


app = FastAPI(title='HSE year project app')


# Дата и время старта работы сервиса
started_at = datetime.now().strftime("%Y-%m-%d %H:%M")

# Список возможных болезней (на них обучалась модель)
diseases = {0: 'Acne and Rosacea', 1: 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 2: 'Atopic Dermatitis',
            3: 'Bullous Disease', 4: 'Cellulitis Impetigo and other Bacterial Infections', 5: 'Eczema', 6: 'Exanthems and Drug Eruptions',
            7: 'Hair Loss Photos Alopecia and other Hair Diseases', 8: 'Herpes HPV and other STDs', 9: 'Light Diseases and Disorders of Pigmentation',
            10: 'Lupus and other Connective Tissue diseases', 11: 'Melanoma Skin Cancer Nevi and Moles', 12: 'Nail Fungus and other Nail Disease',
            13: 'Poison Ivy Photos and other Contact Dermatitis', 14: 'Psoriasis pictures Lichen Planus and related diseases',
            15: 'Scabies Lyme Disease and other Infestations and Bites', 16: 'Seborrheic Keratoses and other Benign Tumors', 17: 'Systemic Disease',
            18: 'Tinea Ringworm Candidiasis and other Fungal Infections', 19: 'Urticaria Hives', 20: 'Vascular Tumors', 21: 'Vasculitis',
            22: 'Warts Molluscum and other Viral Infections'}

# Количество загруженных для предиктов изображений (для статистики)
images_loaded = 0

# Список с оценками пользователей
review = []

# Модель
model = pickle.load(open('model.pkl', 'rb'))


# Корневая директория
@app.get('/')
def root():
    return {'status': 'successful',
            'message': 'Hello! This is a web-site for classifying images with '
                       'skin diseases using machine learning models.'}


# Получение списка всех болезней
@app.get('/diseases/all')
def get_diseases() -> dict:
    if diseases:
        return {'status': 'successful',
                'data': diseases}
    else:
        return {'status': 'failed',
                'message': 'The list of diseases is empty!'}


# Получение названия болезни по его ID
@app.get('/diseases/{disease_id}')
@cache(expire=30)
def get_disease_name(disease_id: int) -> dict:
    time.sleep(2)
    if disease_id in diseases:
        return {'status': 'successful',
                'data': diseases[disease_id]}
    else:
        return {'status': 'failed',
                'message': f'There is no disease with disease_id {disease_id}!'}


# Добавление новой болезни
@app.post('/diseases/new')
def post_new_disease(disease_id: int, disease_name: str) -> dict:
    if disease_id in diseases:
        return {'status': 'failed',
                'message': f'Disease_id {disease_id} is busy!'}
    else:
        diseases[disease_id] = disease_name
        return {'status': 'successful',
                'message': f'Disease {disease_name} with disease_id {disease_id} has been added!'}


# Получение предсказания для загруженного изображения
@app.post('/predict')
def predict(file: UploadFile) -> dict:

    X = []
    with Image.open(file.file) as img:
        img = np.array(ImageOps.grayscale(img).resize((256, 256)))
        pca = PCA(75)
        img_pca = pca.fit_transform(img)
        X.append(img_pca.flatten())
        X = np.array(X)

        file.file.close()

    global images_loaded
    images_loaded += 1  # без global не работает

    return {'status': 'successful',
            'filename': file.filename,
            'predict': f'Your disease is {diseases[model.predict(X)[-1]]}'}


# Получение статистики о работе сервиса
@app.get('/stats')
@cache(expire=60)
def get_stats() -> dict:
    return {'status': 'successful',
            'data': {'started_at': started_at,
                     'images_loaded': images_loaded,
                     'rating': round(np.mean(review), 2) if review else 'There are no reviews yet!'}}


# Оставить отзыв о работе сервиса
@app.post('/review/{rating}')
def post_review(rating: Annotated[int, Path(ge=1, le=5)]) -> dict:
    review.append(rating)
    return {'status': 'successful',
            'message': f'Your mark {rating} has been added. Thank you!'}


# Подключаем Redis
@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost:5370")  # 83.222.9.144
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
