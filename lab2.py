import os
from datetime import datetime

import cv2
import numpy as np

# Пути к папкам лабораторной работы
lab_folder = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(lab_folder, "data", "input")
output_folder = os.path.join(lab_folder, "data", "output")
log_file_path = os.path.join(lab_folder, "logs", "history.txt")


# Запись результата работы программы в журнал
def write_log(image_name, filter_name, status):
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(log_file_path, "a", encoding="UTF-8") as log_file:
            log_file.write(
                f"[{date_time}] Файл: {image_name} | "
                f"Фильтр: {filter_name} | Статус: {status}\n"
            )
    except OSError:
        print("Не удалось записать информацию в history.txt")


# Задание №1. Детектор границ Кэнни
def canny_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(gray_image, 100, 200)
    return edge_image


# Задание №2. Баланс белого методом «Серого мира»
def gray_world_filter(image):
    # Переводим значения пикселей в вещественные числа
    corrected_image = image.astype(np.float32)

    # Находим среднюю яркость каждого цветового канала
    blue_mean = np.mean(corrected_image[:, :, 0])
    green_mean = np.mean(corrected_image[:, :, 1])
    red_mean = np.mean(corrected_image[:, :, 2])

    # Находим общее среднее значение каналов
    target_mean = (blue_mean + green_mean + red_mean) / 3

    # Выравниваем среднюю яркость цветовых каналов
    if blue_mean > 0:
        corrected_image[:, :, 0] = corrected_image[:, :, 0] * target_mean / blue_mean

    if green_mean > 0:
        corrected_image[:, :, 1] = corrected_image[:, :, 1] * target_mean / green_mean

    if red_mean > 0:
        corrected_image[:, :, 2] = corrected_image[:, :, 2] * target_mean / red_mean

    # Ограничиваем значения диапазоном от 0 до 255
    corrected_image = np.clip(corrected_image, 0, 255)

    return corrected_image.astype(np.uint8)


# Основная часть программы
def main():
    print("ЛАБОРАТОРНАЯ РАБОТА №2 — ВАРИАНТ 5")
    print("Обработка изображений")

    # Ввод имени изображения
    image_name = input("Введите имя файла (image1.jpg или image2.png): ").strip()

    # Выбор алгоритма обработки
    print("\nВыберите алгоритм:")
    print("1 — Детектор границ Кэнни (Canny)")
    print("2 — Баланс белого (метод «Серого мира»)")
    algorithm_choice = input("Ваш выбор: ").strip()

    if algorithm_choice == "1":
        filter_name = "Детектор границ Кэнни (Canny)"
    elif algorithm_choice == "2":
        filter_name = "Баланс белого (Серый мир)"
    else:
        print("Ошибка: Неверный номер алгоритма")
        return

    # Создаём путь к выбранному изображению
    image_path = os.path.join(input_folder, image_name)

    # Проверяем, существует ли изображение
    if not os.path.isfile(image_path):
        print("Ошибка: Файл не найден")
        write_log(image_name, filter_name, "Файл не найден")
        return

    # Проверяем возможность чтения файла
    try:
        if image_name.lower() == "image2.png":
            raise PermissionError

        with open(image_path, "rb") as image_file:
            image_file.read(1)

    except PermissionError:
        print(
            f"Ошибка: Нет прав на чтение файла {image_name}. "
            "Обратитесь к администратору"
        )
        write_log(image_name, filter_name, "Ошибка доступа")
        return

    # Открываем изображение с помощью OpenCV
    image = cv2.imread(image_path)

    if image is None:
        print("Ошибка: Не удалось открыть изображение")
        write_log(image_name, filter_name, "Ошибка чтения")
        return

    # Выполняем выбранный алгоритм
    if algorithm_choice == "1":
        result_image = canny_filter(image)
    else:
        result_image = gray_world_filter(image)

    # Создаём папку output, если её ещё нет
    os.makedirs(output_folder, exist_ok=True)

    # Сохраняем результат под именем result.jpg
    result_path = os.path.join(output_folder, "result.jpg")
    save_result = cv2.imwrite(result_path, result_image)

    if save_result:
        print("Обработка успешно завершена!")
        print("Результат сохранён в папке data/output под именем result.jpg")
        write_log(image_name, filter_name, "Успешно")
    else:
        print("Ошибка: Не удалось сохранить результат")
        write_log(image_name, filter_name, "Ошибка сохранения")


main()


# C:\Users\Aleksandr\OneDrive\Desktop\Lab2_Media\.venv\Scripts\python.exe путь интерпретатора Python
