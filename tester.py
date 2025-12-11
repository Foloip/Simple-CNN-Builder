import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_CPP_LOG_LEVEL"] = "3"
os.environ["MLIR_CRASH_REPRODUCER_DIRECTORY"] = ""
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10, mnist
from PIL import Image
import numpy as np


if not os.path.exists("ptm"):
    os.makedirs("ptm")
if not os.path.exists("images"):
    os.makedirs("images")

def model_founder():
    files = os.listdir("ptm")
    models = []
    for file in files:
        if file.endswith(".h5") or file.endswith(".keras"):
            models.append(file)
    return models

def photo_founder():
    files = os.listdir("images")
    images = []
    for file in files:
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            images.append(file)
    return images

def clear_console():
    print("\n" * 100)
error = True
while True:
    clear_console()
    print("""Добро пожаловать в тестировщик моделей.
    Здесь можно протестировать обученную модель""")
    print("1. Протестировать свою модель")
    print("2. Выйти")
    choice = int(input("Выбор: "))
    match choice:
        case 1:
            while True:
                models = model_founder()
                if len(models) == 0:
                    print("Ваши модели не найдены")
                    input("Нажмите enter что бы выйти")
                    error = False
                    break
                print("Список найденых моделей")
                for n in range(len(models)):
                    print(f"{n + 1}) {models[n]}")
                model_name = models[int(input("Выбор: ")) - 1]
                selected_model = f"ptm/{model_name}"
                print("Загрузка модели")
                model = load_model(selected_model)
                print("Модель загружена")
                print("На чём протестировать нейросеть")
                mode = 0
                choice = int(input("1. Своя картинка, 2. Тестовый датасет: "))
                break
        case 2:
            error = False
            break
    while error:
        match choice:
            case 2:
                inplayer = model.input_shape
                if inplayer == (None, 32, 32, 3):
                    print("Выбран датасет CiFar10 автоматически")
                    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
                    x_test = x_test / 255.0
                    print("Датасет загружен")
                    mode = 1
                    break
                else:
                    print("Выбран датасет MNIST автоматически")
                    (x_train, y_train), (x_test, y_test) = mnist.load_data()
                    x_test = x_test / 255.0
                    print("Датасет загружен")
                    mode = 2
                    break
            case 1:
                inplayer = model.input_shape
                images = photo_founder()
                if len(images) == 0:
                    print("Ваши фото не найдены")
                    input("Нажмите enter что бы выйти")
                    break
                print("Список найденых фото")
                print("(Фото будет автоматически конвертировано под формат нейросети)")
                for n in range(len(images)):
                    print(f"{n + 1}) {images[n]}")
                photo_name = images[int(input("Выбор: ")) - 1]
                selected_photo = f"images/{photo_name}"
                image = Image.open(selected_photo)
                if inplayer == (None, 32, 32, 3):
                    image = image.resize((32, 32))
                    image = image.convert("RGB")
                    mode = 3
                else:
                    image = image.convert("L")
                    image = image.resize((28, 28))
                    mode = 4
                nimage = np.array(image)
                nimage = nimage.reshape((1, 32, 32, 3))
                break
    # предсказание пон?
    while True if error == 1 else False:
        match mode:
            case 3:
                print("Режим 1 фотографии CiFar10")
                cifar10_classes = ['самолёт', 'автомобиль', 'птица', 'кошка/кот', 'олень', 'собака', 'лягушка',
                                   'лошадь',
                                   'корабль', 'грузовик']
                result = model.predict(nimage)
                max_index = np.argmax(result)
                print(f"Модель предсказала что это {cifar10_classes[max_index]}, номер класса {max_index + 1}")
                print("Вывод модели")
                print(result)
                print("Кол-во параметров: ", model.count_params())
                input("Нажмите enter что-бы выйти")
                break
            case 4:
                print("Режим 1 фотографии MNIST")
                mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                result = model.predict(nimage)
                max_index = np.argmax(result)
                print(f"Модель предсказала что это {mnist_classes[max_index]}")
                print("Вывод модели")
                print(result)
                print("Кол-во параметров: ", model.count_params())
                input("Нажмите enter что-бы выйти")
                break
            case 1:
                print("Режим датасета CiFar10")
                (loss, accuracy) = model.evaluate(x_test, y_test)
                print("\n")
                print(
                    f"Модель угадывает картинки с вероятностью: {accuracy * 100:.2f}%, и потери составляют {loss:.4f}\n")
                print("Кол-во параметров: ", model.count_params())
                input("Нажмите enter что-бы выйти")
                break
            case 2:
                print("Режим датасета MNIST")
                (loss, accuracy) = model.evaluate(x_test, y_test)
                print("\n")
                print(f"Модель угадывает числа с вероятностью: {accuracy * 100:.2f}%, и потери составляют {loss:.4f}\n")
                print("Кол-во параметров: ", model.count_params())
                input("Нажмите enter что-бы выйти")
                break
