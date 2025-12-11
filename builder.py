import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10, mnist

if not os.path.exists("ptm"):
    os.makedirs("ptm")
def conv2d_builder(filters, strides):
    return layers.Conv2D(filters=filters, kernel_size=(3,3), activation="relu", strides=strides, padding="same")
def MP2d_builder(shape):
    return layers.MaxPooling2D(pool_size=(shape, shape))
def model_founder():
    files = os.listdir("ptm")
    models = []
    for file in files:
        if file.endswith(".h5") or file.endswith(".keras"):
            models.append(file)
    return models

def clear_console():
    print("\n" * 100)

clear_console()
while True:
    clear_console()
    print("Добро пожаловать в конструктор моделей, выберите пункт который вам нужен")
    print("1. Список готовых моделей")
    print('2. Построить свою модель')
    print('3. Помощь')
    print('4. Выйти')
    choice = int(input("Выбор: "))
    match choice:
        case 1:
            clear_console()
            models = model_founder()
            n = 0
            while True:
                if len(models) == 0:
                    print("Готовые модели не найдены")
                    input("Нажмите enter что бы выйти")
                    break
                print("Список найденых моделей")
                for n in range(len(models)):
                    print(f"{n+1}) {models[n]}")
                model_name = models[int(input("Выбор: ")) - 1]
                selected_model = f"ptm/{model_name}"
                print(selected_model)
                print("Загрузка модели")
                model = load_model(selected_model)
                print("Модель загружена")
                print("Кол-во параметров: ", model.count_params())
                print("Размер входных данных: ", model.input_shape)
                print("Формат выходных данных: ", model.output_shape[-1])

                input("Нажмите enter что бы выйти")
                break
        case 2:
            clear_console()
            print("Добро пожаловать в сборщик моделей")
            print("Я попытался упростить процесс сборки что бы собрать нейронную сеть мог каждый")
            print("\n")
            print("Выберите дата сет на котором вы хотите обучить нейросеть")
            print("1. MNIST (Числа от 0 до 9) 2. CiFar10 (10 разных классов картинок) (сложнее MNIST)")
            while True:
                cds = int(input("Выбор: "))
                match cds:
                    case 1:
                        (X_train, y_train), (X_test, y_test) = mnist.load_data()
                        X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
                        X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
                        break
                    case 2:
                        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
                        X_train, X_test = X_train / 255.0, X_test / 255.0
                        break
                    case _:
                        print("Выберите другое число")
            while True:
                layers_model = []
                print("Теперь надо построить модель просто выберите поэтапно что вы хотите")
                print("Мы начнём с свёрток")
                if len(layers_model) == 0:
                    if cds == 1:
                        layers_model.append(layers.Input((28, 28, 1)))
                    else:
                        layers_model.append(layers.Input((32, 32, 3)))
                convf = int(input(f"Введите кол-во фильтров для свёртки, текущий слой: {len(layers_model)}: "))
                convs = int(input(f"Введите шаг свёртки, текущий слой: {len(layers_model)}: "))
                layers_model.append(conv2d_builder(convf, convs))
                while True:
                    print("Выберите пункт")
                    print("1. Следующий этап (рекомендую сначала добавить 2-3 свёртки)")
                    print("2. Добавить ещё свёртку")
                    print("3. Добавить MaxPooling2D")
                    choice = int(input())
                    match choice:
                        case 1:
                            break
                        case 2:
                            convf = int(input(f"Введите кол-во фильтров для свёртки, текущий слой: {len(layers_model)}: "))
                            convs = int(input(f"Введите кол-во шаг свёртки, текущий слой: {len(layers_model)}: "))
                            layers_model.append(conv2d_builder(convf, convs))
                        case 3:
                            mps = int(input(f"Введите размер разбивки, текущий слой: {len(layers_model)}: "))
                            layers_model.append(MP2d_builder(mps))
                        case _:
                            print("Выберите другое число")
                print("Добавляем слой Flatten для преобразования данных")
                layers_model.append(layers.Flatten())
                print("Теперь нужно сделать слой обработки (Dense) что бы получить результат. Рекомендую максимум 2 слоя")
                while True:
                    activ = ''
                    print("1. Добавить слой Dense")
                    print("2. Закончить и добавить Dense(10, softmax)")
                    choice = int(input())
                    match choice:
                        case 1:
                            neur = int(input(f"Выберите кол-во нейронов:  текущий слой: {len(layers_model)}: "))
                            act = int(input(f"Выберите функцию активации 1. ReLU 2. Sigmoid, текущий слой: {len(layers_model)}: "))
                            match act:
                                case 1:
                                    activ = 'relu'
                                case 2:
                                    activ = 'sigmoid'
                                case _:
                                    print("Выберите другое число")
                            layers_model.append(layers.Dense(neur, activation=activ))
                        case 2:
                            print("Добавление финального слоя")
                            layers_model.append(layers.Dense(10, activation='softmax'))
                            print("Сборка модели")
                            model = Sequential(layers=layers_model)
                            print("Компиляция с заготовленными параметрами")
                            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                            print("Модель скомпилирована")
                            input("Нажмите enter что бы продолжить")
                            break
                while True:
                    clear_console()
                    print("Отобразить список слоёв tensorflow?")
                    choice = int(input("1. Да, 2. Нет: "))
                    match choice:
                        case 1:
                            model.summary()
                            input("Нажмите enter что бы продолжить")
                            break
                        case 2:
                            break
                        case _:
                            print("Выберите другое число")
                print("Этап обучения на этом этапе можно обучить нейросеть класифицировать изображения")
                epochs = int(input("Выберите кол-во эпох (рекомендую 10-15): "))
                batch_size = int(input("Выберите batch_size (при высоком batch size нагрузка будет выше но скорость обучения выше: 01"))
                print("Начало обучения")
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
                print("Модель обучена")
                name = input("Выберите название для сохранения: ")
                form = ''
                while True:
                    formc = int(input("Выберете формат сохранение модели: 1. keras, 2. h5 (рекомендую keras): "))
                    match formc:
                        case 1:
                            form = '.keras'
                            break
                        case 2:
                            form = '.h5'
                            break
                        case _:
                            print("Выберите другое число")
                model.save("ptm/" + name + form)
                print("Модель сохранена как ", name + form)
                input("Нажмите enter что бы вернуться в меню")
                break

        case 3:
            clear_console()
            print("Это краткий гайд как собрать свёрточную нейросеть для классификации MNIST или CiFar10")
            print("Для начала требуется неплохой процессор или видеокарта (рекомендуется) для обучения модели")
            print("Так же нужно выбрать дата сет для обучения для начала рекомендую mnist")
            print("И так стоит помнит что у дата сетов разные входы и поэтому модели разные у mnist 1 канал (Ч/Б) а у CiFar10 3 канала (RGB)")
            print("А также у MNIST 28x28 а CiFar10 32x32")
            print("Начнём со свёртки можно использовать 2-3 свёрточных в начале с фильтрами по нарастающей")
            print("Одни из оптимальных вариантов 32-64-128 или 32-32-64 или 16-32-64 тут можно поиграться")
            print("Strides - это шаг свёртки то-есть если на вход подать картинку 32х32 с strides 2 то")
            print("на выходе мы получим вместо массива 32х32 мы получим 16х16 из-за шага свёртки")
            print("MaxPooling2D - это способ сжимания карты признаков он разбивает карту на части указанные в его параметрах и оставляет максимальное число")
            print("Его параметры это размер массива на которые он разобьёт тензор к примеру (2,2) что бы уменьшить карту вдвое")
            print("Flatten превращает 3д или 2д массив в 1д что бы потом исходя из признаков классифицировать картинку его необходимо делать после свёрток")
            print("Dense - это нейронный слой они обрабатывают признаки из свёрток можно поставить после свёрток 64 нейрона с активацией ReLU")
            print("так же можно попробовать поиграть и поставить 32 16 или убрать или повысить до 128 и более (остерегайтесь overfit)")
            print("Конец нейронной сети состоит из нескольких нейронов но для этой программы их всего 10 для обоих дата сетов")
            print("То есть это Dense слой с 10 нейронами и активацией softmax (только softmax это стандарт)")
            print("Этап обучения вообще для каждой нейронной сети он разный и зависит от устройства но попробуйте для начала 15 эпох с batch size 32")
            print("Значение loss должен уменьшатся а accuracy увеличиваться accuracy - это доля вероятности при которой нейронная сеть угадывает картинку")
            print("Loss - это ошибка нейронной сети, мы незнаем что означает его значение но надо что бы оно стало меньше это знак что сеть обучается")
            print("Могу дать несколько советов старайтесь не ставить много эпох можно получить overfit (переобучение)")
            print("После Flatten не нужно ставить больше 128 dense помимо переобучения вы получите много параметров и неэффективную модель")
            print("Рекомендую сохранять модели в формате keras так как h5 устаревший формат")
            print("Можно попробовать для начала обучить следующую модель:")
            print("Свёртка 32 фильтра страйд 1, свёртка 64 фильтра страйд 1, MaxPooling2D (2,2), Flatten, Dense 32 активация ReLU, Dense 10 активация softmax")
            print("На этом гайд закончен.")
            input("Нажмите enter что бы выйти: ")
        case 4:
            break