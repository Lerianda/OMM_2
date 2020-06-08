import numpy as np  # математические формулы
import matplotlib.pyplot as plt  # для создания 3D графиков
from collections import OrderedDict  # для выбора цвета
import imageio  # для создания анимации
import os  # для создания анимации

cmaps = OrderedDict()

Nx = 50  # количество шагов по х
Ny = 50  # количество шагов по у
M = 500  # количество шагов по t
Xm = 1  # максимальный x
Ym = 2  # максимальный y
T = 0.1

x = np.linspace(0, 1, Nx)  # все переменные
y = np.linspace(0, 2, Ny)
t = np.linspace(0, T, M)
u = np.zeros((Nx, Ny, 2 * M + 1))

h_x = Xm / Nx  # шаг цикла
h_y = Ym / Ny
tau = T / M

gamma_x = tau / h_x ** 2   # Доп переменная из методички
gamma_y = tau / h_y ** 2


def f_1(ix, iy, jt):
    return 0.5 * gamma_y * (u[ix, iy - 1, jt - 1] + u[ix, iy + 1, jt - 1]) + (1 - gamma_y) * u[
        ix, iy, jt - 1]


def f_2(ix, iy, jt):
    return 0.5 * gamma_x * (u[ix - 1, iy, jt - 1] + u[ix + 1, iy, jt - 1]) + (1 - gamma_x) * u[ix, iy, jt - 1]


def progon_x(iy, jt):  # явно выражена х, неявно у
    d = np.zeros(Nx)
    sigma = np.zeros(Nx)
    d[1] = 0  # из первого ГУ (u=0), u0=d0*u1+sigma0, Если производная =0, то d=1, sigma=0
    sigma[1] = 0
    a = 0.5 * gamma_x
    b = 1 + gamma_x
    c = 0.5 * gamma_x
    for m in range(1, Nx - 1):  # формулы из методички
        fm = - f_1(m, iy, jt)
        d[m+1] = c / (b - a * d[m])
        sigma[m+1] = (fm - a * sigma[m]) / (a * d[m] - b)
    u[Nx - 1, iy, jt] = 0  # Тут второе ГУ, u равна нулю. Если производная =0, см ниже пример
    for m in range(Nx - 1, 0, -1):
        u[m-1, iy, jt] = d[m] * u[m, iy, jt] + sigma[m]


def progon_y(ix, jt):
    d = np.zeros(Ny)
    sigma = np.zeros(Ny)
    d[1] = 0  # из первого ГУ (u=0). Если производная =0, то d=1, sigma=0
    sigma[1] = 0
    A = 0.5 * gamma_y
    B = 1 + gamma_y
    C = 0.5 * gamma_y
    for m in range(1, Ny - 1):  # формулы из методички
        fm = - f_2(ix, m, jt)
        d[m+1] = C / (B - A * d[m])
        sigma[m+1] = (fm - A * sigma[m]) / (A * d[m] - B)
    u[ix, Ny - 1, jt] = sigma[-1] / (1 - d[-1])  # Тут второе ГУ,производная=0
    # [-1] означает последний элемент массива (Ny-1)
    for m in range(Ny - 1, 0, -1):
        u[ix, m-1, jt] = d[m] * u[ix, m, jt] + sigma[m]


# beginning of the program

for i1 in range(0, Nx, 1):
    for i2 in range(0, Ny, 1):
        u[i1, i2, 0] = np.sin(2*np.pi*x[i1])*np.sin(np.pi*y[i2]/4)

for j in range(1, 2 * M, 2):
    for i2 in range(1, Ny - 1):
        progon_x(i2, j)
    for i1 in range(1, Nx - 1):
        progon_y(i1, j + 1)

# painting

x, y = np.meshgrid(x, y)

for i in range(0, M, 2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("$x$", fontsize=20)
    ax.set_ylabel("$y$", fontsize=20)
    ax.set_zlabel("$u$", fontsize=20)
    ax.set_zlim(-1.01, 1.01)
    ti = round(i * T / M, 3)
    ax.set_title("Рымарь Валерия\nВремя "+str(ti))
    fig.colorbar(ax.plot_surface(x, y, u[:, :, i], cmap=plt.get_cmap('RdYlGn')), shrink=1, aspect=8)
    ax.view_init(30, 20+i/15)  # здесь поворот картинки, если убрать i, она не будет крутиться

    if i < 10:  # Для верной сортировки по имени
        filename = 'framesangle/step00' + str(i) + '.png'
    else:
        if i < 100:
            filename = 'framesangle/step0' + str(i) + '.png'
        else:
            filename = 'framesangle/step' + str(i) + '.png'
    plt.savefig(filename, dpi=96)

    ax.clear()

# animation
folder = 'framesangle'
files = [f"{folder}\\{file}" for file in os.listdir(folder)]

images = [imageio.imread(file) for file in files]
imageio.mimwrite('Angle1.gif', images, fps=10)