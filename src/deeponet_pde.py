from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from IPython.display import clear_output
import itertools

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import deepxde as dde
from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from system import LTSystem, ODESystem, DRSystem, CVCSystem, ADVDSystem
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test

np.random.seed(0)
def test_u_lt(nn, system, T, m, model, data, u, fname):
    """Test Legendre transform"""
    sensors = np.linspace(-1, 1, num=m)
    sensor_value = u(sensors)
    s = system.eval_s(sensor_value)
    ns = np.arange(system.npoints_output)[:, None]
    X_test = [np.tile(sensor_value, (system.npoints_output, 1)), ns]
    y_test = s
    if nn != "opnn":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt("test/u_" + fname, sensor_value)
    np.savetxt("test/s_" + fname, np.hstack((ns, y_test, y_pred)))


def test_u_ode(nn, system, T, m, model, data, u, fname, num=100):
    """Test ODE"""
    sensors = np.linspace(0, T, num=m)[:, None]
    sensor_values = u(sensors)
    x = np.linspace(0, T, num=num)[:, None]
    X_test = [np.tile(sensor_values.T, (num, 1)), x]
    y_test = system.eval_s_func(u, x)
    if nn != "opnn":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt(fname, np.hstack((x, y_test, y_pred)))
    print("L2relative error:", dde.metrics.l2_relative_error(y_test, y_pred))


def test_u_dr(nn, system, T, m, model, data, u, fname):
    """Test Diffusion-reaction"""
    sensors = np.linspace(0, 1, num=m)
    sensor_value = u(sensors)
    s = system.eval_s(sensor_value)
    xt = np.array(list(itertools.product(range(m), range(system.Nt))))
    xt = xt * [1 / (m - 1), T / (system.Nt - 1)]
    X_test = [np.tile(sensor_value, (m * system.Nt, 1)), xt]
    y_test = s.reshape([m * system.Nt, 1])
    if nn != "opnn":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt(fname, np.hstack((xt, y_test, y_pred)))


def test_u_cvc(nn, system, T, m, model, data, u, fname):
    """Test Advection"""
    sensors = np.linspace(0, 1, num=m)
    sensor_value = u(sensors)
    s = system.eval_s(sensor_value)
    xt = np.array(list(itertools.product(range(m), range(system.Nt))))
    xt = xt * [1 / (m - 1), T / (system.Nt - 1)]
    X_test = [np.tile(sensor_value, (m * system.Nt, 1)), xt]
    y_test = s.reshape([m * system.Nt, 1])
    if nn != "opnn":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt("test/u_" + fname, sensor_value)
    np.savetxt("test/s_" + fname, np.hstack((xt, y_test, y_pred)))


def test_u_advd(nn, system, T, m, model, data, u, fname):
    """Test Advection-diffusion"""
    sensors = np.linspace(0, 1, num=m)
    sensor_value = u(sensors)
    s = system.eval_s(sensor_value)
    xt = np.array(list(itertools.product(range(m), range(system.Nt))))
    xt = xt * [1 / (m - 1), T / (system.Nt - 1)]
    X_test = [np.tile(sensor_value, (m * system.Nt, 1)), xt]
    y_test = s.reshape([m * system.Nt, 1])
    if nn != "opnn":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt("test/u_" + fname, sensor_value)
    np.savetxt("test/s_" + fname, np.hstack((xt, y_test, y_pred)))


def lt_system(npoints_output):
    """Legendre transform"""
    return LTSystem(npoints_output)


def ode_system(T):
    """ODE"""

    def g(s, u, x):
        # Antiderivative
        # return u
        # Nonlinear ODE
        # return -s**2 + u
        # Gravity pendulum
        k = 1
        return [s[1], - k * np.sin(s[0]) + u]

    # s0 = [0]
    s0 = [0, 0]  # Gravity pendulum
    return ODESystem(g, s0, T)


def dr_system(T, npoints_output):
    """Diffusion-reaction"""
    D = 0.01
    k = 0.01
    Nt = 100
    return DRSystem(D, k, T, Nt, npoints_output)


def cvc_system(T, npoints_output):
    """Advection"""
    f = None
    g = None
    Nt = 100
    return CVCSystem(f, g, T, Nt, npoints_output)


def advd_system(T, npoints_output):
    """Advection-diffusion"""
    f = None
    g = None
    Nt = 100
    return ADVDSystem(f, g, T, Nt, npoints_output)


def run(problem, system, space, T, m, nn, net, lr, epochs, num_train, num_test,layer_count,train_losses,test_losses, prms):
    # space_test = GRF(1, length_scale=0.1, N=1000, interp="cubic")

    # X_train, y_train = system.gen_operator_data(space, m, num_train)
    # X_test, y_test = system.gen_operator_data(space, m, num_test)
    # # if nn != "opnn":
    # #     X_train = merge_values(X_train)
    # #     X_test = merge_values(X_test)

    # np.savez_compressed("/content/deeponet/data_20k/train.npz", X_train0=X_train[0], X_train1=X_train[1], y_train=y_train)
    # np.savez_compressed("/content/deeponet/data_20k/test.npz", X_test0=X_test[0], X_test1=X_test[1], y_test=y_test)
    # return

    d = np.load("/content/deeponet/data/data_1k/train.npz")
    X_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"]
    d = np.load("/content/deeponet/data/data_1k/test.npz")
    X_test, y_test = (d["X_test0"], d["X_test1"]), d["y_test"]

    # y_train += np.random.normal(loc=0.0, scale=0.001, size=y_train.shape)

    X_test_trim = trim_to_65535(X_test)[0]
    y_test_trim = trim_to_65535(y_test)[0]
    if nn == "opnn":
        data = dde.data.Triple(
            X_train=X_train, y_train=y_train, X_test=X_test_trim, y_test=y_test_trim
        )
    else:
        data = dde.data.DataSet(
            X_train=X_train, y_train=y_train, X_test=X_test_trim, y_test=y_test_trim
        )

    model = dde.Model(data, net)
    model.compile("adam", lr=lr)#, metrics=[mean_squared_error_outlier])
    checker = dde.callbacks.ModelCheckpoint(
        "model/model.ckpt", save_better_only=True, period=1000
    )
    losshistory, train_state = model.train(epochs=epochs, callbacks=[checker])#, batch_size=32)
    
    train_losses.append(train_state.loss_train[0])
    test_losses.append(train_state.loss_test[0])

    # print("# Parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
    prms.append(np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
    print("Parameters:", tf.compat.v1.trainable_variables())
    
    dde.saveplot(
        losshistory, 
        train_state, 
        issave=True, 
        isplot=True, 
        output_dir=f'/content/drive/MyDrive/Pulkit/DeepONet/DON_Pendulum/Results/1k/abs_activation_gd/results_{layer_count}'
    )

    model.restore("model/model.ckpt-" + str(train_state.best_step), verbose=1)
    # safe_test(model, data, X_test, y_test)

    tests = [
        (lambda x: x, "x.dat"),
        (lambda x: np.sin(np.pi * x), "sinx.dat"),
        (lambda x: np.sin(2 * np.pi * x), "sin2x.dat"),
        (lambda x: x * np.sin(2 * np.pi * x), "xsin2x.dat"),
    ]
    # for u, fname in tests:
    #     if problem == "lt":
    #         test_u_lt(nn, system, T, m, model, data, u, fname)
    #     elif problem == "ode":
    #         test_u_ode(nn, system, T, m, model, data, u, fname)
    #     elif problem == "dr":
    #         test_u_dr(nn, system, T, m, model, data, u, fname)
    #     elif problem == "cvc":
    #         test_u_cvc(nn, system, T, m, model, data, u, fname)
    #     elif problem == "advd":
    #         test_u_advd(nn, system, T, m, model, data, u, fname)

    if problem == "lt":
        features = space.random(10)
        sensors = np.linspace(0, 2, num=m)[:, None]
        u = space.eval_u(features, sensors)
        for i in range(u.shape[0]):
            test_u_lt(nn, system, T, m, model, data, lambda x: u[i], str(i) + ".dat")

    if problem == "cvc":
        features = space.random(10)
        sensors = np.linspace(0, 1, num=m)[:, None]
        # Case I Input: V(sin^2(pi*x))
        u = space.eval_u(features, np.sin(np.pi * sensors) ** 2)
        # Case II Input: x*V(x)
        # u = sensors.T * space.eval_u(features, sensors)
        # Case III/IV Input: V(x)
        # u = space.eval_u(features, sensors)
        for i in range(u.shape[0]):
            test_u_cvc(nn, system, T, m, model, data, lambda x: u[i], str(i) + ".dat")

    if problem == "advd":
        features = space.random(10)
        sensors = np.linspace(0, 1, num=m)[:, None]
        u = space.eval_u(features, np.sin(np.pi * sensors) ** 2)
        for i in range(u.shape[0]):
            test_u_advd(nn, system, T, m, model, data, lambda x: u[i], str(i) + ".dat")


def main():
    # Problems:
    # - "lt": Legendre transform
    # - "ode": Antiderivative, Nonlinear ODE, Gravity pendulum
    # - "dr": Diffusion-reaction
    # - "cvc": Advection
    # - "advd": Advection-diffusion
    problem = "ode"
    T = 1
    if problem == "lt":
        npoints_output = 20
        system = lt_system(npoints_output)
    elif problem == "ode":
        system = ode_system(T)
    elif problem == "dr":
        npoints_output = 100
        system = dr_system(T, npoints_output)
    elif problem == "cvc":
        npoints_output = 100
        system = cvc_system(T, npoints_output)
    elif problem == "advd":
        npoints_output = 100
        system = advd_system(T, npoints_output)

    # Function space
    # space = FinitePowerSeries(N=100, M=1)
    space = FiniteChebyshev(N=20, M=1)
    # space = GRF(2, length_scale=0.2, N=2000, interp="cubic")  # "lt"
    # space = GRF(1, length_scale=0.2, N=1000, interp="cubic")
    # space = GRF(T, length_scale=0.2, N=1000 * T, interp="cubic")

    # Hyperparameters
    m = 100
    num_train = 5000
    num_test = 1000
    lr = 0.0001
    epochs = 1

    # Network
    nn = "opnn"
    activation = "abs" #relu
    initializer = "Glorot normal"  # "He normal" or "Glorot normal"
    dim_x = 1 if problem in ["ode", "lt"] else 2
   
    train_losses = []
    test_losses = []
    prms = []
    # if not os.path.isdir("/content/drive/MyDrive/5k/noise_results_5k_001_sgd"):
    #   os.mkdir("/content/drive/MyDrive/5k/noise_results_5k_001_sgd")

    for layer_width in range(5, 6, 5):
        tf.keras.backend.clear_session()

        branch_sizes = [m, layer_width, layer_width]
        trunk_sizes = [dim_x, layer_width, layer_width]
       
        net = dde.maps.DeepONet(
            branch_sizes,
            trunk_sizes,
            activation,
            initializer,
            use_bias=False,
            stacked=False,
        )

        run(problem, system, space, T, m, nn, net, lr, epochs, num_train, num_test,
                layer_width, train_losses, test_losses, prms)
        
        # plt.clf()
        # plt.plot(np.array(train_losses))
        # plt.savefig('/content/drive/MyDrive/Pulkit/DeepONet/DON_Pendulum/Results/1k/abs_activation_gd/train_final.png')
        
        # plt.clf()
        # plt.plot(np.array(test_losses))
        # plt.savefig('/content/drive/MyDrive/Pulkit/DeepONet/DON_Pendulum/Results/1k/abs_activation_gd/test_final.png')

        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        del net

    print(prms)
    
if __name__ == "__main__":
    main()