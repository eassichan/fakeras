# -*- coding: utf-8 -*-

"""
fakeras.exception
~~~~~~~~~~~~~~~~~

本模块集中定义了fakeras的异常类，以及一些公共的异常检测函数.
"""


class FakerasException(Exception):
    """ fakeras异常基类. """
    pass


class BackwardBeforeForward(FakerasException):
    """ 试图在执行正向传播之前执行反向传播. """
    pass


class TensorShapeNotMatch(FakerasException):
    """ 在需要两个张量的形状必须一致的场合发现张量的形状并不相同. """
    pass


def compare_tensor_shape(reference, target):
    """ 比较两个张量的形状是否一致，如果不一致则抛出TensorShapeNotMatch异常，
        主要用于函数的参数验证.

    Parameters
    ----------
    reference: numpy.ndarray
        张量形状比较时作为参考基准的张量.
    target: numpy.ndarray
        待比较的目标张量.

    Raises
    ------
    TensorShapeNotMatch:
        要比较的两个张量的形状不一致.
    """
    if target.shape != reference.shape:
        err_msg = "The shape of target tensor%s is not match reference tensor%s." % (
            str(target.shape),
            str(reference.shape),
        )
        raise TensorShapeNotMatch(err_msg)


def validate_data_type(target_type, param_name, param_value):
    """ 验证参数的数据类型，如果不是期望的类型则抛出TypeError异常.

    Parameters
    ----------
    target_type: python build-in type or other customized class
        参数的目标类型，即符合参数定义的合法数据类型.
    param_name: str
        参数的名字，即函数调用中的参数名.
    param_value: python build-in object or other customized object
        参数值，即函数调用中实际传入的值.

    Rasies
    ------
    TypeError:
        'param_value'指向的对象的类型和'target_type'不一样.
    """
    if not isinstance(param_value, target_type):
        actual_type = type(param_value)
        err_msg = "Parameter '{pname}' must be an instance of {ttype}, but '{atype}' is found.".format(
            pname=param_name, ttype=str(target_type), atype=str(actual_type)
        )
        raise TypeError(err_msg)
