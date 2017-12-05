

class ApiParamsValueError(Exception):
    """
    Api参数值异常
    """
    def __init__(self, name, value, message):
        self.name = name
        self.value = value
        self.message = message


class ApiParamsTypeError(Exception):
    """
    Api参数类型异常
    """
    def __init__(self, name, value, message):
        self.name = name
        self.value = value
        self.message = message

