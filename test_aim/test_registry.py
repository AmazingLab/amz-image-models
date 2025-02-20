from aim import registry


@registry.auto_register()
class RegistryClass(object):
    def __init__(self):
        print('Initializing')

    def __call__(self, *args, **kwargs):
        print("Calling")


class NonRegistryClass(object):
    def __init__(self):
        print('Initializing')

    def __call__(self, *args, **kwargs):
        print("Calling")


if __name__ == '__main__':
    # 查看注册表
    print(registry._registry)
    # 构造类
    obj = registry.build("RegistryClass")
    obj()

    # 覆盖注册
    registry.register(RegistryClass, override=True)
    # 构造覆盖后的类
    obj = registry.build("RegistryClass")
    obj()

    # 构造没有注册的类
    registry.build("test_aim.test_registry.NonRegistryClass", restrict=False)
