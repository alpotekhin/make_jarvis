class MessengerInterfaceSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise Exception("Messenger interface not initialized.")
        return cls._instance

    @classmethod
    def set_instance(cls, instance):
        cls._instance = instance
