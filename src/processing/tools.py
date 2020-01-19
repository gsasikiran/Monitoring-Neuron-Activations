class Tools:
    def __init__(self):
        pass

    @staticmethod
    def get_conv_layers(model):
        '''
            This method returns the list of convolutional layers' numbers.
        :param model: Model architecture
            A neural network model with a stack of layers
        :return: list
            Returns a list of integers with convolutional layers
        '''
        conv_layer_list = []
        for i in range(len(model.layers)):
            layer = model.layers[i]
            # check for convolutional layer
            if 'conv' not in layer.name:
                continue
            # summarize output shape
            conv_layer_list.append(i)
        return conv_layer_list
