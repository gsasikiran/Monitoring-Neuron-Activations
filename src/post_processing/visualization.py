import matplotlib.pyplot as plt


class DisplayActivations:

    def __init__(self):
        pass

    @staticmethod
    def __get_rows_cols(n):
        '''
        Creates better number of rows and cols for plotting
        @param
        n: int
            Total number of feature maps

        @returns int, int
            Returns number of rows and cols to be plotted

        '''
        if n == 32:
            return 8, 4
        elif n == 64:
            return 8, 8
        elif n == 128:
            return 8, 16
        elif n == 256:
            return 16, 16
        elif n == 512:
            return 16, 32

    def plot_activation(self, feature_maps, layer=None):
        """
            Plot the activation layers of the predicted feature maps
        @param
        feature_maps: List
            The list of activation outputs from the predicted layers
        rows: int (default:8)
            Number of rows to be in the plot
        cols: int (default:4)
            Number of cols to be in the plot
        layer: int (default: None)
            The layer of activation to be plotted. If none, plots all the layers

        @returns
            Returns a set of plots with assigned rows and cols
        """

        if not layer:
            feature_maps = feature_maps

            for fmap in feature_maps:

                rows, cols = self.__get_rows_cols(len(fmap[0][0][1]))

                ix = 1
                for _ in range(rows):
                    for _ in range(cols):
                        ax = plt.subplot(rows, cols, ix)
                        ax.set_xticks([])
                        ax.set_yticks([])

                        plt.imshow(fmap[0, :, :, ix - 1], cmap='gray')

                        ix += 1

                plt.show()

        else:
            feature_maps = feature_maps[layer - 1]

            for fmap in feature_maps:

                rows, cols = self.__get_rows_cols(len(fmap[0][1]))
                ix = 1
                for _ in range(rows):
                    for _ in range(cols):
                        ax = plt.subplot(rows, cols, ix)
                        ax.set_xticks([])
                        ax.set_yticks([])

                        plt.imshow(fmap[:, :, ix - 1], cmap='gray')

                        ix += 1

                plt.show()
