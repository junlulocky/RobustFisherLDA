import pandas as pd

class loader(object):

    def __init__(self, file_name, split_token = ','):
        self.file_name = file_name
        self.split_token = split_token

    def load(self):
        df = pd.io.parsers.read_csv(
        filepath_or_buffer = self.file_name,
                            header = None,
                            sep = self.split_token)

        self.dimension = df.shape[1]-1
        df.columns = range(self.dimension) + ['label']
        df.dropna(how = 'all', inplace = True) # to drop the empty line at file-end
        df.tail()

        self.X = df[range(self.dimension)].values
        self.Y = df['label'].values
        distinct_label = list(set(self.Y))

        if len(distinct_label) != 2:
            raise Exception('Two Labels Required\n','Label Sets:\n',distinct_label)

        self.Y = map(lambda x: {distinct_label[0]:1, distinct_label[1]:-1}[x], self.Y)

        return [self.X, self.Y]

if __name__ == "__main__":
    loader = loader('ionosphere/ionosphere.data')
    [X, y] = loader.load()
    print X.shape
