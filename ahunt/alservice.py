import os


class ALServiceBase:
    def __init__(self,root_dir,csv_file=None,als_config = None, session=None,st=None):
        self.root_dir = root_dir
        self.als_config = als_config
        if not csv_file: csv_file = os.path.join(root_dir,'labels.csv')
        self.csv_file = csv_file
        if st: self.st = st
        if session: self.session = session

        # TODO
        # add self.resume() here and remove from ctflow

    def test(self):
        pass
    
    def train(self):
        pass

    def predict(self):
        pass
        
    def __str__(self):
        return str(self.als_config)

    def resume(self):
        pass
