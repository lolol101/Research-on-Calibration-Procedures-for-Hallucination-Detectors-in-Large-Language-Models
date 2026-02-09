import os
import pickle

class Index:
    def __init__(self, base_filename):
        self.base_filename = base_filename
        self.index_filename = f"{base_filename}_index.pkl"
        self.data_filename = f"{base_filename}_data.pkl"
        self.offsets = []
        self.iterations = []
        self.__load_index()

    def __load_index(self):
        if os.path.exists(self.index_filename):
            with open(self.index_filename, 'rb') as f:
                index_data = pickle.load(f)
                self.offsets = index_data.get('offsets', [])
                self.iterations = index_data.get('iterations', [])
            
    def __save_index(self):
        index_data = {
            'offsets': self.offsets,
            'iterations': self.iterations
        }
        with open(self.index_filename, 'wb') as f:
            pickle.dump(index_data, f)
            
    def save_data(self, data, iter=-1, logging=False):
        mode = "ab" if os.path.exists(self.data_filename) else "wb"
        with open(self.data_filename, mode) as f:
            offset = f.tell()
            pickle.dump(data, f)
            
            self.offsets.append(offset)
            self.iterations.append(iter)
            self.__save_index()
                 
        if logging:
            print("Saved data" + f": iteration {iter}" if iter != -1 else "")
            
    def load_data(self, start_iter=0, end_iter=None):
        """Loads all data (avoid in case if data is too heavy)"""
        data = []
        with open(self.data_filename, "rb") as f:
            for i in range(start_iter, end_iter if end_iter else len(self.offsets)):
                f.seek(self.offsets[i])
                data.append(pickle.load(f))
            return data 

    def load_data_generator(self, start_iter=0, end_iter=None, batch_size=1):
        """Makes data generator for heavy data"""
        data = []
        with open(self.data_filename, "rb") as f:
            for i in range(start_iter, end_iter if end_iter else len(self.offsets)):
                f.seek(self.offsets[i])
                data.append(pickle.load(f))   
                if len(data) == batch_size or i == end_iter - 1:
                    yield data
    
    def __len__(self):
        return len(self.offsets)
    
    def clear(self):
        files_to_remove = [self.index_filename, self.data_filename]
        
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"File is deleted: {file_path}")
                except OSError as e:
                    print(f"Error in deleting file {file_path}: {e}")
            else:
                print(f"Файл не существует: {file_path}")
        
        self.offsets = []
        self.iterations = []
        
        print("Index is cleared successfully.")
