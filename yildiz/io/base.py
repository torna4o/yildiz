# astrocore/io/base.py 

class DataSource: 
  def load(self): 
    """ Returns: 
          dict with standard keys: 
          { 
          "t": ..., 
          "y": ..., 
          "meta": ... 
          } 
    """ 
    raise NotImplementedError
