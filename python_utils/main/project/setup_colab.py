#Original Autho jpcano1, modified for personal use from the original 'setup_colab.py'

import setup_colab_general as setup_general
import os
from google.colab import files
from IPython.display import clear_output

def setup_software(download_dataset=True, pretrained=True, 
                      brats=False):
    setup_general.setup_general()
    
    os.system("pip install -q albumentations==0.5.0")
    print("Libraries Installed!")
    #torch_path = "ISIS_4825/ML/Taller_13/torch_utils.py"
    #vis_path = "ISIS_4825/ML/Taller_13/visualization_utils.py"
    #layers_path = "ISIS_4825/ML/Taller_13/layers.py"
    #train_path = "ISIS_4825/ML/Taller_13/train_utils.py"
    #setup_general.download_github_content(torch_path, "utils/torch_utils.py")
    #setup_general.download_github_content(layers_path, "utils/layers.py")
    #setup_general.download_github_content(train_path, "utils/train_utils.py")
    #setup_general.download_github_content(vis_path, "utils/visualization_utils.py")
    #print("Util Functions Downloaded Successfully")
    print("Software Demo Enabled Successfully")
   
 
 def setup_journal():
  
  setup_general.setup_general()
  print("Interactive Paper Enabled Succesfully!)
