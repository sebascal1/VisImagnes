#Original Author jpcano1, modified for personal use from the original 'setup_colab.py'#

import setup_colab_general as setup_general
import os
from google.colab import files
from IPython.display import clear_output

def setup_software():
    setup_general.setup_general()
    
    os.system("pip install -q albumentations==0.5.0")
    print("Libraries Installed!")
    torch_path = "project/torch_utils.py"
    vis_path = "project/visualization_utils.py"
    layers_path = "project/layers.py"
    train_path = "project/train_utils.py"
    lung_path = "project/lung_segment.py"
    setup_general.download_github_content(torch_path, "utils/torch_utils.py")
    setup_general.download_github_content(layers_path, "utils/layers.py")
    setup_general.download_github_content(train_path, "utils/train_utils.py")
    setup_general.download_github_content(vis_path, "utils/visualization_utils.py")
    setup_general.download_github_content(lung_path, "utils/lung_segment.py")
    
    from utils import general as gen
    #if download_dataset:
    train_id ="1zaucizp_3iy_Tlk4NNfNqEtP25qcSKLl"
    test_id = "1uqMqdxDmBeQNu-Zziaa02Yoa6IBsMCDk"
    gen.download_file_from_google_drive(train_id, "train_data.zip", size=1.96e6)
    gen.download_file_from_google_drive(test_id, "test_data.zip", size=805.2e3)
    print("Dataset Downloaded Successfully")
    
    
    print("Util Functions Downloaded Successfully")
    print("Software Demo Enabled Successfully")
   
def setup_journal():
    setup_general.setup_general()
    print("Interactive Paper Enabled Succesfully!")
