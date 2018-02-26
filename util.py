import torch
from torch.autograd import Variable
import torchvision

def var(tensor, requires_grad=True):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    
    if tensor.dim() == 1 or tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        
    var = Variable(tensor.type(dtype), requires_grad=requires_grad)
    
    return var

def list2tensor(list_):
    tensor = torch.zeros(len(list_))
    for i in range(len(list_)):
        tensor[i] = list_[i]
    
    return tensor

def make_target_label(label):    
    # label = (N, class_num)
    label = label.data
    target_label = label.clone()
    
    for n in range(label.size(0)):
        hair_color = label[n, 0:3]
        gender = label[n, 3]
        age = label[n, 4]
        
        # make hair color label
        # if there is no selcted color in label
        if torch.sum(hair_color) == 0:
            # Just choose any color
            hair_color = int((torch.rand(1) * 3)[0])
            target_label[n, hair_color] = 1
        # if there is selected color in label
        else:
            # Just shuffle that label
            if torch.cuda.is_available():
                dtype = torch.cuda.LongTensor
            else:
                dtype = torch.LongTensor
                
            rand_idx = torch.randperm(3).type(dtype)
            target_label[n, 0:3] = hair_color[rand_idx]
            
        # Gender and age
        # Choose opposite value
        # 여기를 그냥 random으로 되게 해야하나?
        target_label[n, 3] = 0 if gender == 1 else 1
        target_label[n, 4] = 0 if age == 1 else 1
    
    return var(target_label)

def denorm(img):
    return (img / 2) + 0.5 

def save_img(real_img, fake_img, img_path):
    N = real_img.size(0)
    
    img = torch.cat((real_img.data, fake_img.data), dim=0)
    torchvision.utils.save_image(img, img_path, nrow=N)