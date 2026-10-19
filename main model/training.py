import torch
torch.cuda.empty_cache()
from torch import nn, optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import time
from  PIL import  Image
import cv2
import numpy as np
import argparse
import re
import copy
import tifffile as tiff
from data_handler import Get_Data   # 224*224 input size
from Model_architecture import ResNet50_ViT


def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        # obtain learning rate
        return param_group['lr']

def learning_curve(record_train,record_test=None):
    plt.style.use('ggplot')

    plt.plot(range(1,len(record_train)+1),record_train,label='train acc',color='red')
    if record_test is not None:
        plt.plot(range(1,len(record_test)+1),record_test,label="test acc",color='blue')

    # Adds a legend to the plot at location 4 (lower right)
    plt.title("learning curve")
    #Set the x-axis tick values to be at intervals of 5, (0 to the length of record_train (inclusive))
    plt.xticks(range(0,len(record_train)+1,5))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.savefig('C:/path/to/your/local_saving/learning_curve.png')
    plt.show()

def plot_loss(train_loss_values, test_loss_values=None, title="Training Loss"):
    plt.style.use('ggplot')

    # Plot loss values over epochs.
    epochs = range(1, len(train_loss_values) + 1)
    plt.plot(epochs, train_loss_values, label='Training Loss', color='red')
    if test_loss_values is not None:
        plt.plot(epochs, test_loss_values, label='Test Loss', color='blue')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig('C:/path/to/your/local_saving/loss_curve.png')
    plt.show()

def label_smooth(label,smooth_rate=0.1):
    label=label*(1-smooth_rate)+ (1-label)*smooth_rate
    return label
    
def main_model(opt, model_path=None, train_iter=None, valid_iter=None, test_iter=None, checkpoint_path=None, save_path=None, best_acc=90.0):
    params_to_update=[]

    model=ResNet50_ViT(num_classes=opt.num_class)
    print(model)

    checkpoint=torch.load('C:/path/to/your/pretrained_model_weight.pt')

    pretrained_params = checkpoint

    # Create a new state_dict with only matching keys
    model_dict = model.state_dict()  # Get the new model's state_dict
    pretrained_dict = {k: v for k, v in pretrained_params.items() if k in model_dict} 

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Freeze parameters that were loaded from the pretrained model
    for name, param in model.named_parameters():
        '''if name in pretrained_dict:  
            param.requires_grad = False  # Freeze if it was in the pretrained model
        else:
            param.requires_grad = True'''
        param.requires_grad = True
            
    for name,param in model.named_parameters():
        if param.requires_grad==True:
            params_to_update.append(param)
            # print(name)

    model = model.to(opt.device)
    # print(params_to_update)

    filename = 'C:/path/to/your/new/saving_model_weight.pt'
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=75.0    

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer=optim.AdamW(params_to_update, lr=opt.learning_rate, weight_decay=opt.weight_decay)

    lr_scheduler=optim.lr_scheduler.StepLR(optimizer,opt.scheduler_step,opt.scheduler_gamma)

    train_acclist=list()
    test_acclist=list()
    train_loss_valueslist=list()
    test_loss_valueslist=list()

    '''---Train---'''
    # record the current time as the start time for measuring the training duration
    start=time.time()
    for epoch in range(opt.num_epochs):
        total,correct,train_loss=0,0,0
        n_samples,n_correct=0,0
        model.train() 
        optimizer.zero_grad()

        print('=================epoch:[{}/{}]======================'.format(epoch+1,A.num_epochs))
        for i,(inputs,labels) in enumerate(train_iter):
            # print(type(labels))
            inputs,labels=inputs.to(opt.device),labels.to(opt.device)

            output=model(inputs)
            print(output)
            # output = label_smooth(output)
            # print(output)
            # print(output.shape)
            # labels = labels.float()
            loss = criterion(output, labels)

            # Normalize loss by the number of accumulation steps
            loss = loss / opt.accumulation_steps
            loss.backward()

            # Perform optimizer step every accumulation_steps
            if (i + 1) % opt.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss+=loss.item()

            _, preds = torch.max(output, 1)
            print(preds)
            n_samples += labels.size(0)
            #print(n_samples)
            n_correct += (preds == labels).sum().item()
            print(labels)
            # print(n_correct)

            train_acc = 100.0 * n_correct / n_samples
            # print(train_acc)

            train_loss=train_loss/len(train_iter)

            # print(i)
            if (i + 1) % opt.num_print == 0:
                print("step: [{}/{}], train_loss: {:.3f} | train_acc: {:6.3f}% | lr: {:.6f}" \
                    .format(i + 1, len(train_iter), train_loss, \
                    train_acc, get_cur_lr(optimizer)))
        
        '''---Test---'''
        model.eval()
        total,correct,test_loss=0,0,0

        with torch.no_grad():
            print("*************************test***************************")

            for inputs,labels in test_iter:
                inputs,labels=inputs.to(opt.device),labels.to(opt.device)

                output=model(inputs)
                
                loss=criterion(output, labels)

                test_loss += loss.item()

                _, preds = torch.max(output, 1)
                print(preds)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                print(labels)


            test_acc=100.0*correct/total
            test_loss=test_loss/len(test_iter)
            print("test_loss:{:.3} | test_acc:{:6.3f}%"\
                .format(test_loss,test_acc)
                )

        # print the time duration of the training process
        print("-----cost time:{:.4f}s----".format(time.time()-start))

        if test_acc>=best_acc:
            best_acc=test_acc
            best_model_wts=copy.deepcopy(model.state_dict())
            #state={'model':model.state_dict(),'best_acc': best_acc, 'optimizer':optimizer.state_dict(),'epoch':epoch}
            torch.save(model.state_dict(),filename)
            print("save model")

        train_acclist.append(train_acc)
        test_acclist.append(test_acc)
        train_loss_valueslist.append(train_loss)
        test_loss_valueslist.append(test_loss)

        if lr_scheduler is not None:
            lr_scheduler.step()

    plot_loss(train_loss_valueslist,test_loss_valueslist)
    learning_curve(train_acclist,test_acclist)

    # model.load_state_dict(best_model_wts) #this is the best model
    
    return model, train_acclist, test_acclist, train_loss_valueslist, test_loss_valueslist


def settings():
    init = argparse.Namespace()
    init.train_batch_size=8       # feed in data amount each time
    init.test_batch_size=8
    init.accumulation_steps = 1
    init.num_epochs=20      # iteration time
    init.num_class=4
    init.learning_rate=1e-4
    init.momentum=0.82
    init.weight_decay=1e-4
    init.num_print=1
    init.test_num=0
    init.scheduler_step=10
    init.scheduler_gamma=0.1
    init.validation_interval=1
    init.train_path= 'C:/path/to/your/Train.txt'
    init.validate_path= None
    init.test_path= 'C:/path/to/your/Val.txt'

    init.device="cuda" if torch.cuda.is_available() else "cpu"
    print("device:",init.device)

    init.model = 'mitonet'
    init.cpu=False
    init.imageSize=1024

    return init


if __name__ == '__main__':
    # load opt
    A=settings()

    # load data
    train_iter, test_iter = Get_Data(A)
    model_path=None
    
    model, train_acclist, test_acclist, train_loss_valueslist, test_loss_valueslist=main_model(A, model_path, train_iter=train_iter, test_iter=test_iter)
    print("Finished Training")
	# save model after training, final model    
    torch.save(model, 'C:/path/to/your/new/saving_final_model_weight_with_accuracy.pt')
    torch.save(model.state_dict(), 'C:/path/to/your/new/saving_final_model_weight.pt')
    
