import logging
import traceback
import torch
import matplotlib.pyplot as plt
import sys
import copy
import os
import time
from torch import nn
import torch.nn.functional as F
#import wandb
import configparser
import argparse
import json
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import f1_score
from sklearn.metrics import top_k_accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data import get_torch_dataset
from model.preresnet import PreResNet


class MyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, lrs, last_epoch=-1):
        self.lrs = lrs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        
        if epoch >= len(self.lrs):
            return [self.lrs[-1] for _ in self.optimizer.param_groups]
        
        return [self.lrs[epoch] for _ in self.optimizer.param_groups]
    

class DynamicScheduler(_LRScheduler):
    def __init__(self, optimizer, init_lr=0.01, last_epoch=-1):
        self.current_lr = init_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.current_lr for _ in self.optimizer.param_groups]

    def set_lr(self, lr):
        self.current_lr = lr
    



class WindowDataset(Dataset):
    def __init__(self, X, y, window=15):
        self.X, self.y = [], []
        i = window
        while i < len(X):
            if i%180 != 0:
                self.X.append(X[i-window:i])
                self.y.append(y[i])
            else:
                i += window 
                self.X.append(X[i-window:i])
                self.y.append(y[i])
            i += 1
        
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout, batch_first):
        super().__init__()

        self.gru = nn.GRU(
            input_size = 60,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :] 
        return self.head(h).squeeze(-1)


def lr_policy_training():
    X_train = torch.tensor(np.load("features_wafer_v3.npy")[:540])
    y_train = torch.tensor(np.load("targets_wafer_v3.npy")[:540])

    X_test =  torch.tensor(np.load("features_wafer_v3.npy")[540:720])
    y_test = torch.tensor(np.load("targets_wafer_v3.npy")[540:720])
        
    train_ds = WindowDataset(X_train, y_train, window=15)
    test_ds = WindowDataset(X_test, y_test, window=15)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model_gru = GRUModel(input_dim = 60, hidden = 129, num_layers = 3, dropout = 0.027, batch_first = True).to(device)


    opt = torch.optim.Adam(model_gru.parameters(), lr=0.000131, weight_decay=1e-3)
    criterion = nn.MSELoss()


    for _ in range(40):
        model_gru.train()
        tr_loss, tr_n = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model_gru(xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_n += xb.size(0)
    

    model_gru.eval()
    preds_scaled, targets_scaled = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            p = model_gru(xb).cpu().numpy()
            preds_scaled.append(p)
            targets_scaled.append(yb.numpy())
    preds_scaled = np.concatenate(preds_scaled).reshape(-1, 1).flatten()
    targets_scaled = np.concatenate(targets_scaled).reshape(-1, 1).flatten()

    #print(len(preds_scaled))
   #print(len(targets_scaled))

    x_plot = np.arange(10, 180)
    plt.plot(preds_scaled, color = 'red')
    plt.plot(targets_scaled, color = 'black')
    plt.ylim(0, 2.5)

#preds_log = y_scaler.inverse_transform(preds_scaled)
#targets_log = y_scaler.inverse_transform(targets_scaled).flatten()


    #print(*[f"Pred: {p:.4f} | True: {t:.4f}" for p, t in zip(preds_scaled, targets_scaled)], sep="\n")

    
    return model_gru


def one_step(model, x_batch, y_batch, loss, loss_func, lr):
    grads = torch.autograd.grad(loss, model.parameters())
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads):
            p -= lr * g
    new_loss = loss_func(model(x_batch), y_batch)
    return new_loss.item()

def research(model, x_batch_list, y_batch_list, loss_func, snapshot, device):
    dict_loss  = {0.0005: [], 0.001: [], 0.01: [], 0.025: [], 0.05: [], 0.1: [], 0.2: [], 0.3: [], 0.4: [],  
                      0.5: [],  0.6: [], 0.7: [], 0.8:[], 0.9:[], 1.0: [], 1.1:[], 1.2: [], 1.3: [], 1.4: [],  
                      1.5: [],  1.6: [], 1.7: [], 1.8:[], 1.9:[], 2.0:[], 2.1: [], 2.2: [], 2.3: [], 2.4: [], 2.5: []}

    base_state = snapshot
        
    for j in dict_loss.keys():
        model_copy = copy.deepcopy(model).to(device)
        model_copy.load_state_dict(base_state)
        model_copy.train()
            
        loss = loss_func(model_copy(x_batch_list[0]), y_batch_list[0])
        dict_loss[j].append(one_step(model_copy, x_batch_list[1], y_batch_list[1], loss,
                                            loss_func, lr=j))
    return dict_loss

def try_cuda(*wargs):
    cuda_args = []
    for arg in wargs:
        if hasattr(arg, 'cuda'):
            cuda_args.append(arg.cuda())
        else:
            print(f"{arg} does not have a .cuda() method.")
            cuda_args.append(arg)
    return tuple(cuda_args)


#def kl_div_logits(p, q, T):
#    loss_func = nn.KLDivLoss(reduction = 'batchmean', log_target=True)
#    loss = loss_func(F.log_softmax(p/T, dim=-1), F.log_softmax(q/T, dim=-1)) * T * T
#    return loss



def kl_div_logits(p_logits, q_logits, T=1.0):
    p = torch.sigmoid(p_logits / T)
    q = torch.sigmoid(q_logits / T)

    p = p.clamp(1e-6, 1 - 1e-6)
    q = q.clamp(1e-6, 1 - 1e-6)

    kl = (
        p * (p.log() - q.log()) +
        (1 - p) * ((1 - p).log() - (1 - q).log())
    )

    return kl.mean() * (T * T)

def kl_sigmoid(p, q, eps=1e-6):
    p = torch.sigmoid(p)
    q = torch.sigmoid(q)
    
    p = torch.clamp(p, eps, 1-eps)
    q = torch.clamp(q, eps, 1-eps)
    
    return (p * (p.log() - q.log()) + (1-p)*( (1-p).log() - (1-q).log() )).mean()

def get_batch(data_loader, batch_index):
    start_index = batch_index * data_loader.batch_size
    end_index = start_index + data_loader.batch_size
    batch_data = []
    batch_targets = []
    
    for i in range(start_index, end_index):
        if i >= len(data_loader.dataset):
            break
        data, target = data_loader.dataset[i]
        batch_data.append(data)
        batch_targets.append(target)
    
    return torch.stack(batch_data), torch.stack(batch_targets)


def evaluate(teacher, student, loader, epoch):

    teacher_vector = []
    student_vector = []
    snapshot_dir = os.path.join('/kaggle/working/snapshots/', args.exp_name)

    os.makedirs(snapshot_dir, exist_ok=True)

    teacher.eval()
    student.eval()
    criterion = nn.BCEWithLogitsLoss()
    teacher_loss, student_loss = 0, 0
    teacher_correct, student_correct = 0, 0
    student_correct_strict = 0
    total, total_strict = 0, 0
    start = time.time()
    
    # Сохраняем первый батч
    first_batch_inputs = None
    first_batch_targets = None
    
    for batch_idx, batch in enumerate(loader):
        with torch.no_grad():
            inputs, targets = try_cuda(*batch[:2])


            # Сохраняем первый батч
            if batch_idx == 0:
                first_batch_inputs = inputs.cpu()
                first_batch_targets = targets.cpu()
            if batch_idx == 1:
                second_batch_inputs = inputs.cpu()
                second_batch_targets = targets.cpu()

            #teacher_pred=teacher(inputs)
            #student_pred=student(inputs)
            #teacher_loss+=F.cross_entropy(teacher_pred, targets)
            #student_loss+=F.cross_entropy(student_pred, targets)
            #total += targets.size(0)
            #teacher_correct+=teacher_pred.max(1)[1].eq(targets).sum().item()

            #if batch_idx == 0:
            #    accuracy_f = student_pred.max(1)[1].eq(targets).sum().item()
            
            #student_correct+=student_pred.max(1)[1].eq(targets).sum().item()

            #teacher_vector.append(np.hstack([targets_float.detach().cpu().numpy(), teacher_pred.detach().cpu().numpy()]))
            #student_vector.append(np.hstack([targets_float.detach().cpu().numpy(), student_pred.detach().cpu().numpy()]))

            teacher_logits = teacher(inputs)
            student_logits = student(inputs)

            teacher_loss += criterion(teacher_logits, targets.float())
            student_loss += criterion(student_logits, targets.float())

            teacher_probs = torch.sigmoid(teacher_logits)
            student_probs = torch.sigmoid(student_logits)

            teacher_preds = (teacher_probs > 0.5).float()
            student_preds = (student_probs > 0.5).float()

            total += targets.numel()

            teacher_correct += (teacher_preds == targets).sum().item()
            student_correct += (student_preds == targets).sum().item()
            student_correct_strict += (student_preds == targets).all(dim=1).sum().item()
            total_strict += targets.size(0)
            

            if batch_idx == 0:
                accuracy_f = (student_preds == targets).sum().item()

            teacher_vector.append(np.hstack([targets.detach().cpu().numpy(), teacher_probs.detach().cpu().numpy()]))

            student_vector.append(np.hstack([targets.detach().cpu().numpy(), student_probs.detach().cpu().numpy()]))

       #print("Preds: ", student_preds)
        #print("Targets: ", targets)

    end = time.time()
    step=epoch
    
    avg_teacher_loss = teacher_loss.item() / len(loader)
    avg_student_loss = student_loss.item() / len(loader)
    
    print('[Eval] Epoch: %d | Teacher Test Loss: %.3f | Teacher Test Acc: %.3f | Student Test Loss: %.3f | Student Test Acc: %.3f | Student Test Acc Strict: %.3f | Time: %.3f | '
            % (step, avg_teacher_loss, 100. * teacher_correct / total, avg_student_loss, 100. * student_correct/ total, 100. * student_correct_strict/total_strict,
               end-start))

               
    
    torch.save({'epoch': epoch, 
                'student_acc' :   100. * student_correct / total,
                'epoch': epoch,
                'teacher_vector': teacher_vector,
                'student_vector': student_vector,
                'teacher_loss' :  avg_teacher_loss,
                'teacher_acc' :   100. * teacher_correct / total,
                'student_loss' :  avg_student_loss,
                'student_acc_f' : accuracy_f,
                'student_acc' :   100. * student_correct / total,
                'teacher_state_dict': teacher.state_dict(),
                'student_state_dict': student.state_dict(),
                'batch_inputs': [first_batch_inputs, second_batch_inputs],
                'batch_targets': [first_batch_targets, second_batch_targets]
                }, os.path.join(snapshot_dir, f'data_epoch_{epoch}.pt'))

    student_vector = np.concatenate(student_vector, axis=0)[:, 1:]

    mean_vector = student_vector.mean(axis=0)
    std_vector = student_vector.std(axis=0)
    
    device = torch.device(f"cuda:{args.gpu}")
    template_model = PreResNet(num_classes=8, depth=20, input_size=32).to(device)


    print("Запуск исследования для student...")
    result_student = research(
        model=template_model,
        x_batch_list=[first_batch_inputs.to(device), second_batch_inputs.to(device)],
        y_batch_list=[first_batch_targets.to(device), second_batch_targets.to(device)],
        loss_func=F.cross_entropy,
        snapshot=student.state_dict(),
        device = device)
    
    steps = np.column_stack(list(result_student.values()))[0]
    features = torch.tensor(np.concatenate([steps, mean_vector, std_vector]))
    #wandb.log({'teacher test acc': 100. * teacher_correct / total, 'student test acc': 100. * student_correct / total}, step=step)
    
    return [avg_teacher_loss, avg_student_loss, features]


def train(teacher, student, loader, epoch, args, teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler):
    criterion = nn.BCEWithLogitsLoss()
    teacher.train()
    student.train()
    
    total_teacher_loss = 0
    student_correct, teacher_correct = 0, 0
    total_samples = 0
    start_time = time.time()

    # Проход по части датасета (согласно твоей логике // 10)
    for _ in range(len(loader) // 50):
        for idx, (inputs, targets) in enumerate(loader):
            if idx > 10:
                break
            
            inputs, targets = try_cuda(inputs, targets)
            targets_float = targets.float()

            # --- ОСНОВНОЙ ШАГ: ОБНОВЛЕНИЕ ОБОИХ МОДЕЛЕЙ ---
            teacher_optimizer.zero_grad()
            student_optimizer.zero_grad()

            # Получаем логиты
            teacher_pred = teacher(inputs)
            student_pred = student(inputs)

            if args.loss == 'kl_ce':
                # Считаем лоссы раздельно
                teacher_loss = criterion(teacher_pred, targets_float) + \
                               args.alpha * kl_div_logits(teacher_pred, student_pred.detach(), args.T)
                
                student_loss = criterion(student_pred, targets_float) + \
                               args.alpha * kl_div_logits(student_pred, teacher_pred.detach(), args.T)

                # Объединяем, чтобы PyTorch не ругался на двойной backward
                total_loss = teacher_loss + student_loss
                total_loss.backward()
                
                teacher_optimizer.step()
                student_optimizer.step()
                
                total_teacher_loss += teacher_loss.item()

            # Подсчет точности (Accuracy) для BCE
            with torch.no_grad():
                t_labels = (torch.sigmoid(teacher_pred) > 0.5).float()
                s_labels = (torch.sigmoid(student_pred) > 0.5).float()
                
                teacher_correct += (t_labels == targets).all(dim=1).sum().item()
                student_correct += (s_labels == targets).all(dim=1).sum().item()
                total_samples += targets.size(0)

            # --- ДОПОЛНИТЕЛЬНЫЙ ШАГ: ОБУЧЕНИЕ ТОЛЬКО СТУДЕНТА ---
            for _ in range(args.student_steps_ratio - 1):
                s_inputs, s_targets = get_batch(loader, args.student_index)
                s_inputs, s_targets = try_cuda(s_inputs, s_targets)
                args.student_index = (args.student_index + 1) % len(loader)
                
                # Здесь ТОЖЕ убираем log_softmax, работаем с логитами
                t_logits_extra = teacher(s_inputs)
                s_logits_extra = student(s_inputs)
                
                if args.loss == 'kl_ce':
                    s_loss_extra = criterion(s_logits_extra, s_targets.float()) + \
                                   args.alpha * kl_div_logits(s_logits_extra, t_logits_extra.detach(), args.T)
                    
                    student_optimizer.zero_grad()
                    s_loss_extra.backward()
                    student_optimizer.step()

    end_time = time.time()
    
    # Печать статистики
    print('[Train] Epoch: %d | Teacher Loss: %.3f | Teacher Acc: %.3f | Student Acc: %.3f | Time: %.3f | Teacher lr=%.4f | Student lr=%.4f |'
            % (epoch, total_teacher_loss / (len(loader)//10 * 11), 100. * teacher_correct / total_samples, 
               100. * student_correct / total_samples, end_time - start_time, teacher_scheduler.get_last_lr()[0], student_scheduler.get_last_lr()[0] ))
    
    teacher_scheduler.step()
    student_scheduler.step()


parser = argparse.ArgumentParser(description='PyTorch Image Classification')
parser.add_argument('--exp_name', type=str, default='LoT_ResNet')
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--models_num', type=int, default=2)
parser.add_argument('--detach', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T', type=float, default=1.5)
parser.add_argument('--student_index', type=int, default=0, help='an independent index for student updating')
parser.add_argument('--student_steps_ratio', type=int, default=4)
parser.add_argument('--loss', type=str, default='kl_ce', choices=['kl', 'kl_ce', 'symmetric_kl', 'symmetric_kl_ce'])
# original
#parser.add_argument('--num_classes', type=int, default= 100, choices = [100, 10, 9])
parser.add_argument('--dataset', type=str, default='cifar100', choices = ['cifar10', 'cifar100', 'mydataset'])
parser.add_argument('--datadir', type=str, default='data', help='data directory')
parser.add_argument('--input_size', type=int, default=32, help='image input size')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--depth_list', type=str, default='110_20', help='resnet model depth list')
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--scheduler', type=str, default='cosine')
parser.add_argument('--epochs', type=int, default=180)
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default='ckpt/LoT_ResNet'+randomhash+'CIFAR.pt', help='path to save the final model')
args = parser.parse_args()
#print(json.dumps(vars(args), indent=4))


def main():
    try:

        model_gru = lr_policy_training()
        
        features_list = []

        config=configparser.ConfigParser()
        config.read('key.config')
        #wandb_username=config.get('WANDB', 'USER_NAME')
        #wandb_key=config.get('WANDB', 'API_KEY')        
        #wandb.login(key=wandb_key)
        #wandb.init(project='LoT_ResNet_CIFAR_'+args.dataset, entity=wandb_username, name=args.exp_name)
        print(args.depth_list)
        depth_list = [int(number) for number in args.depth_list.split('_')]
        print(depth_list)
        depth_list=''.join(char for char in str(args.depth_list) if char.isdigit())
        print(depth_list)
        depth_list=[int(depth_list[2*i:2*i+2]) for i in range(len(depth_list)//2)]
        print(depth_list)
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(int(args.gpu))
        train_loader, test_loader = get_torch_dataset(args)

        # init teacher
        torch.manual_seed(args.seed)
        print('teacher depth:', depth_list[0])
       #teacher=PreResNet(num_classes=args.num_classes, depth=depth_list[0], input_size=args.input_size)
        teacher=PreResNet(num_classes=8, depth=depth_list[0], input_size=args.input_size)
        teacher, =try_cuda(teacher)

        # init student
        torch.manual_seed(args.seed+1)
        print('student depth:', depth_list[1])
        #student=PreResNet(num_classes=args.num_classes, depth=depth_list[1], input_size=args.input_size)
        student=PreResNet(num_classes=8, depth=depth_list[1], input_size=args.input_size)
        student, =try_cuda(student)
        args.student_index=0

        total_params = sum(p.numel() for p in teacher.parameters())
        print(f"Total number of teacher parameters: {total_params:,}")
        total_params = sum(p.numel() for p in student.parameters())
        print(f"Total number of student parameters: {total_params:,}")

        epoch = 0

        snapshot_dir = '/kaggle/working/snapshots/' + args.exp_name
        os.makedirs(snapshot_dir, exist_ok=True)
        print(f"Snapshots will be saved to: {snapshot_dir}")

        print(f"==== train and evaluate unequal restart ====")
        if args.optimizer=='sgd':
            #teacher_optimizer = torch.optim.SGD(lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True, params=teacher.parameters())
            #student_optimizer = torch.optim.SGD(lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True, params=student.parameters())
            teacher_optimizer = torch.optim.SGD(lr=0.02, params=teacher.parameters())
            student_optimizer = torch.optim.SGD(lr=0.02, params=student.parameters())
        if args.scheduler=='cosine':
            teacher_scheduler = torch.optim.lr_scheduler.LambdaLR(teacher_optimizer, lr_lambda=lambda epoch: 1.0)
            student_scheduler = torch.optim.lr_scheduler.LambdaLR(student_optimizer, lr_lambda=lambda epoch: 1.0)
            #teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=teacher_optimizer)
            #student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=student_optimizer)
        if args.scheduler == 'custom':
            lrs = [float(lrs[i]) for i in range(len(lrs))]
            student_scheduler = MyLRScheduler(student_optimizer, lrs)
            teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=teacher_optimizer)
        if args.scheduler == 'dynamic':
            student_scheduler = DynamicScheduler(student_optimizer, init_lr=0.01)
            teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=args.epochs, eta_min=0, optimizer=teacher_optimizer)
            
        features_list.append(evaluate(teacher, student, test_loader, 0)[2])
        for epoch in range(1, args.epochs+1):
            train(teacher, student, train_loader, epoch, args, teacher_optimizer, student_optimizer, teacher_scheduler, student_scheduler)
            features_list.append(evaluate(teacher, student, test_loader, epoch)[2])

            if len(features_list) >= 1:
                model_gru.eval()

                #with torch.no_grad():
                #    x_batch_gru = torch.stack(features_list[-15:]).float().to(device)
                    
                #    x_batch_gru = x_batch_gru.unsqueeze(0)
                #    print(x_batch_gru.shape)
                #    p = model_gru(x_batch_gru).cpu().numpy()
                #    if p <= 0:
                #        p = torch.tensor(0.001)
                #print(p)
                #if epoch <= 1:
                #    student_scheduler.set_lr(np.cos(epoch/180))
                #else:
                #    print("goooool")
                #    student_scheduler.set_lr(p.item())
            #student_scheduler.step()

            torch.save({'lr' : student_scheduler.get_lr()}, os.path.join(snapshot_dir, f'lr_data_epoch_{epoch}.pt'))

        torch.save(teacher.state_dict(), args.save+'_teacher.pt')
        torch.save(student.state_dict(), args.save+'_student.pt')
        print('ckpt location:', args.save)
        #wandb.finish()

    except Exception:
        logging.error(traceback.format_exc())
        return float('NaN')


if __name__ == '__main__':
    main()
